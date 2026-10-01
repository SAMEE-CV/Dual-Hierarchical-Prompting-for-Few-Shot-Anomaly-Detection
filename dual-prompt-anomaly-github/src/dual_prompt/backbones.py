"""Frozen CLIP backbones with explicit hierarchical prompt injection."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PromptableBackbone(nn.Module, ABC):
    visual_width: int
    embed_dim: int
    num_blocks: int
    grid_size: tuple[int, int]
    logit_scale: float
    text_width: int
    text_num_blocks: int

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        """Return one L2-normalized embedding per text."""

    @abstractmethod
    def encode_texts_with_prompts(
        self, texts: Sequence[str], prompts: Mapping[int, Tensor]
    ) -> Tensor:
        """Encode text while replacing context slots at selected transformer depths."""

    @abstractmethod
    def encode_image_with_prompts(
        self, images: Tensor, prompts: Mapping[int, Tensor]
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Return normalized global embeddings and selected patch features."""


class OpenClipBackbone(PromptableBackbone):
    """Adapter around OpenCLIP's VisionTransformer.

    Prompt tokens are inserted before a selected residual block and removed
    immediately afterwards, matching Eq. (5) in the manuscript.
    """

    def __init__(self, model_name: str, pretrained: str, device: torch.device) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as error:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "Install open-clip-torch to use a production CLIP backbone"
            ) from error

        self.model = open_clip.create_model(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        visual = self.model.visual
        required = ("_embeds", "transformer", "ln_post", "proj")
        missing = [name for name in required if not hasattr(visual, name)]
        if missing:
            raise TypeError(
                f"Backbone {model_name!r} is not a supported OpenCLIP VisionTransformer; "
                f"missing {missing}"
            )
        self.visual_width = int(visual.conv1.out_channels)
        self.embed_dim = int(self.model.text_projection.shape[-1])
        self.num_blocks = len(visual.transformer.resblocks)
        self.text_width = int(self.model.token_embedding.embedding_dim)
        self.text_num_blocks = len(self.model.transformer.resblocks)
        grid = visual.grid_size
        self.grid_size = (int(grid[0]), int(grid[1])) if isinstance(grid, tuple) else (grid, grid)
        self.logit_scale = float(self.model.logit_scale.exp().detach().cpu())

    def train(self, mode: bool = True) -> OpenClipBackbone:
        # The backbone always remains in evaluation mode; gradients can still
        # flow through it into inserted prompts.
        super().train(False)
        self.model.eval()
        return self

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(texts)).to(device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens, normalize=True)
        return embeddings.float()

    def encode_texts_with_prompts(
        self, texts: Sequence[str], prompts: Mapping[int, Tensor]
    ) -> Tensor:
        if not prompts:
            return self.encode_texts(texts)
        prompt_lengths = {prompt.shape[-2] for prompt in prompts.values()}
        if len(prompt_lengths) != 1:
            raise ValueError("All hierarchical text prompts must use the same token length")
        prompt_length = next(iter(prompt_lengths))
        if prompt_length + 2 > self.model.context_length:
            raise ValueError("Text prompt length leaves no room for the hard description")

        # The placeholder tokens reserve stable context positions. At each
        # selected depth those positions are replaced by the layer's learned
        # prompt, preserving CLIP's original sequence length and causal mask.
        prefix = " ".join(["X"] * prompt_length)
        device = next(self.model.parameters()).device
        tokens = self.tokenizer([f"{prefix} {text}" for text in texts]).to(device)
        cast_dtype = self.model.transformer.get_cast_dtype()
        position = self.model.positional_embedding.to(cast_dtype)
        x = self.model.token_embedding(tokens).to(cast_dtype) + position

        for index, block in enumerate(self.model.transformer.resblocks):
            if index in prompts:
                prompt = prompts[index].to(device=x.device, dtype=x.dtype)
                if prompt.ndim == 2:
                    prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
                contextual = prompt + position[1 : 1 + prompt_length].unsqueeze(0)
                x = torch.cat((x[:, :1], contextual, x[:, 1 + prompt_length :]), dim=1)
            x = block(x, attn_mask=self.model.attn_mask)

        x = self.model.ln_final(x)
        from open_clip.transformer import text_global_pool

        x = text_global_pool(
            x,
            tokens,
            self.model.text_pool_type,
            eos_token_id=getattr(self.model, "text_eos_id", None),
        )
        projection = self.model.text_projection
        if isinstance(projection, nn.Linear):
            x = projection(x)
        elif projection is not None:
            x = x @ projection
        return F.normalize(x.float(), dim=-1)

    def encode_image_with_prompts(
        self, images: Tensor, prompts: Mapping[int, Tensor]
    ) -> tuple[Tensor, dict[int, Tensor]]:
        visual = self.model.visual
        x = visual._embeds(images.to(dtype=visual.conv1.weight.dtype))
        features: dict[int, Tensor] = {}

        for index, block in enumerate(visual.transformer.resblocks):
            if index in prompts:
                prompt = prompts[index].to(device=x.device, dtype=x.dtype)
                if prompt.ndim == 2:
                    prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
                prompt_length = prompt.shape[1]
                augmented = torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1)
                augmented = block(augmented)
                x = torch.cat((augmented[:, :1], augmented[:, 1 + prompt_length :]), dim=1)
                features[index] = x[:, 1:].float()
            else:
                x = block(x)

        if hasattr(visual, "_pool"):
            pooled, _ = visual._pool(x)
        else:  # OpenCLIP compatibility fallback
            pooled = x[:, 0]
        pooled = visual.ln_post(pooled)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return F.normalize(pooled.float(), dim=-1), features


class _ToyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + torch.tanh(self.projection(self.norm(tokens)))


class ToyBackbone(PromptableBackbone):
    """Small deterministic frozen backbone for tests and smoke runs only."""

    def __init__(
        self, image_size: int = 32, patch_size: int = 8, width: int = 32, embed_dim: int = 24
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Toy image_size must be divisible by patch_size")
        self.visual_width = width
        self.embed_dim = embed_dim
        self.num_blocks = 3
        self.text_width = width
        self.text_num_blocks = 3
        side = image_size // patch_size
        self.grid_size = (side, side)
        self.logit_scale = 10.0
        self.patch = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, 1 + side * side, width) * 0.01)
        self.blocks = nn.ModuleList(_ToyBlock(width) for _ in range(self.num_blocks))
        self.visual_projection = nn.Linear(width, embed_dim, bias=False)
        generator = torch.Generator().manual_seed(1234)
        self.text_projection = nn.Parameter(torch.randn(256, embed_dim, generator=generator) * 0.1)
        self.text_prompt_projection = nn.Parameter(
            torch.randn(width, embed_dim, generator=generator) * 0.05
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> ToyBackbone:
        super().train(False)
        return self

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = torch.tensor(
                list(digest) * 8, dtype=torch.float32, device=self.text_projection.device
            )
            values = values[:256] / 127.5 - 1.0
            rows.append(values)
        return F.normalize(torch.stack(rows) @ self.text_projection, dim=-1)

    def encode_texts_with_prompts(
        self, texts: Sequence[str], prompts: Mapping[int, Tensor]
    ) -> Tensor:
        base = self.encode_texts(texts)
        effect = torch.zeros(self.embed_dim, device=base.device, dtype=base.dtype)
        for layer in sorted(prompts):
            prompt = prompts[layer]
            effect = effect + prompt.mean(dim=-2) @ self.text_prompt_projection
        return F.normalize(base + 0.05 * effect.unsqueeze(0), dim=-1)

    def encode_image_with_prompts(
        self, images: Tensor, prompts: Mapping[int, Tensor]
    ) -> tuple[Tensor, dict[int, Tensor]]:
        patches = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(images.shape[0], -1, -1)
        x = torch.cat((cls, patches), dim=1) + self.position
        features: dict[int, Tensor] = {}
        for index, block in enumerate(self.blocks):
            if index in prompts:
                prompt = prompts[index]
                if prompt.ndim == 2:
                    prompt = prompt.unsqueeze(0).expand(images.shape[0], -1, -1)
                length = prompt.shape[1]
                augmented = block(torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1))
                x = torch.cat((augmented[:, :1], augmented[:, 1 + length :]), dim=1)
                features[index] = x[:, 1:]
            else:
                x = block(x)
        global_embedding = F.normalize(self.visual_projection(x[:, 0]), dim=-1)
        return global_embedding, features


def create_backbone(
    name: str, pretrained: str, image_size: int, device: torch.device
) -> PromptableBackbone:
    if name.lower() == "toy":
        return ToyBackbone(image_size=image_size).to(device)
    return OpenClipBackbone(name, pretrained, device).to(device)


def validate_prompt_layers(backbone: PromptableBackbone, layers: Sequence[int]) -> None:
    invalid = [layer for layer in layers if layer < 0 or layer >= backbone.num_blocks]
    if invalid:
        raise ValueError(
            f"Prompt layers {invalid} are invalid for a {backbone.num_blocks}-block backbone"
        )
    expected_patches = math.prod(backbone.grid_size)
    if expected_patches < 1:
        raise ValueError("Backbone returned an invalid patch grid")
