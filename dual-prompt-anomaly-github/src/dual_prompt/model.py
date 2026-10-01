"""Paper-aligned Dual-Prompt model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import PromptableBackbone, validate_prompt_layers
from .prompts import PromptSet


def _aggregate_text_embeddings(backbone: PromptableBackbone, texts: Sequence[str]) -> Tensor:
    embeddings = F.normalize(backbone.encode_texts(texts), dim=-1)
    return F.normalize(embeddings.mean(dim=0), dim=-1)


def _aggregate_prompted_text_embeddings(
    backbone: PromptableBackbone, texts: Sequence[str], prompts: dict[int, Tensor]
) -> Tensor:
    embeddings = F.normalize(backbone.encode_texts_with_prompts(texts, prompts), dim=-1)
    return F.normalize(embeddings.mean(dim=0), dim=-1)


class DualPromptModel(nn.Module):
    """MaPLe-style dual hierarchical prompts over a frozen CLIP backbone.

    ``dual`` mode implements trainable text prompts at multiple text-transformer
    depths and couples them into the hierarchical visual prompts. The
    ``visual_only`` mode follows the narrower formal equations in the supplied
    manuscript and is retained as a documented ablation.
    """

    def __init__(
        self,
        backbone: PromptableBackbone,
        prompt_set: PromptSet,
        prompt_layers: Sequence[int],
        prompt_length: int = 4,
        conditioner: str = "linear",
        prompt_mode: str = "dual",
        text_prompt_layers: Sequence[int] | None = None,
        text_prompt_length: int | None = None,
    ) -> None:
        super().__init__()
        validate_prompt_layers(backbone, prompt_layers)
        if conditioner != "linear":
            raise ValueError("Only the paper's reported linear conditioner is currently supported")
        if prompt_mode not in {"dual", "visual_only"}:
            raise ValueError("prompt_mode must be 'dual' or 'visual_only'")
        self.backbone = backbone
        self.prompt_mode = prompt_mode
        self.prompt_layers = tuple(int(layer) for layer in prompt_layers)
        self.prompt_length = prompt_length
        self.normal_texts = tuple(prompt_set.normal)
        self.defective_texts = tuple(prompt_set.defective)

        if text_prompt_layers is None:
            text_prompt_layers = prompt_layers
        if text_prompt_length is None:
            text_prompt_length = prompt_length
        self.text_prompt_layers = tuple(int(layer) for layer in text_prompt_layers)
        self.text_prompt_length = text_prompt_length
        if prompt_mode == "dual":
            invalid_text_layers = [
                layer
                for layer in self.text_prompt_layers
                if layer < 0 or layer >= backbone.text_num_blocks
            ]
            if invalid_text_layers:
                raise ValueError(
                    f"Text prompt layers {invalid_text_layers} are invalid for a "
                    f"{backbone.text_num_blocks}-block text encoder"
                )
            if len(self.text_prompt_layers) != len(self.prompt_layers):
                raise ValueError("Dual mode requires one text prompt depth per visual prompt depth")
            if self.text_prompt_length != self.prompt_length:
                raise ValueError("Dual mode requires equal text and visual prompt lengths")

        normal = _aggregate_text_embeddings(backbone, self.normal_texts)
        defective = _aggregate_text_embeddings(backbone, self.defective_texts)
        self.register_buffer("fixed_text_anchors", torch.stack((normal, defective)))

        self.prompt_banks = nn.ParameterDict()
        self.conditioners = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        self.text_prompt_banks = nn.ParameterDict()
        self.text_gates = nn.ParameterDict()
        self.text_to_vision = nn.ModuleDict()
        input_width = 2 * backbone.embed_dim
        output_width = prompt_length * backbone.visual_width
        for depth_index, layer in enumerate(self.prompt_layers):
            key = str(layer)
            bank = nn.Parameter(torch.empty(prompt_length, backbone.visual_width))
            nn.init.normal_(bank, std=0.02)
            self.prompt_banks[key] = bank
            projection = nn.Linear(input_width, output_width)
            nn.init.normal_(projection.weight, std=0.02)
            nn.init.zeros_(projection.bias)
            self.conditioners[key] = projection
            self.gates[key] = nn.Parameter(torch.zeros(()))

            if prompt_mode == "dual":
                text_layer = self.text_prompt_layers[depth_index]
                text_key = str(text_layer)
                text_bank = nn.Parameter(torch.empty(self.text_prompt_length, backbone.text_width))
                nn.init.normal_(text_bank, std=0.02)
                self.text_prompt_banks[text_key] = text_bank
                self.text_gates[text_key] = nn.Parameter(torch.zeros(()))
                coupling = nn.Linear(backbone.text_width, backbone.visual_width)
                nn.init.normal_(coupling.weight, std=0.02)
                nn.init.zeros_(coupling.bias)
                self.text_to_vision[key] = coupling

        self._cached_text_anchors: Tensor | None = None

    def train(self, mode: bool = True) -> DualPromptModel:
        super().train(mode)
        self.backbone.eval()
        if mode:
            self._cached_text_anchors = None
        return self

    def generated_text_prompts(self) -> dict[int, Tensor]:
        if self.prompt_mode != "dual":
            return {}
        return {
            layer: torch.sigmoid(self.text_gates[str(layer)]) * self.text_prompt_banks[str(layer)]
            for layer in self.text_prompt_layers
        }

    def current_text_anchors(self) -> Tensor:
        if self.prompt_mode == "visual_only":
            return self.fixed_text_anchors
        if not self.training and self._cached_text_anchors is not None:
            return self._cached_text_anchors
        prompts = self.generated_text_prompts()
        normal = _aggregate_prompted_text_embeddings(self.backbone, self.normal_texts, prompts)
        defective = _aggregate_prompted_text_embeddings(
            self.backbone, self.defective_texts, prompts
        )
        anchors = torch.stack((normal, defective))
        if not self.training:
            self._cached_text_anchors = anchors.detach()
        return anchors

    def generated_visual_prompts(
        self, text_anchors: Tensor, text_prompts: dict[int, Tensor]
    ) -> dict[int, Tensor]:
        semantic_context = text_anchors.flatten()
        prompts: dict[int, Tensor] = {}
        for depth_index, layer in enumerate(self.prompt_layers):
            key = str(layer)
            offset = self.conditioners[key](semantic_context).view(
                self.prompt_length, self.backbone.visual_width
            )
            coupled = 0.0
            if self.prompt_mode == "dual":
                text_layer = self.text_prompt_layers[depth_index]
                coupled = self.text_to_vision[key](text_prompts[text_layer])
            prompts[layer] = torch.sigmoid(self.gates[key]) * (
                self.prompt_banks[key] + offset + coupled
            )
        return prompts

    def generated_prompts(self) -> dict[int, Tensor]:
        """Return the current visual prompts for inspection and compatibility."""
        text_prompts = self.generated_text_prompts()
        return self.generated_visual_prompts(self.current_text_anchors(), text_prompts)

    def prompt_regularization(self) -> Tensor:
        visual = sum(
            self.prompt_banks[str(layer)].square().sum() + self.gates[str(layer)].square()
            for layer in self.prompt_layers
        )
        if self.prompt_mode == "visual_only":
            return visual
        textual = sum(
            self.text_prompt_banks[str(layer)].square().sum() + self.text_gates[str(layer)].square()
            for layer in self.text_prompt_layers
        )
        return visual + textual

    def forward(self, images: Tensor) -> tuple[Tensor, dict[int, Tensor]]:
        text_prompts = self.generated_text_prompts()
        text_anchors = self.current_text_anchors()
        visual_prompts = self.generated_visual_prompts(text_anchors, text_prompts)
        global_features, patch_features = self.backbone.encode_image_with_prompts(
            images, visual_prompts
        )
        logits = self.backbone.logit_scale * global_features @ text_anchors.T
        return logits, patch_features

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Return a compact state dict that excludes the frozen backbone."""
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        current = self.state_dict()
        unknown = set(state) - set(current)
        if unknown:
            raise RuntimeError(f"Unknown checkpoint keys: {sorted(unknown)}")
        missing_trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name not in state
        ]
        if missing_trainable:
            raise RuntimeError(f"Checkpoint is missing trainable keys: {missing_trainable}")
        self.load_state_dict(state, strict=False)
