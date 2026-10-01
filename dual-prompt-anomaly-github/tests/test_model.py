import torch

from dual_prompt.backbones import ToyBackbone
from dual_prompt.memory import FeatureMemory, anomaly_map, image_anomaly_score
from dual_prompt.model import DualPromptModel
from dual_prompt.prompts import prompts_for_category


def test_prompted_forward_shapes_and_frozen_backbone() -> None:
    backbone = ToyBackbone(image_size=32)
    model = DualPromptModel(backbone, prompts_for_category("bottle"), [0, 1, 2], 2)
    logits, features = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 2)
    assert set(features) == {0, 1, 2}
    assert all(feature.shape == (2, 16, 32) for feature in features.values())
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.prompt_banks.parameters())
    assert any(parameter.requires_grad for parameter in model.text_prompt_banks.parameters())
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.text_prompt_banks.parameters())


def test_mvtec_prompts_include_category_specific_defects() -> None:
    bottle = prompts_for_category("bottle")
    assert any("large break" in prompt for prompt in bottle.defective)
    assert len(bottle.defective) > 7


def test_memory_scoring_is_bounded() -> None:
    backbone = ToyBackbone(image_size=32)
    model = DualPromptModel(backbone, prompts_for_category("bottle"), [0, 1, 2], 2)
    logits, features = model(torch.randn(2, 3, 32, 32))
    memory = FeatureMemory(
        banks={
            layer: torch.nn.functional.normalize(value[0], dim=-1)
            for layer, value in features.items()
        },
        grid_size=(4, 4),
    )
    pixel_map = anomaly_map(features, memory, (32, 32), [0.3, 0.4, 0.3], 1, 8)
    score, semantic, spatial = image_anomaly_score(logits, pixel_map, 0.5, 10)
    assert pixel_map.shape == (2, 1, 32, 32)
    assert torch.all((pixel_map >= 0) & (pixel_map <= 1))
    assert torch.all((score >= 0) & (score <= 1))
    assert semantic.shape == spatial.shape == (2,)
