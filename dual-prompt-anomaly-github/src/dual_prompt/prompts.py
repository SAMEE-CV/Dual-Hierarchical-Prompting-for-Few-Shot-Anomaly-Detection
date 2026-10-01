"""Fixed, auditable hard-prompt templates for MVTec AD."""

from __future__ import annotations

from dataclasses import dataclass

NORMAL_TEMPLATES = (
    "a photo of a flawless {category}",
    "a photo of a defect-free {category}",
    "a clean and undamaged {category}",
    "a normal {category} with no manufacturing defect",
    "an intact {category} in perfect condition",
    "a high-quality industrial photo of a good {category}",
    "a {category} with a regular surface and correct structure",
)

DEFECT_TEMPLATES = (
    "a photo of a defective {category}",
    "a damaged {category} with a manufacturing defect",
    "an anomalous {category} with an irregular surface",
    "a flawed {category} with visible damage",
    "a contaminated, scratched, cracked, or deformed {category}",
    "a low-quality industrial photo of a bad {category}",
    "a {category} with a structural or surface anomaly",
)

CATEGORY_ALIASES = {
    "carpet": "carpet texture",
    "grid": "metal grid texture",
    "leather": "leather texture",
    "tile": "ceramic tile",
    "wood": "wood texture",
    "bottle": "bottle",
    "cable": "electrical cable",
    "capsule": "capsule",
    "hazelnut": "hazelnut",
    "metal_nut": "metal nut",
    "pill": "pharmaceutical pill",
    "screw": "screw",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "zipper": "zipper",
}

CATEGORY_DEFECT_PHRASES = {
    "carpet": (
        "an abnormal color patch",
        "a cut or hole",
        "metal contamination",
        "a loose or embedded thread",
    ),
    "grid": (
        "a bent or broken grid structure",
        "glue contamination",
        "metal contamination",
        "an embedded thread",
    ),
    "leather": (
        "abnormal coloration",
        "a cut or puncture",
        "a fold",
        "glue contamination",
    ),
    "tile": (
        "a crack",
        "a glue strip",
        "a gray stroke",
        "oil contamination",
        "a rough surface patch",
    ),
    "wood": (
        "abnormal coloration",
        "a hole",
        "liquid contamination",
        "a scratch",
        "multiple combined defects",
    ),
    "bottle": (
        "a large break",
        "a small break",
        "surface contamination",
    ),
    "cable": (
        "a bent wire",
        "swapped or missing cable components",
        "cut inner or outer insulation",
        "missing wire",
        "punctured insulation",
        "multiple combined defects",
    ),
    "capsule": (
        "a crack",
        "a faulty imprint",
        "a puncture",
        "a scratch",
        "a squeezed shape",
    ),
    "hazelnut": (
        "a crack",
        "a cut",
        "a hole",
        "an abnormal printed mark",
    ),
    "metal_nut": (
        "a bent shape",
        "abnormal coloration",
        "an incorrect orientation",
        "a scratch",
    ),
    "pill": (
        "abnormal coloration",
        "contamination",
        "a crack or scratch",
        "a faulty imprint",
        "an incorrect pill type",
        "multiple combined defects",
    ),
    "screw": (
        "a manipulated front",
        "a scratched head or neck",
        "damage on the thread side",
        "damage on the thread top",
    ),
    "toothbrush": ("a defective brush head",),
    "transistor": (
        "a bent lead",
        "a cut lead",
        "a damaged case",
        "a misplaced component",
    ),
    "zipper": (
        "broken or split teeth",
        "a damaged fabric border or interior",
        "a rough surface",
        "squeezed teeth",
        "multiple combined defects",
    ),
}


@dataclass(frozen=True, slots=True)
class PromptSet:
    normal: tuple[str, ...]
    defective: tuple[str, ...]


def prompts_for_category(category: str) -> PromptSet:
    """Instantiate the disclosed prompt ensemble for a category."""
    readable_name = CATEGORY_ALIASES.get(category, category.replace("_", " "))
    category_specific = tuple(
        f"a {readable_name} with {phrase}" for phrase in CATEGORY_DEFECT_PHRASES.get(category, ())
    )
    return PromptSet(
        normal=tuple(template.format(category=readable_name) for template in NORMAL_TEMPLATES),
        defective=tuple(template.format(category=readable_name) for template in DEFECT_TEMPLATES)
        + category_specific,
    )
