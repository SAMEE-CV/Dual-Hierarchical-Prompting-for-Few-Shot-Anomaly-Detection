# Methodology and equation-to-code map

This document maps the formal method in the supplied manuscript to the implementation. Symbols follow the paper where practical.

## Hard text anchors - Eq. (3)

`prompts.py` discloses every normal and defective text template, including MVTec category-specific defect descriptions. `DualPromptModel` encodes each description with the frozen text encoder, L2-normalizes each embedding, averages by state, and L2-normalizes again. These fixed anchors support the formal `visual_only` mode and initialize the semantic conditioning used by the intended dual mode.

## Hierarchical textual prompts

In default `dual` mode, each selected text-transformer depth owns a trainable prompt bank and sigmoid gate. Placeholder tokens reserve stable context positions after CLIP's start token. Before each selected frozen text block, that layer's soft prompt replaces the reserved embeddings while preserving the original sequence length, positional embeddings, and causal attention mask. The resulting normal and defective anchors are recomputed during adaptation and cached during inference.

The text insertion rule is a MaPLe-style implementation of the architecture shown in Figure 2. The manuscript does not provide a corresponding equation, so the prompt positions, token replacement rule, and initialization are explicitly recorded as implementation choices.

## Text-conditioned visual prompts - Eq. (4)

For each selected layer `j`, the model owns:

- a prompt bank with shape `[prompt_length, visual_width]`;
- a linear map from the concatenated two-anchor vector to a prompt-shaped offset;
- a scalar gate initialized to zero.

In dual mode, the corresponding textual prompt is additionally projected token-by-token into the visual token width. The generated visual prompt is `sigmoid(gate) * (bank + anchor_offset + coupled_text_prompt)`. In `visual_only` mode the coupling term is omitted, reproducing Eq. (4). Prompt banks and linear weights use a zero-mean normal initialization with standard deviation 0.02; biases are zero.

## Transformer insertion - Eq. (5)

Immediately before a selected residual block, prompt tokens are inserted after the class token and before patch tokens. After that block, output prompt tokens are removed. The affected class and patch tokens continue to the next block. This prevents cumulative sequence growth.

## Normal-only optimization - Eqs. (6)-(8)

The final class token is projected into CLIP's shared embedding space and normalized. Fixed CLIP logit scaling is applied to similarities with the normal and defective anchors. Every support image has target index zero (normal). The objective is two-class cross-entropy plus squared text/visual prompt-bank and gate regularization. Text prompts, visual prompts, coupling layers, conditioners, and gates receive updates; all CLIP parameters remain frozen.

## Normal memories - Eq. (9)

After the final optimizer step, the model recomputes prompts, extracts normalized support patch features, and builds one memory per selected layer. If a layer has more than `memory.max_size` entries, a seeded random subset is retained. The reduction rule is reported in `run_metadata.json`.

## Localization - Eqs. (10)-(11)

Each normalized query patch is compared with all retained normal features in its layer. The anomaly response is half of one minus the mean cosine similarity of the `k` nearest entries. Responses are reshaped to the backbone patch grid, bilinearly resized, and combined with fixed nonnegative layer weights that sum to one.

## Image scoring - Eqs. (12)-(14)

The semantic score is the defective-class probability. The spatial score is the mean of the largest configured number of heatmap pixels. Their convex combination is the reported image anomaly score.

## Manuscript ambiguity

The title, introductory text, and architecture figure describe soft textual prompts, but the formal equations define fixed text anchors and trainable visual prompts only. Section 3.3 explicitly says only visual prompt banks, text-conditioning projections, and gates are optimized. No text-prompt token count or insertion equation is supplied. This repository therefore exposes two named modes: `dual` implements the user's stated MaPLe-style intent and Figure 2, while `visual_only` follows the narrower formal equations. Exact parity still requires official author code.
