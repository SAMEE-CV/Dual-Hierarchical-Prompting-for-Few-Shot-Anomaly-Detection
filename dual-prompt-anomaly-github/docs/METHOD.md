# Method

Dual-Prompt adapts a frozen CLIP ViT-B/16 using a small set of trainable parameters:

1. Normal and defective hard descriptions provide semantic context.
2. Hierarchical textual prompts adapt selected CLIP text blocks.
3. Text prompts are projected into coupled visual prompts at matching ViT blocks.
4. Normal patch features create one memory bank per selected visual layer.
5. Image-text similarity and memory distances produce image scores and anomaly maps.

Training uses only normal support images. CLIP parameters remain frozen.

The manuscript describes dual textual/visual prompts in its title and architecture, while its formal equations specify only the visual branch. The default `dual` mode enables both branches; `visual_only` is available as an ablation.
