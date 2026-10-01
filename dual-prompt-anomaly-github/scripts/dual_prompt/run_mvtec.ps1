param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$Category = "all",
    [int]$Epochs = 50,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

python -m dual_prompt train `
    --config configs/trainers/DualPrompt/vit_b16_4shot.yaml `
    --data-root $DataRoot `
    --category $Category `
    --epochs $Epochs `
    --device $Device

