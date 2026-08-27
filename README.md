# SHIN Multimodal Prompt Models

Code for three EEG-fNIRS prompt-fusion models evaluated on the SHIN dataset.
Only the selected main-model implementations and their corresponding configurations are included; diagnostic caches, trained weights, datasets, run outputs, and ablation-only entry points are excluded.

## Models

1. `01-sgformer-graph-prompt`: CBraMod with an SGFormer graph prompt and gated K/V injection.
2. `02-nve-dual-branch-prompt`: CBraMod with an explicit NVE-SGFormer spatial prompt and an independent temporal prompt branch.
3. `03-fnirst-multi-prompt`: fNIRS-T channel/patch/joint prompts with a universal final-layer cross-attention adapter.

## Setup

Install the dependencies in `requirements.txt`, obtain the SHIN dataset and CBraMod pretrained weights separately, then update the placeholder paths in the selected JSON configuration.

Each model directory is self-contained and includes its model implementation, training entry point, SHIN data loader, and selected MI/MA configurations.

## Data and checkpoints

The SHIN dataset, pretrained CBraMod checkpoint, learned reference heads, and experiment outputs are not redistributed in this repository. Follow the original dataset and model licenses when obtaining them.

## Reproducibility note

Model selection in the reported experiments used fusion validation accuracy. Test data were not used to select checkpoints.
