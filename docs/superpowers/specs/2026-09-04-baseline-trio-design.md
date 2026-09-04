# SegNeXt, RepSTDC, Mask2Former baseline design

## Goal
Add three paper baselines to the existing OpenEarthMap training flow without reimplementing their architectures: SegNeXt-T, RepSTDC-CA, and Mask2Former-Swin-Tiny.

## Decisions
- Keep the existing 9-class OpenEarthMap split, evaluator, optimizer/checkpoint/W&B flow.
- SegNeXt: reuse MMSegmentation MSCAN + LightHamHead with the official SegNeXt-T settings and official MSCAN-T pretrained checkpoint.
- RepSTDC: reuse the pinned official RepSTDC `mmseg_geo` implementation, but expose 9 output classes instead of the upstream OEM recipe's 8 foreground classes / `reduce_zero_label=True`.
- Mask2Former: reuse Hugging Face Transformers already in the project. Evaluation converts query class/mask scores to differentiable dense semantic logits. Training uses Mask2Former's native Hungarian mask-classification loss because replacing it with dense CE+Dice would no longer be faithful to Mask2Former.
- OpenMMLab models run in a dedicated compatible environment created by `scripts/setup_openmmlab_baselines.sh`; do not disturb `work-env` or compile MMCV against its newer Torch/CUDA stack.
- Add the three models to the normal registry/CLI so outer training, validation, test, checkpointing, W&B, and error analysis remain shared.

## Safety / compatibility
The RepSTDC upstream checkout stays pinned. No upstream training script is invoked, so its hard-coded notification side effect is avoided. The dedicated OpenMMLab environment is used only for SegNeXt/RepSTDC imports.
