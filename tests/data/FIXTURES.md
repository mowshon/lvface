# Test fixture provenance

The committed fixtures contain no photographs or externally sourced biometric data.

- `golden_crop_T.npy` is a synthetic 112×112 RGB color pattern created for this project to
  exercise channel order, normalization, alignment, and inference deterministically.
- `golden_pre_T.npy` is the frozen preprocessing result produced from `golden_crop_T.npy`.
- `golden_raw_T.npy` is the frozen raw output from the pinned T ONNX model. Its model and
  preprocessing provenance is recorded in `golden_meta.json`.

The local celebrity photos under `examples/assets/` are development-only inputs. They are
git-ignored and excluded from release artifacts.
