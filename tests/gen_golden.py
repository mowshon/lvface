"""Generate frozen parity fixtures using the legacy ONNX inferencer."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from inference_onnx import LVFaceONNXInferencer  # noqa: E402
from lvface.registry import DEFAULT_MODEL, MODELS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    data_dir = PACKAGE_ROOT / "tests" / "data"
    crop_path = data_dir / "golden_crop_T.npy"
    crop = np.load(crop_path)
    legacy = LVFaceONNXInferencer(str(args.model_path), use_gpu=False)
    preprocessed = legacy._preprocess_image(crop[..., ::-1])
    raw = legacy.ort_session.run(
        [legacy.output_name],
        {legacy.input_name: preprocessed},
    )[0]

    np.save(data_dir / "golden_pre_T.npy", preprocessed[0])
    np.save(data_dir / "golden_raw_T.npy", raw)
    model = MODELS[DEFAULT_MODEL]
    metadata = {
        "crop_color_order": "RGB",
        "model": DEFAULT_MODEL,
        "mirror_repo_id": model.repo_id,
        "mirror_revision": model.revision,
        "official_repo_id": "bytedance-research/LVFace",
        "official_revision": "b12702ab1f5c721748e054a66dc90e1edd1f0724",
        "model_sha256": model.sha256,
        "preprocessing": "RGB_CHW_x_div_255_minus_0.5_div_0.5_v1",
        "generator": "legacy LVFaceONNXInferencer._preprocess_image",
    }
    (data_dir / "golden_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
