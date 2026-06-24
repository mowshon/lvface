import json
import subprocess
import sys

import lvface


def test_version_is_exposed() -> None:
    assert lvface.__version__ == "0.2.0"
    assert lvface.__all__ == [
        "AlignmentError",
        "BBox",
        "ComparisonResult",
        "DEFAULT_MODEL",
        "Embedding",
        "Face",
        "FaceDetector",
        "FaceEmbedder",
        "FaceRecognizer",
        "InsightFaceDetector",
        "LVFaceOnnxEmbedder",
        "MODELS",
        "Match",
        "MatchResult",
        "Model",
        "NoFaceError",
        "__version__",
        "load_image",
        "metrics",
        "resolve_model_path",
        "resolve_weights",
    ]


def test_lazy_public_exports_are_available() -> None:
    assert lvface.__getattr__("metrics").__name__ == "lvface.metrics"
    assert lvface.DEFAULT_MODEL == "LVFace-T_Glint360K"
    assert lvface.resolve_weights.__name__ == "resolve_weights"


def test_import_has_no_optional_runtime_dependencies() -> None:
    optional_modules = (
        "cv2",
        "huggingface_hub",
        "insightface",
        "onnxruntime",
        "requests",
        "scipy",
        "sklearn",
        "torch",
    )
    script = (
        "import json, sys; import lvface; "
        f"print(json.dumps(sorted(set({optional_modules!r}) & set(sys.modules))))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
