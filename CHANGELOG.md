# Changelog

## 0.1.0

- Initial ONNX-only LVFace embedding API for Python 3.11–3.13.
- Pluggable face detector and embedder adapters.
- Face comparison, search, group matching, and conservative identity clustering.
- Revision-pinned, checksum-validated optional model downloads.

CPU inference is the supported runtime for this release. The default cosine threshold is
provisional and must be calibrated for each deployment. LVFace embedding-weight licensing is
unresolved because the official metadata and model-card prose conflict; the default InsightFace
detector weights are separately restricted to non-commercial research use.
