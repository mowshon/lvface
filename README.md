# lvface

`lvface` detects faces, aligns them, produces 512-dimensional LVFace embeddings, and compares
people across portraits, group photos, or whole albums. The high-level API is small, while the
detector and embedder are both replaceable.

This package was inspired by ByteDance's
[`bytedance/lvface`](https://github.com/bytedance/lvface) repository and was created to make
this splendid model easy to install and use through the `pip` package manager.

```python
from lvface import FaceRecognizer

recognizer = FaceRecognizer("LVFace-T_Glint360K")
result = recognizer.compare("id-photo.jpg", "selfie.jpg")

print(result.is_match)
print(f"cosine={result.cosine:.4f}, display={result.percentage:.1f}%")
```

> Face recognition is biometric processing. Get informed consent, protect stored embeddings,
> define retention rules, and evaluate accuracy and bias on data representative of your users.

## Why lvface?

- One pipeline for paths, image bytes, URLs, and RGB NumPy arrays.
- Every face in an image can be returned, not only the largest one.
- Multi-face search, group-photo matching, and album clustering are built in.
- Released LVFace ONNX models run through ONNX Runtime; PyTorch is not required.
- Custom detectors and embedders plug into the same `FaceRecognizer`.
- Named weights are revision-pinned and checksum-verified.

## Install

Python 3.11 or newer is required.

```bash
# Recommended: recognition from ordinary photos + automatic weight download
python -m pip install "lvface[detect,hub]"

# Local ONNX weights and already aligned 112×112 face crops
python -m pip install lvface

# Add guarded http(s) image loading
python -m pip install "lvface[detect,hub,http]"
```

The `[detect]` extra installs the default InsightFace detector. The `[hub]` extra lets a
registered model name download its pinned ONNX file on first construction:

```python
recognizer = FaceRecognizer("LVFace-T_Glint360K")
```

To keep weights under your own control, pass a local file. This path never accesses Hugging Face
and does not need `[hub]`:

```python
recognizer = FaceRecognizer("/models/LVFace-T_Glint360K.onnx")
```

Model resolution and download happen while `FaceRecognizer` is constructed. The ONNX Runtime
session itself is still created lazily on the first embedding call.

CPU is the supported runtime for the 0.1 release. There is no `[gpu]` extra because
`onnxruntime` and `onnxruntime-gpu` provide the same Python package. For best-effort NVIDIA CUDA
use on Linux or Windows, install all extras first, then replace the runtime:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

## A 60-second tour

### Compare two photos

```python
from lvface import FaceRecognizer

recognizer = FaceRecognizer(device="auto")
result = recognizer.compare("first.jpg", "second.jpg", select="largest")

if result.is_match:
    print(f"Likely the same person ({result.percentage:.1f}% display score)")
```

`percentage` is a readable, threshold-centered display score. It is not a probability or a
calibrated confidence value. Use `cosine` and a threshold calibrated for your own camera,
population, and risk tolerance to make decisions.

### Get an embedding for every face

```python
faces = recognizer.analyze("team-photo.jpg")

for face in faces:
    vector = face.embedding.vector
    print(face.face_index, face.bbox, vector.shape)  # (512,)
```

`analyze()` performs load → detect → align → embed and returns a `Face` for every alignable
face. Each result carries its bounding box, five landmarks, aligned crop, and L2-normalized
embedding.

If you only need vectors:

```python
embeddings = recognizer.embed("team-photo.jpg")  # list[Embedding]
one_embedding = recognizer.embed("portrait.jpg", select="largest")
```

### Store face embeddings with FAISS

The FAISS example stores every detected face in a local cosine-similarity index. A small JSON
file keeps the image and face index associated with each vector:

```bash
python -m pip install faiss-cpu
python examples/embed_and_store.py
```

Edit the `IMAGE` constant at the top of the script before running it. The index uses cosine
similarity, matching `lvface`'s canonical comparison metric. The script writes `faces.index` and
`faces.json`. For production, treat both files as biometric data: restrict access, encrypt
backups, and delete vectors when their source data must be removed.

To search the saved index, edit `QUERY_IMAGE` in the companion example:

```bash
python examples/search_faiss.py
```

It embeds the largest face in the query photo, loads `faces.index`, and prints the nearest stored
faces with their cosine similarity. Use the same LVFace model for indexing and searching.

### Find someone in a group photo

```python
hits = recognizer.find(
    "person-to-find.jpg",
    "group-photo.jpg",
    top_k=3,
)

for hit in hits:
    print(hit.candidate.face_index, hit.percentage, hit.candidate.bbox)
```

### Match two group photos

```python
result = recognizer.match("group-before.jpg", "group-after.jpg")

for pair in result.pairs:
    print(
        pair.query.face_index,
        "↔",
        pair.candidate.face_index,
        f"{pair.percentage:.1f}%",
    )
```

The default greedy assignment uses each face at most once. Install `lvface[hungarian]` and pass
`assignment="hungarian"` for globally optimal one-to-one assignment.

### Group an album by identity

```python
identities = recognizer.group(["day-1.jpg", "day-2.jpg", "day-3.jpg"])

for identity in identities:
    print([(face.image_index, face.face_index) for face in identity])
```

Clustering is conservative: every member must meet the threshold against every other member, and
one identity cannot contain two faces from the same image unless `one_per_image=False`.

## API at a glance

| Call | Result |
| --- | --- |
| `analyze(image)` | Every detected face, aligned crop, and embedding |
| `embed(image)` | Embeddings for every face |
| `embed(image, select="largest")` | One explicitly selected embedding |
| `embed_aligned(crop)` | Embed one pre-aligned 112×112 RGB crop |
| `compare(a, b)` | Cosine, Euclidean distance, display score, and decision |
| `verify(a, b)` | Boolean match decision |
| `find(query, gallery)` | One-to-many ranked face search |
| `match(a, b)` | Full many-to-many matrix and assigned pairs |
| `group(images)` | Conservative identity clusters across images |

Accepted image inputs are a path, `http(s)` URL with `[http]`, encoded bytes, or an RGB
`uint8` NumPy array. NumPy arrays are assumed to be RGB, not OpenCV BGR.

## Bring your own detector

A detector only needs to subclass `FaceDetector` and provide lazy `load()` plus `detect()`.
Each detected `Face` should contain a bounding box and five ArcFace-order landmarks. The base
class supplies the 112×112 alignment implementation.

```python
detector = MyDetector(...)
recognizer = FaceRecognizer(
    embedder="LVFace-T_Glint360K",
    detector=detector,
)
faces = recognizer.analyze("photo.jpg")
```

[`examples/custom_detector.py`](examples/custom_detector.py) is a complete OpenCV YuNet adapter
and shows the important part explicitly: the custom detector instance is passed into
`FaceRecognizer`, so its detections flow through alignment and LVFace embedding.

Change the detector model and image paths at the bottom of the example, then run it:

```bash
python examples/custom_detector.py
```

Custom embedding backends follow the same pattern: subclass `FaceEmbedder`, lazily initialize the
runtime in `load()`, and implement `_forward(batch)` to return an `(N, 512)` floating-point
array. The base class validates 112×112 RGB inputs, preprocesses them, batches inference, and
returns validated `Embedding` objects.

## Concepts that matter

**Embedding.** A 512-number representation of an aligned face. Embeddings returned by the public
API are L2-normalized.

**Cosine similarity.** The decision metric. Higher means more similar. The packaged `0.35`
default is a provisional starting point, not a domain-general operating threshold.

**Euclidean distance.** A diagnostic value. For normalized vectors,
`euclidean² = 2 - 2 × cosine`.

**Alignment.** Five facial landmarks are warped onto the ArcFace 112×112 template before
embedding. Good detection and alignment are part of recognition quality, not merely
preprocessing details.

**Display percentage.** A sigmoid mapping centered on the decision threshold. It is for UI
display only and must not be presented as probability, certainty, or an estimated false-match
rate.

## Runnable examples

Each example is intentionally a small, direct Python script. Open one, replace the sample image
paths, and run it:

```bash
python examples/verify_two_faces.py
python examples/embed_and_store.py
python examples/search_faiss.py
python examples/find_in_group.py
python examples/match_two_group_photos.py
python examples/cluster_album.py
python examples/custom_detector.py
```

From a source checkout:

```bash
python -m pip install -e ".[detect,hub]"
```

## Weights, licenses, and citation

The package code is MIT licensed.

The default InsightFace model packs, including `buffalo_l`, are separately licensed for
non-commercial research use. Applications requiring other terms should supply a detector with
appropriate weights or pass pre-aligned crops with `detector=None`.

LVFace embedding-weight licensing is unresolved. The official repository metadata declares MIT,
while its model-card prose restricts downloaded models to non-commercial research. The
unofficial
[`Mowshon/lvface-weights`](https://huggingface.co/Mowshon/lvface-weights) preservation mirror
grants no additional rights. `lvface` pins mirror revision
`83b567cd6a3fc34434667e4415b6125feceb39ea`; the mirror records unchanged files from official
[`bytedance-research/LVFace`](https://huggingface.co/bytedance-research/LVFace) revision
`b12702ab1f5c721748e054a66dc90e1edd1f0724`. Review the official model card and seek
clarification from the authors when necessary.

Use of the weights requires citation of the original work:

```bibtex
@inproceedings{you2025lvface,
  title={{LVFace}: Progressive Cluster Optimization for Large Vision Models in Face Recognition},
  author={You, Jinghan and Li, Shanglin and Sun, Yuanrui and Wei, Jiangchuan and Guo, Mingyu and Feng, Chao and Ran, Jiao},
  booktitle={ICCV},
  year={2025}
}
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
```
