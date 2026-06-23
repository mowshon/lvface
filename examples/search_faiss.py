"""Search for a face in a saved FAISS index."""

import json

import numpy as np

from lvface import FaceRecognizer

try:
    import faiss
except ImportError:
    raise SystemExit("Install FAISS first: python -m pip install faiss-cpu") from None


QUERY_IMAGE = "person-to-find.jpg"
TOP_K = 3

recognizer = FaceRecognizer("LVFace-T_Glint360K")
query = recognizer.embed(QUERY_IMAGE, select="largest")
query_vector = np.asarray([query.vector], dtype="float32")

index = faiss.read_index("faces.index")

with open("faces.json", encoding="utf-8") as file:
    metadata = json.load(file)

if index.ntotal == 0:
    raise SystemExit("The FAISS index is empty")

if len(metadata) != index.ntotal:
    raise SystemExit("faces.json does not match faces.index")

result_count = min(TOP_K, index.ntotal)
scores, positions = index.search(query_vector, result_count)

for score, position in zip(scores[0], positions[0], strict=True):
    face = metadata[position]
    print(
        f"{face['image']}, face {face['face_index']}: "
        f"cosine similarity={score:.4f}"
    )
