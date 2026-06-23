"""Extract every face embedding and store it in a FAISS index."""

import json

import numpy as np

from lvface import FaceRecognizer

try:
    import faiss
except ImportError:
    raise SystemExit("Install FAISS first: python -m pip install faiss-cpu") from None


IMAGE = "group-photo.jpg"

recognizer = FaceRecognizer("LVFace-T_Glint360K")
faces = recognizer.analyze(IMAGE)

embeddings = [
    face.embedding.vector
    for face in faces
    if face.embedding is not None
]

if not embeddings:
    raise SystemExit("No faces found")

vectors = np.stack(embeddings).astype("float32")

# LVFace embeddings are normalized, so inner product is cosine similarity.
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "faces.index")

metadata = [
    {
        "image": IMAGE,
        "face_index": face.face_index,
    }
    for face in faces
    if face.embedding is not None
]

with open("faces.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(f"Stored {index.ntotal} face embeddings")
