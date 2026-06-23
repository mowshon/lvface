"""Find one person in a group photo."""

from lvface import FaceRecognizer

recognizer = FaceRecognizer("LVFace-T_Glint360K")

matches = recognizer.find(
    query="person-to-find.jpg",
    gallery="group-photo.jpg",
    top_k=3,
)

for match in matches:
    face = match.candidate
    print(
        f"Face {face.face_index}: "
        f"cosine={match.score:.4f}, "
        f"display={match.percentage:.1f}%"
    )
