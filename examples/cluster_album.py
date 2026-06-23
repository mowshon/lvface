"""Group appearances of the same person across several photos."""

from lvface import FaceRecognizer

photos = [
    "album/photo-1.jpg",
    "album/photo-2.jpg",
    "album/photo-3.jpg",
]

recognizer = FaceRecognizer("LVFace-T_Glint360K")
people = recognizer.group(photos)

for person_index, faces in enumerate(people):
    print(f"Person {person_index}:")
    for face in faces:
        print(f"  {photos[face.image_index]}, face {face.face_index}")
