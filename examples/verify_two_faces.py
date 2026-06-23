"""Compare the largest face in two photos."""

from lvface import FaceRecognizer

recognizer = FaceRecognizer("LVFace-T_Glint360K")

result = recognizer.compare("first-person.jpg", "second-person.jpg")

print("Same person:", result.is_match)
print("Cosine similarity:", result.cosine)
print("Display score:", result.percentage)
