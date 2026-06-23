"""Find people who appear in two group photos."""

from lvface import FaceRecognizer

recognizer = FaceRecognizer("LVFace-T_Glint360K")

result = recognizer.match("first-group.jpg", "second-group.jpg")

for match in result.pairs:
    print(
        f"First photo face {match.query.face_index} matches "
        f"second photo face {match.candidate.face_index}: "
        f"{match.percentage:.1f}%"
    )
