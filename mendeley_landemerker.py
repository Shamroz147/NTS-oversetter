import cv2
import csv
import numpy as np 
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)


DATAMAPPE = Path("mendeley/raadata_BS_H_N")
UTFIL = "mendeley.csv"

f = open(UTFIL, "w", newline="", encoding="utf-8")
writer = csv.writer(f)
header = ["bokstav"] + [f"{akse}{i}" for i in range(21) for akse in "xyz"]
writer.writerow(header)


for mappe in sorted(DATAMAPPE.iterdir()):
    if not mappe.is_dir():
        continue

    bokstav = mappe.name

    funnet = 0
    ikke_funnet = 0

    for bildefil in mappe.glob("*.jp*g"):
        bilde = cv2.imdecode(np.fromfile(str(bildefil), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bilde is None:
            continue

        rgb = cv2.cvtColor(bilde, cv2.COLOR_BGR2RGB)
        mp_bilde = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultat = landmarker.detect(mp_bilde)

        if resultat.hand_landmarks:
            koordinater = []
            for punkt in resultat.hand_landmarks[0]:
                koordinater.extend([punkt.x, punkt.y, punkt.z])
            writer.writerow([bokstav] + koordinater)
            funnet += 1
        else:
            ikke_funnet += 1

    print(f"{bokstav}: {funnet} funnet, {ikke_funnet} uten hånd")

f.close()