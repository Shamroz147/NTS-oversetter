import cv2
import joblib
import numpy as np
import mediapipe as mp 
from mediapipe.tasks import python as mb_python
from mediapipe.tasks.python import vision


def normaliser(rad, speilvend=False, forhold=1.0):
    punkter = np.array(rad, dtype=float).reshape(21, 3) #Gjør om 63 tall til tabell med 21 rader (punkter) og 3 kolonner (x, y, z)
    punkter = punkter - punkter[0]

    punkter[:, 0] = punkter[:, 0] * forhold
    punkter[:, 2] = punkter[:, 2] * forhold

    if speilvend:
        punkter[:, 0] = -punkter[:, 0]


    skala = np.linalg.norm(punkter[9])
    punkter = punkter / skala

    return punkter.flatten()

model = joblib.load("modell.joblib")

options = vision.HandLandmarkerOptions(
    base_options=mb_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fant ikke kameraet")
    exit()

VISNING = {"Æ": "AE", "Ø": "OE", "Å": "AA"}


while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultat = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if resultat.hand_landmarks:
        koordinater = []
        for punkt in resultat.hand_landmarks[0]:
            cv2.circle(frame, (int(punkt.x * w), int(punkt.y * h)), 5, (0, 255, 0), -1)
            koordinater.extend([punkt.x, punkt.y, punkt.z])


        x = normaliser(koordinater, forhold=w / h)


        bokstav = model.predict([x])[0]


        sikkerhet = model.predict_proba([x])[0].max()

        tekst = f"{VISNING.get(bokstav, bokstav)}   {sikkerhet:.0%}"
        cv2.putText(frame, tekst, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)

    cv2.imshow("NTS-oversetter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()