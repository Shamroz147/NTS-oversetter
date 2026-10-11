import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fant ikke kameraet")
    exit()


FIL = "data.csv"

BOKSTAVER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ")
idx = 0

ANTALL_PER_OPPTAKK = 100
opptak = False
tatt_opp = 0


fil_finnes = os.path.exists(FIL)
f = open(FIL, "a", newline="", encoding="utf-8")
writer = csv.writer(f)


if not fil_finnes:
    header = ["bokstav"] + [f"{akse}{i}" for i in range(21) for akse in "xyz"]
    writer.writerow(header)



while True:
    ok, frame = cap.read()

    if not ok:
        print("Klarte ikke å lese fra kameraet")
        break


    bokstav = BOKSTAVER[idx]

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_bilde = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultat = landmarker.detect(mp_bilde)

    koordinater = []

    if resultat.hand_landmarks:
        hand = resultat.hand_landmarks[0]
        h, w = frame.shape[:2]

        for punkt in hand:
            px = int(punkt.x * w)
            py = int(punkt.y * h)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            koordinater.extend([punkt.x, punkt.y, punkt.z])


    if opptak and koordinater:
        writer.writerow([bokstav] + koordinater)
        tatt_opp +=1

        if tatt_opp >= ANTALL_PER_OPPTAKK:
            opptak = False
            print(f"Ferdig med {ANTALL_PER_OPPTAKK} eksempler for {bokstav}")


    cv2.putText(frame, bokstav, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    if opptak:
        cv2.putText(frame, f"OPPTAK {tatt_opp}/{ANTALL_PER_OPPTAKK}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    tast = cv2.waitKey(1) & 0xFF
    if tast == ord('q'):
        break

    if tast == ord(" ") and not opptak:
        opptak = True
        tatt_opp = 0

    if tast == ord("n") and not opptak:
        idx = (idx + 1) % len(BOKSTAVER)

    if tast == ord("p") and not opptak:
        idx = (idx - 1) % len(BOKSTAVER)
        
    

f.close()
cap.release()
cv2.destroyAllWindows()