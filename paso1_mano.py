# Paso 1: abrir la cámara y dibujar los puntos de la mano
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODELO = "hand_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")

# La primera vez descarga el modelo que reconoce manos
if not os.path.exists(MODELO):
    print("Descargando el modelo de manos (solo la primera vez)...")
    urllib.request.urlretrieve(URL_MODELO, MODELO)

# Qué puntos de la mano se unen con líneas (los 21 puntos de MediaPipe)
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

opciones = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODELO),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(opciones)

camara = cv2.VideoCapture(0)
inicio = time.time()

while True:
    ok, frame = camara.read()
    if not ok:
        print("No pude leer la cámara. Revisá los permisos de cámara de VS Code.")
        break

    frame = cv2.flip(frame, 1)  # efecto espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    milisegundos = int((time.time() - inicio) * 1000)
    resultado = detector.detect_for_video(imagen, milisegundos)

    alto, ancho, _ = frame.shape
    for mano in resultado.hand_landmarks:
        puntos = [(int(p.x * ancho), int(p.y * alto)) for p in mano]
        for a, b in CONEXIONES:
            cv2.line(frame, puntos[a], puntos[b], (255, 255, 255), 2)
        for x, y in puntos:
            cv2.circle(frame, (x, y), 5, (180, 80, 255), -1)

    cv2.putText(frame, "Apreta Q para salir", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Lector de senas - paso 1", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
