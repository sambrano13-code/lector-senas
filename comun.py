# Funciones que comparten todos los pasos del lector de señas
import os
import sys
import urllib.request

from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

# Qué cámara usar: 0 = la primera. Se puede elegir al correr, ej: python paso2_grabar.py 1
CAMARA = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Las señas que vamos a reconocer: el abecedario y algunas palabras
LETRAS = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
PALABRAS = ["Hola", "Mi nombre es", "Gracias", "Sí", "No", "Por favor", "Chau"]
SENAS = PALABRAS + LETRAS


def es_letra(sena):
    return len(sena) == 1

MODELO_MANOS = "hand_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")

MODELO_CUERPO = "pose_landmarker_lite.task"
URL_CUERPO = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# Partes del cuerpo que se dibujan: hombros, brazos, cadera y cara
CONEXIONES_CUERPO = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                     (11, 23), (12, 24), (23, 24), (9, 10)]
PUNTOS_CARA = [0, 2, 5, 9, 10]   # nariz, ojos, boca

CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

FUENTES = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/System/Library/Fonts/Supplemental/Arial.ttf",
           "/Library/Fonts/Arial.ttf"]


class DetectorManos:
    """Abre MediaPipe y devuelve los 21 puntos de la mano en cada imagen."""

    def __init__(self):
        if not os.path.exists(MODELO_MANOS):
            print("Descargando el modelo de manos (solo la primera vez)...")
            urllib.request.urlretrieve(URL_MODELO, MODELO_MANOS)
        opciones = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODELO_MANOS),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.detector = vision.HandLandmarker.create_from_options(opciones)
        self.ms = 0

    def detectar(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.ms += 33
        resultado = self.detector.detect_for_video(imagen, self.ms)
        manos = []
        for puntos, lado in zip(resultado.hand_landmarks, resultado.handedness):
            manos.append((puntos, lado[0].category_name))   # "Left" o "Right"
        return manos


class DetectorCuerpo:
    """Detecta hombros, brazos, cara y cadera."""

    def __init__(self):
        if not os.path.exists(MODELO_CUERPO):
            print("Descargando el modelo del cuerpo (solo la primera vez)...")
            urllib.request.urlretrieve(URL_CUERPO, MODELO_CUERPO)
        opciones = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODELO_CUERPO),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.detector = vision.PoseLandmarker.create_from_options(opciones)
        self.ms = 0

    def detectar(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.ms += 33
        resultado = self.detector.detect_for_video(imagen, self.ms)
        if resultado.pose_landmarks:
            return resultado.pose_landmarks[0]
        return None


def dibujar_cuerpo(frame, cuerpo):
    if cuerpo is None:
        return
    alto, ancho, _ = frame.shape
    p = [(int(q.x * ancho), int(q.y * alto)) for q in cuerpo]
    for a, b in CONEXIONES_CUERPO:
        cv2.line(frame, p[a], p[b], (255, 200, 80), 3)
    for i in PUNTOS_CARA + [11, 12, 13, 14]:
        cv2.circle(frame, p[i], 6, (255, 200, 80), -1)


def dibujar_mano(frame, mano):
    alto, ancho, _ = frame.shape
    puntos = [(int(p.x * ancho), int(p.y * alto)) for p in mano]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], (255, 255, 255), 2)
    for x, y in puntos:
        cv2.circle(frame, (x, y), 5, (180, 80, 255), -1)


def normalizar(mano):
    """Convierte los 21 puntos en 42 números que no dependen de dónde
    está la mano ni de qué tan cerca de la cámara."""
    puntos = np.array([[p.x, p.y] for p in mano])
    puntos = puntos - puntos[0]               # la muñeca queda en (0, 0)
    escala = np.max(np.linalg.norm(puntos, axis=1))
    if escala > 0:
        puntos = puntos / escala
    return puntos.flatten()


def _tamano_mano(mano):
    """Distancia muñeca - nudillo del dedo medio: sirve de 'regla'."""
    return max(np.hypot(mano[9].x - mano[0].x, mano[9].y - mano[0].y), 1e-6)


class Caracteristicas:
    """Convierte lo que ve la cámara en una lista de números para el modelo:
    - la forma de cada mano (izquierda y derecha, o ceros si no está)
    - dónde está una mano respecto de la otra
    - cómo se movió la mano principal en el último medio segundo
    - dónde están las manos respecto del cuerpo y de la cara
    """

    HISTORIA = 16          # cuadros que recuerda (~0,5 s)
    MUESTRAS = (15, 10, 5)  # cuadros hacia atrás que compara con el actual

    def __init__(self):
        self.historia = deque(maxlen=self.HISTORIA)

    def calcular(self, manos, cuerpo=None):
        if not manos:
            self.historia.clear()
            return None

        por_lado = {}
        for puntos, lado in manos:
            por_lado.setdefault(lado, puntos)

        vector = []
        for lado in ("Left", "Right"):
            if lado in por_lado:
                vector.extend(normalizar(por_lado[lado]))
                vector.append(1.0)
            else:
                vector.extend([0.0] * 42)
                vector.append(0.0)

        # Posición de una mano respecto de la otra
        if "Left" in por_lado and "Right" in por_lado:
            izq, der = por_lado["Left"], por_lado["Right"]
            escala = (_tamano_mano(izq) + _tamano_mano(der)) / 2
            vector.extend([(der[0].x - izq[0].x) / escala, (der[0].y - izq[0].y) / escala])
        else:
            vector.extend([0.0, 0.0])

        # Movimiento de la mano principal (muñeca y punta del índice)
        principal = por_lado.get("Right", por_lado.get("Left"))
        escala = _tamano_mano(principal)
        actual = np.array([principal[0].x, principal[0].y, principal[8].x, principal[8].y])
        self.historia.append((actual, escala))
        for atras in self.MUESTRAS:
            if len(self.historia) > atras:
                pasado, _ = self.historia[-1 - atras]
                vector.extend((actual - pasado) / escala)
            else:
                vector.extend([0.0] * 4)

        # Manos respecto del cuerpo: centro = entre los hombros, regla = ancho de hombros
        if cuerpo is not None:
            hom_i, hom_d, nariz = cuerpo[11], cuerpo[12], cuerpo[0]
            cx, cy = (hom_i.x + hom_d.x) / 2, (hom_i.y + hom_d.y) / 2
            ancho = max(np.hypot(hom_i.x - hom_d.x, hom_i.y - hom_d.y), 1e-6)
            for lado in ("Left", "Right"):
                if lado in por_lado:
                    m = por_lado[lado]
                    vector.extend([(m[0].x - cx) / ancho, (m[0].y - cy) / ancho,     # muñeca
                                   (m[8].x - cx) / ancho, (m[8].y - cy) / ancho,     # punta índice
                                   (m[8].x - nariz.x) / ancho, (m[8].y - nariz.y) / ancho])  # índice vs cara
                else:
                    vector.extend([0.0] * 6)
            vector.append(1.0)
        else:
            vector.extend([0.0] * 13)

        return np.array(vector)


_cache_fuentes = {}


def _fuente(tamano):
    if tamano not in _cache_fuentes:
        fuente = None
        for ruta in FUENTES:
            if os.path.exists(ruta):
                fuente = ImageFont.truetype(ruta, tamano)
                break
        _cache_fuentes[tamano] = fuente or ImageFont.load_default()
    return _cache_fuentes[tamano]


def escribir(frame, texto, posicion, tamano=28, color=(255, 255, 255), centrado=False):
    """Escribe texto con tildes y ñ (OpenCV solo no puede)."""
    imagen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dibujo = ImageDraw.Draw(imagen)
    fuente = _fuente(tamano)
    x, y = posicion
    if centrado:
        caja = dibujo.textbbox((0, 0), texto, font=fuente)
        x = x - (caja[2] - caja[0]) // 2
    dibujo.text((x, y), texto, font=fuente, fill=color)
    frame[:] = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)


def subtitulo(frame, texto):
    """Franja oscura abajo con la palabra grande, como en una película."""
    alto, ancho, _ = frame.shape
    franja = frame.copy()
    cv2.rectangle(franja, (0, alto - 110), (ancho, alto), (0, 0, 0), -1)
    cv2.addWeighted(franja, 0.6, frame, 0.4, 0, frame)
    # Achica la letra si la frase es larga
    tamano = 64
    while tamano > 24:
        caja = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), texto, font=_fuente(tamano))
        if caja[2] - caja[0] < ancho - 40:
            break
        tamano -= 4
    escribir(frame, texto, (ancho // 2, alto - 55 - tamano // 2 - 10), tamano=tamano, centrado=True)
