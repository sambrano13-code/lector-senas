# Paso 4: el lector en vivo, con subtítulo abajo y voz
# - Las palabras se dicen apenas se reconocen.
# - Las letras se van juntando (deletreo) y, cuando bajás la mano,
#   la Mac dice la palabra completa. Ej: "Hola, mi nombre es ROMINA".
# - Para repetir una letra (ej. la doble L), bajá o mové un poco la mano entre una y otra.
# Teclas: C = borrar la frase   V = ver/ocultar lo que 'piensa' el modelo   Q = salir
import os
import subprocess
import unicodedata
from collections import deque

import cv2
import joblib

from comun import CAMARA, DetectorManos, DetectorCuerpo, Caracteristicas, dibujar_mano, dibujar_cuerpo, escribir, subtitulo, es_letra

CUADROS_ESTABLES = 12      # cuántos cuadros seguidos tiene que ver la misma seña
CONFIANZA_MINIMA = 0.70
CUADROS_PARA_CERRAR_PALABRA = 20   # mano abajo ~0,7 s: termina el deletreo
CUADROS_PARA_BORRAR = 150          # mano abajo ~5 s: se limpia el subtítulo
VOZ_MAC = "Paulina"        # voz en español de la Mac si no hay grabación propia

modelo = joblib.load("modelo_senas.pkl")
detector = DetectorManos()
detector_cuerpo = DetectorCuerpo()
caracteristicas = Caracteristicas()
camara = cv2.VideoCapture(CAMARA)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # pide calidad HD
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ultimas = deque(maxlen=CUADROS_ESTABLES)
frase = []          # palabras ya dichas
letras = ""         # deletreo en curso
ultima_sena = None
cuadros_sin_mano = 0
ver_diagnostico = True
top3 = []


def nombre_archivo(palabra):
    """'Por favor' -> 'por_favor', 'Sí' -> 'si', 'Mi nombre es' -> 'mi_nombre_es'"""
    sin_tildes = unicodedata.normalize("NFD", palabra)
    sin_tildes = "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")
    return sin_tildes.lower().replace(" ", "_")


def hablar(texto):
    base = os.path.join("voces", nombre_archivo(texto))
    for extension in (".m4a", ".mp3", ".wav", ".aiff"):
        if os.path.exists(base + extension):
            subprocess.Popen(["afplay", base + extension])   # tu propia voz
            return
    texto = texto.replace('"', "")
    subprocess.Popen(f'say -v {VOZ_MAC} "{texto}" || say "{texto}"', shell=True)


def cerrar_deletreo():
    global letras
    if letras:
        palabra = letras.capitalize()
        frase.append(palabra)
        hablar(palabra)
        letras = ""


while True:
    ok, frame = camara.read()
    if not ok:
        print("No pude leer la cámara.")
        break
    frame = cv2.flip(frame, 1)
    manos = detector.detectar(frame)
    cuerpo = detector_cuerpo.detectar(frame)
    dibujar_cuerpo(frame, cuerpo)
    vector = caracteristicas.calcular(manos, cuerpo)

    mano_visible = vector is not None
    if vector is not None:
        cuadros_sin_mano = 0
        for puntos, _ in manos:
            dibujar_mano(frame, puntos)
        probabilidades = modelo.predict_proba([vector])[0]
        mejor = probabilidades.argmax()
        orden = probabilidades.argsort()[::-1][:3]
        top3 = [(modelo.classes_[i], probabilidades[i]) for i in orden]
        ultimas.append(modelo.classes_[mejor] if probabilidades[mejor] >= CONFIANZA_MINIMA else None)

        if len(ultimas) == CUADROS_ESTABLES and len(set(ultimas)) == 1:
            sena = ultimas[0]
            if sena and sena != ultima_sena:
                if es_letra(sena):
                    letras += sena
                else:
                    cerrar_deletreo()
                    frase.append(sena)
                    hablar(sena)
            ultima_sena = sena      # None si dudó: permite repetir la misma letra
    else:
        cuadros_sin_mano += 1
        ultimas.clear()
        if cuadros_sin_mano == CUADROS_PARA_CERRAR_PALABRA:
            cerrar_deletreo()
            ultima_sena = None
        if cuadros_sin_mano == CUADROS_PARA_BORRAR:
            frase.clear()

    escribir(frame, "C: borrar   V: diagnóstico   Q: salir", (10, 10), 22, (120, 255, 120))
    if ver_diagnostico and mano_visible:
        for i, (nombre, prob) in enumerate(top3):
            color = (120, 255, 120) if prob >= CONFIANZA_MINIMA else (200, 200, 200)
            escribir(frame, f"{nombre}: {prob:.0%}", (10, 45 + i * 30), 26 if i == 0 else 22, color)
    texto = " ".join(frase + ([letras + "_"] if letras else []))
    if texto:
        subtitulo(frame, texto)

    cv2.imshow("Lector de senas", frame)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord("v"):
        ver_diagnostico = not ver_diagnostico
    if tecla == ord("c"):
        frase.clear()
        letras = ""
        ultima_sena = None

camara.release()
cv2.destroyAllWindows()
