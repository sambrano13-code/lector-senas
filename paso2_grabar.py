# Paso 2: grabar ejemplos de cada seña
#   D = siguiente seña      A = seña anterior
#   ESPACIO = grabar 100 ejemplos de la seña elegida (con 2 segundos para acomodarte)
#   Q = salir
# Señas con movimiento: repetí el movimiento todo el tiempo mientras graba.
# Señas quietas: quedate quieta (podés cambiar un poco la distancia, pero despacio).
import csv
import os
import time
from collections import Counter

import cv2

from comun import CAMARA, SENAS, DetectorManos, DetectorCuerpo, Caracteristicas, dibujar_mano, dibujar_cuerpo, escribir

ARCHIVO = "datos.csv"
EJEMPLOS_POR_TANDA = 100
ESPERA = 2.0

conteo = Counter()
if os.path.exists(ARCHIVO):
    with open(ARCHIVO, encoding="utf-8") as f:
        for fila in csv.reader(f):
            if fila:
                conteo[fila[0]] += 1

detector = DetectorManos()
detector_cuerpo = DetectorCuerpo()
caracteristicas = Caracteristicas()
camara = cv2.VideoCapture(CAMARA)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # pide calidad HD
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
archivo = open(ARCHIVO, "a", newline="", encoding="utf-8")
escritor = csv.writer(archivo)

indice = 0           # qué seña está elegida
grabando = False
inicio_tanda = 0
guardados = 0

while True:
    ok, frame = camara.read()
    if not ok:
        print("No pude leer la cámara.")
        break
    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape
    manos = detector.detectar(frame)
    cuerpo = detector_cuerpo.detectar(frame)
    dibujar_cuerpo(frame, cuerpo)
    for puntos, _ in manos:
        dibujar_mano(frame, puntos)
    vector = caracteristicas.calcular(manos, cuerpo)

    sena = SENAS[indice]
    faltan = sum(1 for s in SENAS if conteo[s] == 0)
    escribir(frame, f"Seña {indice + 1}/{len(SENAS)}:  {sena}", (10, 10), 44, (0, 220, 255))
    escribir(frame, f"Ejemplos grabados: {conteo[sena]}   |   Señas sin grabar: {faltan}", (10, 65), 22)
    escribir(frame, "A ← anterior   D → siguiente   ESPACIO grabar   Q salir", (10, 95), 22, (120, 255, 120))

    if grabando:
        restante = ESPERA - (time.time() - inicio_tanda)
        if restante > 0:
            escribir(frame, f"Prepará '{sena}'... {restante:.0f}",
                     (ancho // 2, alto - 80), 40, (0, 220, 255), centrado=True)
        elif vector is not None:
            escritor.writerow([sena] + [f"{v:.5f}" for v in vector])
            guardados += 1
            conteo[sena] += 1
            escribir(frame, f"Grabando '{sena}': {guardados}/{EJEMPLOS_POR_TANDA}",
                     (ancho // 2, alto - 80), 40, (80, 80, 255), centrado=True)
            if guardados >= EJEMPLOS_POR_TANDA:
                archivo.flush()
                print(f"Listo: {sena} ({conteo[sena]} ejemplos)")
                grabando = False
                indice = min(indice + 1, len(SENAS) - 1)   # pasa sola a la siguiente
        else:
            escribir(frame, "No veo la mano", (ancho // 2, alto - 80), 40, (80, 80, 255), centrado=True)

    cv2.imshow("Lector de senas - grabar", frame)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if not grabando:
        if tecla == ord("d"):
            indice = (indice + 1) % len(SENAS)
        elif tecla == ord("a"):
            indice = (indice - 1) % len(SENAS)
        elif tecla == ord(" "):
            grabando = True
            inicio_tanda = time.time()
            guardados = 0

archivo.close()
camara.release()
cv2.destroyAllWindows()
print("Ejemplos guardados:", dict(conteo))
