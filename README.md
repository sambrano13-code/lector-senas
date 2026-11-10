# 🤟 Lector de Lengua de Señas

Traductor en tiempo real de **Lengua de Señas Argentina (LSA)** a texto y voz, usando la cámara.

Reconoce el **abecedario completo** (A–Z, con Ñ) y las palabras **Hola, Mi nombre es, Gracias, Sí, No, Por favor y Chau**. Muestra lo que se dice como **subtítulo** y lo **lee en voz alta**. También permite **deletrear**: las letras se juntan y, al bajar la mano, se dice la palabra completa.

> Ejemplo: *"Hola, mi nombre es R-O-M-I-N-A"* → 🔊 "Hola, mi nombre es Romina"

## ¿Cómo funciona?

1. **MediaPipe** detecta en cada imagen los 21 puntos de cada mano y la posición del cuerpo y la cara.
2. Esos puntos se convierten en características: la forma de las dos manos, dónde está una respecto de la otra, el **movimiento** del último medio segundo y la posición de las manos respecto del **cuerpo y la cara**.
3. Un modelo de **Machine Learning** (Random Forest, scikit-learn) entrenado con ejemplos propios clasifica la seña.
4. Para evitar errores, solo acepta una seña cuando la ve de forma estable durante varios cuadros.
5. El resultado se muestra como subtítulo y se reproduce en voz (grabaciones propias o la voz del sistema).

**Precisión:** ~98% en ejemplos que el modelo no vio durante el entrenamiento.

## Tecnologías

Python · MediaPipe · OpenCV · scikit-learn · NumPy · Pillow

## Cómo usarlo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python scikit-learn

python paso2_grabar.py      # grabar ejemplos de cada seña
python paso3_entrenar.py    # entrenar el modelo
python paso4_lector.py      # lector en vivo
```

## Archivos

| Archivo | Qué hace |
|---|---|
| `comun.py` | Detección de manos y cuerpo, cálculo de características, subtítulos |
| `paso1_mano.py` | Prueba: dibuja los puntos de la mano |
| `paso2_grabar.py` | Graba ejemplos de cada seña |
| `paso3_entrenar.py` | Entrena el modelo y muestra la precisión |
| `paso4_lector.py` | Traductor en vivo con subtítulos y voz |
| `voces/` | Grabaciones de voz propias para cada palabra |

## Próximos pasos

- Más vocabulario y frases completas.
- Modelo de secuencias (LSTM) para señas con movimiento más complejas.
- Versión web o para celular.
- Entrenar con muchas personas distintas para que funcione con cualquiera.

## Autora

**Romina Sambrano**: estudiante de la Tecnicatura Superior en Desarrollo de Software.

[LinkedIn](https://linkedin.com/in/romina-sambrano) · [GitHub](https://github.com/sambrano13-code)
