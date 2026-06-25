# Entrenar un modelo propio: tornillos y rondanas

YOLO preentrenado (COCO) **no** conoce tornillos ni rondanas. Aquí entrenas tu
propio modelo con **Roboflow** (para etiquetar) + **Ultralytics** (para entrenar).
Al final solo cambias `--modelo` en `conteo_banda.py` y el conteo funciona igual.

```
  FOTOS  ─►  ETIQUETAR en Roboflow  ─►  EXPORTAR (YOLOv11)  ─►  ENTRENAR  ─►  best.pt  ─►  conteo_banda.py
 (paso 1)         (paso 2)                 (paso 3)            (paso 4)               (paso 5)
```

> **Clases que vamos a usar:** `tornillo` y `rondana` (2 clases).

---

## Paso 1 — Reunir fotos

Usa el script de captura (cámara o video). Pulsa **ESPACIO** para guardar cada foto:

```bash
python capturar_imagenes.py --fuente 0
# o automático, 1 de cada 15 fotogramas de un video:
python capturar_imagenes.py --fuente piezas.mp4 --cada 15
```

Las imágenes quedan en `dataset_crudo/`. **Recomendaciones para que el modelo
funcione bien** (los tornillos/rondanas son pequeños y metálicos, lo difícil):

- **Cantidad:** 150–300 imágenes para empezar (más = mejor).
- **Variedad:** muchos ángulos, rotaciones y posiciones de cada pieza.
- **Casos reales:** piezas solas, varias juntas, **tocándose** y medio tapadas.
- **Iluminación y fondo:** varios; idealmente sobre la **banda real**.
- **Equilibrio:** parecido número de tornillos y de rondanas.

---

## Paso 2 — Etiquetar en Roboflow

1. Entra a **https://roboflow.com** y crea una cuenta (plan gratuito sirve).
2. **Create New Project** → tipo **Object Detection**. Nómbralo p. ej. `tornillos-rondanas`.
3. **Upload** las imágenes de `dataset_crudo/`.
4. En cada imagen, dibuja una caja alrededor de cada pieza y asígnale la clase
   **`tornillo`** o **`rondana`**. (Crea esas dos clases la primera vez.)
   - Etiqueta **todas** las piezas de cada imagen, sin dejar ninguna.
   - Ajusta las cajas pegadas al borde de la pieza.
5. Cuando termines: **Annotate → aprobar las imágenes**.

---

## Paso 3 — Generar versión y exportar

1. **Generate** una versión del dataset. Roboflow ofrece:
   - **Train/Valid/Test split** (deja ~70/20/10).
   - **Preprocessing**: *Resize* a 640×640.
   - **Augmentations** (recomendado para pocas fotos): rotación, brillo,
     flip horizontal, algo de blur. Multiplican el dataset.
2. **Export Dataset** → formato **YOLOv11** (o "YOLOv8/Ultralytics", es compatible)
   → **download zip to computer**.
3. Descomprime el zip dentro del proyecto en una carpeta llamada `dataset/`.
   Debe contener `data.yaml`, `train/`, `valid/` (y quizá `test/`).

El `data.yaml` se verá así (lo genera Roboflow):

```yaml
train: ../train/images
val: ../valid/images
nc: 2
names: ['tornillo', 'rondana']
```

> **Alternativa con código** (en vez de descargar el zip a mano): instala el SDK
> `pip install roboflow` y descarga el dataset con tu API key:
> ```python
> from roboflow import Roboflow
> rf = Roboflow(api_key="TU_API_KEY")
> dataset = rf.workspace("tu-workspace").project("tornillos-rondanas")\
>             .version(1).download("yolov11")
> print(dataset.location)  # carpeta con el data.yaml
> ```

---

## Paso 4 — Entrenar

### Opción A (RÁPIDA y recomendada): Google Colab con GPU gratis

Tu PC **no tiene GPU**, así que entrenar aquí es lento. En Colab es gratis y rápido:

1. Abre https://colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → GPU**.
3. Sube tu `dataset/` (o descárgalo con el SDK de Roboflow de arriba) y ejecuta:

```python
!pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="dataset/data.yaml", epochs=100, imgsz=640, name="tornillos_rondanas")
```

4. Descarga el archivo `runs/detect/tornillos_rondanas/weights/best.pt` y
   cópialo a esta carpeta del proyecto.

### Opción B: entrenar localmente (CPU, lento pero funciona)

```bash
python entrenar.py --data dataset/data.yaml --epochs 100
```

En CPU puede tardar de minutos a varias horas según cuántas imágenes haya.
Para acelerar al probar: baja épocas (`--epochs 30`). Para piezas muy pequeñas,
sube resolución: `--imgsz 960`.

### Opción C: Roboflow Train (en la nube, sin código)

En Roboflow, botón **Train** → entrena en su nube y luego descargas los pesos.

---

## Paso 5 — Usar tu modelo en el conteo

Cuando tengas `best.pt`, solo cámbialo en el script de siempre:

```bash
# Cámara en vivo
python conteo_banda.py --modelo runs/detect/tornillos_rondanas/weights/best.pt --fuente 0

# Video, contando ambas clases y guardando el resultado
python conteo_banda.py --modelo best.pt --fuente piezas.mp4 --clases tornillo,rondana --guardar salida.mp4

# Contar SOLO tornillos
python conteo_banda.py --modelo best.pt --fuente piezas.mp4 --clases tornillo
```

El panel mostrará el total y el desglose por clase (tornillo / rondana).

---

## ¿El modelo cuenta mal? Cómo mejorarlo

| Síntoma | Solución |
|---|---|
| No detecta piezas | Más imágenes y variedad; baja `--conf` (p. ej. 0.25). |
| Confunde tornillo con rondana | Más ejemplos de ambos; revisa etiquetas mal puestas. |
| Falla con piezas pequeñas | Entrena con `--imgsz 960`; acerca la cámara. |
| Detecta de más (ruido) | Sube `--conf`; añade fotos del fondo sin piezas. |
| Cuenta doble | Coloca la línea perpendicular al movimiento; revisa `--posicion`. |

Revisa también las gráficas que genera el entrenamiento en
`runs/detect/tornillos_rondanas/` (curvas de pérdida, matriz de confusión).
