# Conteo de objetos sobre banda transportadora (Visión Artificial)

Sistema que cuenta objetos que circulan por una banda transportadora usando
visión por computadora. Detecta los objetos, les hace seguimiento y los
**cuenta cuando cruzan una línea virtual**, separando por dirección.

Incluye **dos enfoques**:

| Script | Enfoque | Cuándo usarlo |
|---|---|---|
| `conteo_banda.py` | **YOLO** (deep learning, repo [Ultralytics](https://github.com/ultralytics/ultralytics)) | Objetos reales y variados (botellas, cajas, frutas, piezas…). Robusto. |
| `conteo_demo.py` | **Visión clásica** (OpenCV: sustracción de fondo + contornos) | Entorno controlado con fondo fijo. Ligero, sin red neuronal. Ideal para probar ya. |

Ambos comparten la misma lógica de conteo (`contador.py`).

---

## 1. Requisitos

Ya están instalados en este equipo (Python 3.14). Para reproducir en otra máquina:

```bash
pip install -r requirements.txt
```

Paquetes: `ultralytics` (YOLO + tracking), `opencv-python`, `numpy`.

---

## 2. Probarlo YA (sin cámara ni videos)

Genera un video sintético de banda y cuéntalo con el detector clásico:

```bash
python generar_demo.py                       # crea demo_banda.mp4
python conteo_demo.py --fuente demo_banda.mp4
```

Se abre una ventana en vivo. Pulsa **q** o **Esc** para salir. Resultado esperado:
`Total contados: 18` (todos hacia la DERECHA).

Para no abrir ventana y guardar el video anotado:

```bash
python conteo_demo.py --fuente demo_banda.mp4 --no-mostrar --guardar salida.mp4
```

---

## 3. Uso real con YOLO

### Cámara web

```bash
python conteo_banda.py --fuente 0
```

### Archivo de video

```bash
python conteo_banda.py --fuente banda.mp4
```

> La **primera vez** descarga los pesos del modelo (`yolo11n.pt`, ~5 MB).
> Requiere conexión a internet solo esa primera vez.

### Opciones útiles

```bash
# Línea vertical al 60% del ancho (banda que avanza en horizontal)
python conteo_banda.py --fuente banda.mp4 --linea vertical --posicion 0.6

# Contar SOLO ciertas clases (nombres del dataset COCO)
python conteo_banda.py --fuente banda.mp4 --clases bottle,cup

# Línea a medida con dos puntos: x1,y1,x2,y2
python conteo_banda.py --fuente banda.mp4 --puntos 100,400,1180,400

# Guardar resultado y procesar sin ventana
python conteo_banda.py --fuente banda.mp4 --no-mostrar --guardar salida.mp4

# Modelo más preciso (más lento): yolo11s.pt / yolo11m.pt
python conteo_banda.py --fuente banda.mp4 --modelo yolo11s.pt
```

Ver todas las opciones: `python conteo_banda.py --help`.

> **Clases:** YOLO viene entrenado con el dataset COCO (80 clases: `person`,
> `bottle`, `cup`, `book`, `cell phone`, `apple`, `banana`, etc.). Si tus
> objetos no están en COCO (p. ej. una pieza industrial específica), hay que
> **entrenar un modelo propio** con tus imágenes; ver sección 6.

---

## 4. ¿Cómo funciona?

```
  video/cámara ─► DETECCIÓN ─► SEGUIMIENTO (id por objeto) ─► CONTEO por cruce de línea
                  (YOLO o          (ByteTrack en YOLO /         (contador.py)
                   OpenCV)          seguidor.py en demo)
```

1. **Detección:** se localizan los objetos en cada fotograma.
   - YOLO: red neuronal preentrenada.
   - Demo: sustracción de fondo (MOG2) + contornos → detecta lo que se mueve.
2. **Seguimiento:** a cada objeto se le asigna un identificador estable para
   no contarlo dos veces.
3. **Conteo:** una línea divide la imagen en dos lados. Cuando el centroide de
   un objeto cambia de lado, se cuenta una vez en la dirección del cruce
   (producto cruz del punto respecto a la línea, en `contador.py`).

---

## 5. Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `conteo_banda.py` | **Principal.** Conteo con YOLO (detección + tracking). |
| `conteo_demo.py` | Conteo con visión clásica de OpenCV (para demo / entornos fijos). |
| `generar_demo.py` | Crea un video sintético de banda para pruebas. |
| `contador.py` | Lógica de conteo por cruce de línea (compartida). |
| `seguidor.py` | Seguidor de centroides simple (solo modo demo). |
| `requirements.txt` | Dependencias. |

---

## 6. Ajustes y siguientes pasos

- **Posición de la línea:** mueve `--posicion` (0 a 1) o usa `--puntos` para
  ponerla perpendicular al movimiento de tu banda.
- **Sensibilidad (demo):** `--area-min` filtra contornos pequeños (ruido);
  súbelo si cuenta basura, bájalo si ignora objetos pequeños.
- **Confianza (YOLO):** `--conf` (por defecto 0.35). Súbelo para menos falsos
  positivos.
- **Entrenar un modelo propio** (objetos fuera de COCO, p. ej. **tornillos y
  rondanas**): ver la guía completa paso a paso en **[ENTRENAMIENTO.md](ENTRENAMIENTO.md)**
  (capturar fotos → etiquetar en Roboflow → entrenar → usar `best.pt`).
  Scripts de apoyo: `capturar_imagenes.py` (reunir el dataset) y `entrenar.py`.
  Luego usa tus pesos: `python conteo_banda.py --modelo runs/detect/tornillos_rondanas/weights/best.pt --fuente banda.mp4`
- **GPU:** para video en tiempo real con muchos objetos conviene una GPU NVIDIA
  e instalar la versión CUDA de PyTorch (aquí está la CPU, suficiente para
  videos y pruebas).
```
