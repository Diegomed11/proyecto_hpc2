"""
entrenar.py
-----------
Entrena un modelo YOLO propio para detectar tornillos y rondanas, a partir de
un dataset etiquetado y exportado desde Roboflow (formato "YOLOv11" /
Ultralytics, que incluye un archivo data.yaml).

Uso típico:
  python entrenar.py --data dataset/data.yaml --epochs 100

Parte de los pesos preentrenados yolo11n.pt (transfer learning): así se
necesitan muchas menos imágenes que entrenar desde cero.

Al terminar, los pesos quedan en:
  runs/detect/tornillos_rondanas/weights/best.pt
y se usan directamente en el sistema de conteo:
  python conteo_banda.py --modelo runs/detect/tornillos_rondanas/weights/best.pt --fuente 0

NOTA: en este equipo no hay GPU, así que el entrenamiento corre en CPU y es
lento. Para ir rápido, entrena en Google Colab (GPU gratis) o en Roboflow, y
trae solo el best.pt aquí. Ver ENTRENAMIENTO.md.
"""

import argparse

from ultralytics import YOLO


def parsear_args():
    p = argparse.ArgumentParser(description="Entrena YOLO con tornillos y rondanas.")
    p.add_argument("--data", default="dataset/data.yaml",
                   help="Ruta al data.yaml exportado por Roboflow.")
    p.add_argument("--base", default="yolo11n.pt",
                   help="Pesos base (yolo11n.pt = rápido, yolo11s.pt = más preciso).")
    p.add_argument("--epochs", type=int, default=100, help="Número de épocas.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Tamaño de imagen. Súbelo (p. ej. 960) si las piezas son pequeñas.")
    p.add_argument("--batch", type=int, default=-1,
                   help="Tamaño de lote (-1 = automático).")
    p.add_argument("--dispositivo", default="cpu",
                   help="'cpu' o '0' para la primera GPU.")
    p.add_argument("--nombre", default="tornillos_rondanas",
                   help="Nombre de la carpeta de resultados.")
    return p.parse_args()


def main():
    args = parsear_args()
    print(f"[info] entrenando desde {args.base} con datos de {args.data}")
    print(f"[info] dispositivo: {args.dispositivo} (cpu = lento, sin GPU)")

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.dispositivo,
        name=args.nombre,
        patience=30,        # corta si deja de mejorar 30 épocas seguidas
        plots=True,         # genera gráficas y matriz de confusión
    )

    mejor = f"runs/detect/{args.nombre}/weights/best.pt"
    print("\n===== ENTRENAMIENTO TERMINADO =====")
    print(f"Pesos del mejor modelo: {mejor}")
    print("Pruébalo en el conteo con:")
    print(f"  python conteo_banda.py --modelo {mejor} --fuente 0")
    print(f"  python conteo_banda.py --modelo {mejor} --fuente piezas.mp4 --clases tornillo,rondana")


if __name__ == "__main__":
    main()
