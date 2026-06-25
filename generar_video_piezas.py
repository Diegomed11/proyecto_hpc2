"""
generar_video_piezas.py
-----------------------
Crea un video de banda transportadora con tornillos y rondanas REALES,
recortados de las imágenes etiquetadas del dataset (BOLTS.v1i.yolov11).

Como son las mismas piezas con las que entrenó el modelo, el detector las
reconoce de verdad. Sirve para probar `conteo_banda.py --modelo best.pt`.

Uso:
  python generar_video_piezas.py
  python generar_video_piezas.py --dataset BOLTS.v1i.yolov11 --split train --objetos 30

Luego, para detectar y contar (¡línea VERTICAL, las piezas van de izq. a der.!):
  python conteo_banda.py --modelo best.pt --fuente video_piezas.mp4 --linea vertical --guardar deteccion.mp4
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import yaml


def parsear_args():
    p = argparse.ArgumentParser(description="Genera video de banda con piezas reales.")
    p.add_argument("--dataset", default="BOLTS.v1i.yolov11", help="Carpeta del dataset.")
    p.add_argument("--split", default="train", choices=["train", "valid", "test"])
    p.add_argument("--salida", default="video_piezas.mp4")
    p.add_argument("--ancho", type=int, default=960)
    p.add_argument("--alto", type=int, default=540)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--objetos", type=int, default=30, help="Cuántas piezas desfilan.")
    p.add_argument("--tam", type=int, default=85, help="Tamaño máx. (px) de cada pieza.")
    p.add_argument("--velocidad", type=int, default=5, help="Píxeles por fotograma.")
    p.add_argument("--separacion", type=int, default=130, help="Separación entre piezas (px).")
    return p.parse_args()


def recortar_piezas(dataset, split, tam_max, n_objetivo):
    """Recorta objetos reales usando las cajas de los labels YOLO."""
    img_dir = os.path.join(dataset, split, "images")
    lbl_dir = os.path.join(dataset, split, "labels")
    nombres = yaml.safe_load(open(os.path.join(dataset, "data.yaml")))["names"]

    labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
    random.shuffle(labels)
    piezas = []

    for lbl in labels:
        base = os.path.splitext(os.path.basename(lbl))[0]
        # Busca la imagen correspondiente (jpg/png/jpeg).
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(img_dir, base + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        for linea in open(lbl):
            partes = linea.split()
            if len(partes) != 5:
                continue
            cls, cx, cy, w, h = partes
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 12 or y2 - y1 < 12:      # ignora recortes diminutos
                continue
            recorte = img[y1:y2, x1:x2].copy()

            # Escala para que quepa en 'tam_max' conservando proporción.
            rh, rw = recorte.shape[:2]
            escala = tam_max / max(rh, rw)
            if escala < 1:
                recorte = cv2.resize(recorte, (int(rw * escala), int(rh * escala)))
            piezas.append({"img": recorte, "clase": nombres[int(cls)]})
            if len(piezas) >= n_objetivo:
                return piezas
    return piezas


def fondo_banda(ancho, alto):
    fondo = np.full((alto, ancho, 3), 95, np.uint8)
    y0, y1 = int(alto * 0.16), int(alto * 0.84)
    fondo[y0:y1] = (72, 72, 72)
    for x in range(0, ancho, 40):
        cv2.line(fondo, (x, y0), (x, y1), (88, 88, 88), 2)
    cv2.line(fondo, (0, y0), (ancho, y0), (40, 40, 40), 4)
    cv2.line(fondo, (0, y1), (ancho, y1), (40, 40, 40), 4)
    return fondo, y0, y1


def pegar(frame, sprite, cx, cy):
    """Pega 'sprite' centrado en (cx, cy), recortando si sale del borde."""
    h, w = sprite.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sx0, sy0 = fx0 - x0, fy0 - y0
    frame[fy0:fy1, fx0:fx1] = sprite[sy0:sy0 + (fy1 - fy0), sx0:sx0 + (fx1 - fx0)]


def main():
    args = parsear_args()
    random.seed(7)

    piezas = recortar_piezas(args.dataset, args.split, args.tam, args.objetos)
    if not piezas:
        print("[error] no se pudieron recortar piezas. ¿Existe la carpeta del dataset?")
        return
    print(f"[info] {len(piezas)} piezas recortadas del dataset")

    fondo, y0, y1 = fondo_banda(args.ancho, args.alto)
    margen = args.tam // 2 + 8
    objetos = []
    for i, p in enumerate(piezas):
        objetos.append({
            "img": p["img"],
            "x": -margen - i * args.separacion,
            "y": random.randint(y0 + margen, y1 - margen),
        })

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.salida, fourcc, args.fps, (args.ancho, args.alto))

    total = (args.ancho + len(piezas) * args.separacion) // args.velocidad + 30
    for _ in range(total):
        frame = fondo.copy()
        for o in objetos:
            o["x"] += args.velocidad
            if -margen <= o["x"] <= args.ancho + margen:
                pegar(frame, o["img"], o["x"], o["y"])
        out.write(frame)

    out.release()
    print(f"[info] video generado: {args.salida} "
          f"({args.ancho}x{args.alto}, {total} frames)")
    print("[info] pruébalo con:")
    print(f"  python conteo_banda.py --modelo best.pt --fuente {args.salida} "
          f"--linea vertical --guardar deteccion.mp4")


if __name__ == "__main__":
    main()
