"""
generar_demo.py
---------------
Genera un video sintético de una banda transportadora con objetos que se
desplazan de izquierda a derecha. Sirve para probar el conteo sin necesidad
de una cámara ni de descargar videos.

  python generar_demo.py                # crea demo_banda.mp4
  python generar_demo.py --objetos 25 --salida banda.mp4
"""

import argparse

import cv2
import numpy as np


def parsear_args():
    p = argparse.ArgumentParser(description="Genera un video demo de banda.")
    p.add_argument("--salida", default="demo_banda.mp4")
    p.add_argument("--ancho", type=int, default=960)
    p.add_argument("--alto", type=int, default=540)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--objetos", type=int, default=18,
                   help="Número total de objetos que cruzarán la banda.")
    p.add_argument("--velocidad", type=int, default=4,
                   help="Píxeles por fotograma que avanza cada objeto.")
    return p.parse_args()


def fondo_banda(ancho, alto):
    """Crea la imagen base de la banda (gris con bordes y rodillos)."""
    fondo = np.full((alto, ancho, 3), 90, np.uint8)
    cinta_y0, cinta_y1 = int(alto * 0.18), int(alto * 0.82)
    fondo[cinta_y0:cinta_y1] = (70, 70, 70)
    # Rodillos transversales (líneas más claras) para dar textura.
    for x in range(0, ancho, 40):
        cv2.line(fondo, (x, cinta_y0), (x, cinta_y1), (85, 85, 85), 2)
    # Bordes de la cinta.
    cv2.line(fondo, (0, cinta_y0), (ancho, cinta_y0), (40, 40, 40), 4)
    cv2.line(fondo, (0, cinta_y1), (ancho, cinta_y1), (40, 40, 40), 4)
    return fondo, cinta_y0, cinta_y1


def main():
    args = parsear_args()
    fondo, y0, y1 = fondo_banda(args.ancho, args.alto)
    rng = np.random.default_rng(7)

    colores = [(60, 76, 231), (113, 204, 46), (219, 152, 52),
               (182, 89, 155), (94, 73, 52), (15, 196, 241)]

    # Genera objetos escalonados para que no se solapen al entrar.
    separacion = 100
    objetos = []
    for i in range(args.objetos):
        r = int(rng.integers(16, 26))
        objetos.append({
            "x": -r - i * separacion,
            "y": int(rng.integers(y0 + r + 10, y1 - r - 10)),
            "r": r,
            "color": colores[i % len(colores)],
            "forma": "circulo" if i % 2 == 0 else "cuadrado",
        })

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.salida, fourcc, args.fps, (args.ancho, args.alto))

    # Suficientes fotogramas para que el último objeto salga del cuadro.
    total_frames = (args.ancho + args.objetos * separacion) // args.velocidad + 30
    for _ in range(total_frames):
        frame = fondo.copy()
        for o in objetos:
            o["x"] += args.velocidad
            x, y, r = int(o["x"]), int(o["y"]), o["r"]
            if -r <= x <= args.ancho + r:
                if o["forma"] == "circulo":
                    cv2.circle(frame, (x, y), r, o["color"], -1)
                    cv2.circle(frame, (x, y), r, (20, 20, 20), 2)
                else:
                    cv2.rectangle(frame, (x - r, y - r), (x + r, y + r),
                                  o["color"], -1)
                    cv2.rectangle(frame, (x - r, y - r), (x + r, y + r),
                                  (20, 20, 20), 2)
        out.write(frame)

    out.release()
    print(f"[info] video demo generado: {args.salida} "
          f"({args.ancho}x{args.alto}, {total_frames} frames, {args.objetos} objetos)")


if __name__ == "__main__":
    main()
