"""
conteo_demo.py
--------------
Conteo con visión artificial CLÁSICA (sin red neuronal), pensado para un
entorno controlado como una banda transportadora: fondo estático y objetos
en movimiento.

Pipeline:
  1. Sustracción de fondo (MOG2) -> separa lo que se mueve del fondo fijo.
  2. Morfología + contornos -> obtiene los objetos y sus centroides.
  3. Seguidor de centroides (`seguidor.py`) -> asigna un id a cada objeto.
  4. Conteo por cruce de línea (`contador.py`).

Es ideal para PROBAR el sistema con el video sintético:
  python generar_demo.py
  python conteo_demo.py --fuente demo_banda.mp4

También funciona con tu propia cámara o video si el fondo es estable:
  python conteo_demo.py --fuente 0 --linea vertical --posicion 0.5
"""

import argparse
import sys

import cv2

from contador import ContadorLinea
from seguidor import SeguidorCentroides


def parsear_args():
    p = argparse.ArgumentParser(
        description="Conteo en banda con visión clásica (OpenCV).")
    p.add_argument("--fuente", default="demo_banda.mp4",
                   help="Índice de cámara o ruta a un video.")
    p.add_argument("--linea", choices=["horizontal", "vertical"],
                   default="vertical", help="Orientación de la línea de conteo.")
    p.add_argument("--posicion", type=float, default=0.5,
                   help="Posición de la línea como fracción del frame (0-1).")
    p.add_argument("--area-min", type=int, default=500,
                   help="Área mínima (px) para considerar un contorno objeto.")
    p.add_argument("--guardar", default=None, help="Ruta del video anotado.")
    p.add_argument("--mostrar", action=argparse.BooleanOptionalAction,
                   default=True, help="Mostrar ventana en vivo.")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def resolver_fuente(fuente):
    return int(fuente) if str(fuente).isdigit() else fuente


def main():
    args = parsear_args()

    cap = cv2.VideoCapture(resolver_fuente(args.fuente))
    if not cap.isOpened():
        print(f"[error] no se pudo abrir la fuente: {args.fuente}")
        sys.exit(1)

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pos = max(0.0, min(1.0, args.posicion))
    if args.linea == "horizontal":
        a, b = (0, int(alto * pos)), (ancho, int(alto * pos))
        et_ab, et_ba = "ENTRAN", "SALEN"
    else:
        a, b = (int(ancho * pos), 0), (int(ancho * pos), alto)
        et_ab, et_ba = "IZQUIERDA", "DERECHA"

    contador = ContadorLinea(a, b, et_ab, et_ba)
    seguidor = SeguidorCentroides(dist_max=80, max_perdidos=20)
    restador = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    escritor = None
    if args.guardar:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        escritor = cv2.VideoWriter(args.guardar, fourcc, fps, (ancho, alto))

    print("[info] procesando... (q o Esc para salir)")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        mascara = restador.apply(frame)
        _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroides, cajas = [], []
        for c in contornos:
            if cv2.contourArea(c) < args.area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            centroides.append((x + w / 2, y + h / 2))
            cajas.append((x, y, w, h))

        objetos = seguidor.actualizar(centroides)

        # Dibuja cajas detectadas.
        for (x, y, w, h) in cajas:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

        # Asocia cada id a su centroide y aplica el conteo.
        for tid, (cx, cy) in objetos.items():
            cruce = contador.actualizar(tid, (cx, cy))
            color = (0, 0, 255) if cruce else (0, 220, 0)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
            cv2.putText(frame, f"#{tid}", (int(cx) - 10, int(cy) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        contador.dibujar(frame)

        if escritor is not None:
            escritor.write(frame)
        if args.mostrar:
            cv2.imshow("Conteo en banda - clasico (OpenCV)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    if escritor is not None:
        escritor.release()
        print(f"[info] video guardado en: {args.guardar}")
    cv2.destroyAllWindows()

    print("\n===== RESULTADO =====")
    print(f"Total contados : {contador.total}")
    print(f"{et_ab:9}: {contador.conteo_ab}")
    print(f"{et_ba:9}: {contador.conteo_ba}")


if __name__ == "__main__":
    main()
