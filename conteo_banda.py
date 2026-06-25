"""
conteo_banda.py
---------------
Conteo de objetos sobre una banda transportadora con visión artificial.

Detector: YOLO de Ultralytics (https://github.com/ultralytics/ultralytics).
El modelo detecta los objetos en cada fotograma y su tracker integrado
(ByteTrack) les asigna un id estable; sobre esos ids aplicamos el conteo
por cruce de línea (módulo `contador.py`).

Ejemplos de uso
---------------
  # Cámara web (índice 0), línea horizontal en el centro
  python conteo_banda.py --fuente 0

  # Archivo de video, línea vertical al 60% del ancho
  python conteo_banda.py --fuente banda.mp4 --linea vertical --posicion 0.6

  # Contar solo botellas y tazas, y guardar el video anotado
  python conteo_banda.py --fuente banda.mp4 --clases bottle,cup --guardar salida.mp4

  # Línea a medida con dos puntos (x1,y1,x2,y2)
  python conteo_banda.py --fuente banda.mp4 --puntos 100,400,1180,400

La primera vez descarga automáticamente los pesos del modelo (~5 MB).
"""

import argparse
import sys

import cv2
from ultralytics import YOLO


def parsear_args():
    p = argparse.ArgumentParser(
        description="Conteo de objetos sobre banda transportadora (YOLO).")
    p.add_argument("--fuente", default="0",
                   help="Índice de cámara (0,1,...) o ruta a un video.")
    p.add_argument("--modelo", default="yolo11n.pt",
                   help="Pesos YOLO (yolo11n/s/m.pt). Se descargan solos.")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Confianza mínima de detección (0-1).")
    p.add_argument("--clases", default=None,
                   help="Filtra clases por nombre, separadas por coma. "
                        "Ej: bottle,cup,box. Vacío = todas.")
    p.add_argument("--linea", choices=["horizontal", "vertical"],
                   default="horizontal",
                   help="Orientación de la línea de conteo.")
    p.add_argument("--posicion", type=float, default=0.5,
                   help="Posición de la línea como fracción del frame (0-1).")
    p.add_argument("--puntos", default=None,
                   help="Línea a medida 'x1,y1,x2,y2' (ignora --linea/--posicion).")
    p.add_argument("--guardar", default=None,
                   help="Ruta del video de salida anotado (opcional).")
    p.add_argument("--mostrar", action=argparse.BooleanOptionalAction,
                   default=True, help="Mostrar ventana en vivo (--no-mostrar para ocultar).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Procesar como máximo N fotogramas (0 = sin límite).")
    return p.parse_args()


def resolver_fuente(fuente):
    return int(fuente) if str(fuente).isdigit() else fuente


def calcular_linea(args, ancho, alto):
    if args.puntos:
        x1, y1, x2, y2 = (int(v) for v in args.puntos.split(","))
        return (x1, y1), (x2, y2)
    pos = max(0.0, min(1.0, args.posicion))
    if args.linea == "horizontal":
        y = int(alto * pos)
        return (0, y), (ancho, y)
    x = int(ancho * pos)
    return (x, 0), (x, alto)


def resolver_clases(args, nombres):
    """Convierte 'bottle,cup' en lista de índices según el modelo."""
    if not args.clases:
        return None
    inverso = {v.lower(): k for k, v in nombres.items()}
    indices, desconocidas = [], []
    for c in args.clases.split(","):
        c = c.strip().lower()
        if c in inverso:
            indices.append(inverso[c])
        elif c:
            desconocidas.append(c)
    if desconocidas:
        print(f"[aviso] clases no reconocidas por el modelo: {desconocidas}")
    return indices or None


def main():
    args = parsear_args()

    print(f"[info] cargando modelo {args.modelo} ...")
    model = YOLO(args.modelo)
    clases = resolver_clases(args, model.names)

    cap = cv2.VideoCapture(resolver_fuente(args.fuente))
    if not cap.isOpened():
        print(f"[error] no se pudo abrir la fuente: {args.fuente}")
        sys.exit(1)

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    a, b = calcular_linea(args, ancho, alto)

    # Importación tardía para no acoplar el módulo de conteo a YOLO.
    from contador import ContadorLinea
    etiqueta_ab = "ENTRAN" if args.linea == "horizontal" else "IZQUIERDA"
    etiqueta_ba = "SALEN" if args.linea == "horizontal" else "DERECHA"
    contador = ContadorLinea(a, b, etiqueta_ab, etiqueta_ba)

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

        # Detección + seguimiento en un solo paso. persist=True mantiene
        # los ids entre fotogramas consecutivos.
        resultados = model.track(
            frame, persist=True, conf=args.conf, classes=clases,
            tracker="bytetrack.yaml", verbose=False)
        r = resultados[0]

        ids_vivos = []
        if r.boxes is not None and r.boxes.id is not None:
            cajas = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, cls in zip(cajas, ids, clss):
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                nombre = model.names.get(int(cls), str(cls))
                ids_vivos.append(tid)

                cruce = contador.actualizar(tid, (cx, cy), nombre)

                # Resaltar en verde brillante el fotograma en que cruza.
                color = (0, 0, 255) if cruce else (0, 220, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, f"{nombre} #{tid}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        contador.olvidar(ids_vivos)
        contador.dibujar(frame, mostrar_clases=True)

        if escritor is not None:
            escritor.write(frame)
        if args.mostrar:
            cv2.imshow("Conteo en banda - YOLO", frame)
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
    print(f"{etiqueta_ab:9}: {contador.conteo_ab}")
    print(f"{etiqueta_ba:9}: {contador.conteo_ba}")
    if contador.por_clase:
        print("Por clase:")
        for nombre, c in sorted(contador.por_clase.items(), key=lambda kv: -kv[1]):
            print(f"  {nombre}: {c}")


if __name__ == "__main__":
    main()
