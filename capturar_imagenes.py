"""
capturar_imagenes.py
--------------------
Captura imágenes para construir el dataset de entrenamiento (tornillos y
rondanas). Toma fotos desde la cámara o un video y las guarda en una carpeta;
luego esas imágenes se suben a Roboflow para etiquetarlas.

Consejos para un buen dataset de tornillos/rondanas:
  - Muchos ángulos, posiciones y rotaciones de cada pieza.
  - Variar iluminación y fondo (idealmente la banda real).
  - Incluir piezas solas, juntas, tocándose y parcialmente tapadas.
  - Apunta a 150-300 imágenes en total para empezar (más = mejor).

Uso:
  # Guardar un fotograma cada vez que pulses ESPACIO (cámara 0)
  python capturar_imagenes.py --fuente 0

  # Guardar automáticamente 1 de cada 15 fotogramas de un video
  python capturar_imagenes.py --fuente piezas.mp4 --cada 15

Teclas: ESPACIO = guardar foto · q / Esc = salir
"""

import argparse
import os

import cv2


def parsear_args():
    p = argparse.ArgumentParser(description="Captura imágenes para el dataset.")
    p.add_argument("--fuente", default="0", help="Índice de cámara o ruta de video.")
    p.add_argument("--salida", default="dataset_crudo", help="Carpeta destino.")
    p.add_argument("--prefijo", default="img", help="Prefijo del nombre de archivo.")
    p.add_argument("--cada", type=int, default=0,
                   help="Guardar automáticamente 1 de cada N fotogramas (0 = solo manual).")
    return p.parse_args()


def main():
    args = parsear_args()
    os.makedirs(args.salida, exist_ok=True)

    fuente = int(args.fuente) if str(args.fuente).isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"[error] no se pudo abrir la fuente: {args.fuente}")
        return

    # Continúa numerando si ya había imágenes en la carpeta.
    existentes = [f for f in os.listdir(args.salida) if f.startswith(args.prefijo)]
    contador = len(existentes)
    n_frame = 0
    print("[info] ESPACIO = guardar · q/Esc = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frame += 1
        guardar = False

        if args.cada > 0 and n_frame % args.cada == 0:
            guardar = True

        vista = frame.copy()
        cv2.putText(vista, f"Guardadas: {contador}   (ESPACIO=foto, q=salir)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Captura de dataset", vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 32:            # ESPACIO
            guardar = True
        elif tecla in (ord("q"), 27):
            break

        if guardar:
            ruta = os.path.join(args.salida, f"{args.prefijo}_{contador:04d}.jpg")
            cv2.imwrite(ruta, frame)
            contador += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[info] {contador} imágenes en la carpeta '{args.salida}'")
    print("[info] siguiente paso: súbelas a Roboflow para etiquetarlas.")


if __name__ == "__main__":
    main()
