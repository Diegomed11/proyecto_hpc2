"""
contador.py
-----------
Lógica de conteo por cruce de línea virtual.

Esta clase es independiente del detector: lo único que necesita es, para
cada objeto, un identificador estable (track id) y la posición de su
centroide (cx, cy). Por eso la reutilizan tanto el modo YOLO
(`conteo_banda.py`) como el modo clásico de demostración (`conteo_demo.py`).

Idea: una línea recta divide el plano en dos lados. Para un punto P se
calcula el signo del producto cruz respecto al segmento A-B. Cuando un
mismo objeto cambia de signo entre un fotograma y el siguiente, es porque
cruzó la línea, y se cuenta una vez en la dirección correspondiente.
"""

from collections import defaultdict

import cv2


def lado_de_linea(p, a, b):
    """Signo del lado de la línea A-B en el que cae el punto P.

    Devuelve un número > 0, < 0 o == 0 (sobre la línea). El signo es
    consistente: todos los puntos de un mismo lado comparten signo.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class ContadorLinea:
    """Cuenta objetos que cruzan una línea virtual, separando por dirección."""

    def __init__(self, a, b, etiqueta_ab="ENTRAN", etiqueta_ba="SALEN"):
        self.a = (int(a[0]), int(a[1]))
        self.b = (int(b[0]), int(b[1]))
        self.etiqueta_ab = etiqueta_ab
        self.etiqueta_ba = etiqueta_ba

        self._ultimo_signo = {}          # track_id -> último signo conocido (-1/1)
        self.conteo_ab = 0               # cruces del lado negativo al positivo
        self.conteo_ba = 0               # cruces del lado positivo al negativo
        self.por_clase = defaultdict(int)  # nombre_clase -> total contado

    def actualizar(self, track_id, centroide, nombre_clase=None):
        """Procesa un objeto. Devuelve 'AB', 'BA' o None si no cruzó."""
        s = lado_de_linea(centroide, self.a, self.b)
        signo = 1 if s > 0 else (-1 if s < 0 else 0)

        previo = self._ultimo_signo.get(track_id)
        cruce = None

        if previo is not None and signo != 0 and previo != 0 and signo != previo:
            if signo > 0:
                self.conteo_ab += 1
                cruce = "AB"
            else:
                self.conteo_ba += 1
                cruce = "BA"
            if nombre_clase is not None:
                self.por_clase[nombre_clase] += 1

        if signo != 0:
            self._ultimo_signo[track_id] = signo

        return cruce

    @property
    def total(self):
        return self.conteo_ab + self.conteo_ba

    def olvidar(self, ids_vivos):
        """Libera el estado de ids que ya no están en escena (opcional)."""
        ids_vivos = set(ids_vivos)
        for tid in list(self._ultimo_signo):
            if tid not in ids_vivos:
                del self._ultimo_signo[tid]

    # ------------------------------------------------------------------ dibujo
    def dibujar(self, frame, mostrar_clases=False):
        """Dibuja la línea y un panel con los conteos sobre el frame."""
        # Línea de conteo
        cv2.line(frame, self.a, self.b, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, self.a, 5, (0, 255, 255), -1)
        cv2.circle(frame, self.b, 5, (0, 255, 255), -1)

        # Panel semitransparente arriba a la izquierda
        lineas = [
            f"TOTAL: {self.total}",
            f"{self.etiqueta_ab}: {self.conteo_ab}",
            f"{self.etiqueta_ba}: {self.conteo_ba}",
        ]
        if mostrar_clases and self.por_clase:
            top = sorted(self.por_clase.items(), key=lambda kv: -kv[1])[:6]
            lineas.append("- por clase -")
            lineas += [f"{n}: {c}" for n, c in top]

        ancho_panel = 240
        alto_panel = 28 * len(lineas) + 16
        sub = frame[10:10 + alto_panel, 10:10 + ancho_panel].copy()
        rect = sub.copy()
        rect[:] = (30, 30, 30)
        cv2.addWeighted(rect, 0.55, sub, 0.45, 0, sub)
        frame[10:10 + alto_panel, 10:10 + ancho_panel] = sub

        y = 38
        for i, texto in enumerate(lineas):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, texto, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
            y += 28
        return frame
