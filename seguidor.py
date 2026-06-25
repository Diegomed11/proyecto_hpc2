"""
seguidor.py
-----------
Seguidor de centroides muy simple (vecino más cercano).

Solo se usa en el modo demo (`conteo_demo.py`), porque el detector clásico
de OpenCV no asigna identificadores a los objetos. En el modo YOLO el
seguimiento lo hace el propio tracker (ByteTrack/BoT-SORT) de Ultralytics,
así que este módulo no interviene.

Asigna un id entero estable a cada objeto emparejando los centroides de un
fotograma con los del anterior por cercanía. Si un objeto desaparece varios
fotogramas seguidos, se descarta su id.
"""

import numpy as np


class SeguidorCentroides:
    def __init__(self, dist_max=70, max_perdidos=20):
        self.sig_id = 0
        self.dist_max = dist_max
        self.max_perdidos = max_perdidos
        self.objetos = {}    # id -> (cx, cy)
        self.perdidos = {}   # id -> fotogramas sin verse

    def actualizar(self, centroides):
        """Recibe lista de (cx, cy) y devuelve dict id -> (cx, cy)."""
        # Sin detecciones: envejecer todo y eliminar lo perdido.
        if len(centroides) == 0:
            for tid in list(self.perdidos):
                self.perdidos[tid] += 1
                if self.perdidos[tid] > self.max_perdidos:
                    self.objetos.pop(tid, None)
                    self.perdidos.pop(tid, None)
            return {}

        centroides = [tuple(map(float, c)) for c in centroides]

        # Sin objetos previos: registrar todos como nuevos.
        if len(self.objetos) == 0:
            for c in centroides:
                self._registrar(c)
            return dict(self.objetos)

        ids = list(self.objetos.keys())
        prev = np.array([self.objetos[i] for i in ids])
        nuevos = np.array(centroides)

        # Matriz de distancias entre objetos previos y nuevos centroides.
        d = np.linalg.norm(prev[:, None, :] - nuevos[None, :, :], axis=2)

        filas = d.min(axis=1).argsort()
        usados_col = set()
        usados_fila = set()

        for fila in filas:
            col = d[fila].argmin()
            if col in usados_col:
                continue
            if d[fila, col] > self.dist_max:
                continue
            tid = ids[fila]
            self.objetos[tid] = centroides[col]
            self.perdidos[tid] = 0
            usados_fila.add(fila)
            usados_col.add(col)

        # Filas (objetos previos) no emparejadas -> envejecer.
        for i, tid in enumerate(ids):
            if i not in usados_fila:
                self.perdidos[tid] += 1
                if self.perdidos[tid] > self.max_perdidos:
                    self.objetos.pop(tid, None)
                    self.perdidos.pop(tid, None)

        # Columnas (centroides nuevos) no emparejadas -> objetos nuevos.
        for j, c in enumerate(centroides):
            if j not in usados_col:
                self._registrar(c)

        return dict(self.objetos)

    def _registrar(self, centroide):
        self.objetos[self.sig_id] = centroide
        self.perdidos[self.sig_id] = 0
        self.sig_id += 1
