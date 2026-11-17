# io_utils.py
import numpy as np
import cv2

def imread_unicode(path):
    """
    Читает изображение по пути, поддерживающему Unicode (Windows + кириллица).
    Возвращает np.ndarray BGR или None, если файл битый/нечитаемый.
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None