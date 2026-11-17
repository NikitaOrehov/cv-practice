# dataset.py
import os
import cv2
from glob import glob

SUBDATASETS = ['NNSUDataset', 'ExtDataset']
CLASS_OTHER = '00_Other'

def _discover_classes(scan_paths):
    """Собирает все имена классов из всех подпапок, объединяя их в множество."""
    classes = set()
    for p in scan_paths:
        for d in os.listdir(p):
            full = os.path.join(p, d)
            if os.path.isdir(full):
                classes.add(d)
    return sorted(classes)   # 00_Other окажется первым — это ок

def load_split(root_dir, include_other=True):
    """
    root_dir — путь до trainval/ или test/.
    Возвращает: (image_paths, labels, class_names)
    image_paths — список строк (пути), НЕ массивы. Картинки читаются позже,
    чтобы не держать 28 классов в RAM одновременно.
    """
    scan_paths = [os.path.join(root_dir, s)
                  for s in SUBDATASETS
                  if os.path.isdir(os.path.join(root_dir, s))]
    if not scan_paths:
        scan_paths = [root_dir]

    class_names = _discover_classes(scan_paths)
    if not include_other:
        class_names = [c for c in class_names if c != CLASS_OTHER]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for sp in scan_paths:
        for cls in class_names:
            cls_dir = os.path.join(sp, cls)
            if not os.path.isdir(cls_dir):
                continue
            # Поддерживаемые расширения — стандартные для этого датасета
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                for img_path in glob(os.path.join(cls_dir, ext)):
                    # Проверка, что файл читается (быстро через imread с флагом "проверить")
                    paths.append(img_path)
                    labels.append(class_to_idx[cls])

    import numpy as np
    return paths, np.array(labels), class_names