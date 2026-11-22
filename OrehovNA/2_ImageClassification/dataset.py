# dataset.py
import os
import cv2
import numpy as np
from glob import glob

SUBDATASETS = ['NNSUDataset', 'ExtDataset']
CLASS_OTHER = '00_Other'


class Dataset:
    def __init__(self, paths: list[str], labels: np.ndarray,
                 class_names: list[str]):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray | None, int]:
        img = self._read(self.paths[idx])
        return img, int(self.labels[idx])

    def _read(self, path: str) -> np.ndarray | None:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size == 0:
                return None
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            return None

    @staticmethod
    def read_one(path: str) -> np.ndarray:
        """Прочитать конкретный путь (для predict). Бросает, если не читается."""
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size == 0:
                raise FileNotFoundError(path)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            return img
        except OSError as e:
            raise FileNotFoundError(path) from e

    def iter_images(self):
        """Итератор (image, label) по всем изображениям, пропуская битые."""
        for i in range(len(self)):
            img, label = self[i]
            if img is None:
                continue
            yield img, label

    def subset(self, indices) -> "Dataset":
        """Подвыборка — для отладки на N изображениях."""
        return Dataset(
            [self.paths[i] for i in indices],
            self.labels[indices],
            self.class_names,
        )


# ---------- Загрузка из директории ----------

def _discover_classes(scan_paths):
    classes = set()
    for p in scan_paths:
        for d in os.listdir(p):
            if os.path.isdir(os.path.join(p, d)):
                classes.add(d)
    return sorted(classes)


def load_dataset(root_dir: str, include_other: bool = True) -> Dataset:
    """
    Загружает trainval/ или test/ с объединением NNSUDataset + ExtDataset.
    Дедупликация по realpath (Windows + регистронезависимая ФС).
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
    seen = set()
    for sp in scan_paths:
        for cls in class_names:
            cls_dir = os.path.join(sp, cls)
            if not os.path.isdir(cls_dir):
                continue
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                for img_path in glob(os.path.join(cls_dir, ext)):
                    real = os.path.realpath(img_path)
                    if real in seen:
                        continue
                    seen.add(real)
                    paths.append(img_path)
                    labels.append(class_to_idx[cls])

    return Dataset(paths, np.array(labels), class_names)


def load_split(root_dir: str, include_other: bool = True) -> Dataset:
    """Алиас для обратной совместимости."""
    return load_dataset(root_dir, include_other)