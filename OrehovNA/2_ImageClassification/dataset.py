# dataset.py
import os
import cv2
import numpy as np
from glob import glob

SUBDATASETS = ['NNSUDataset', 'ExtDataset']

def _discover_classes(scan_paths):
    classes = set()
    for p in scan_paths:
        for d in os.listdir(p):
            full = os.path.join(p, d)
            if os.path.isdir(full):
                classes.add(d)
    return sorted(classes)

def load_split(root_dir):
    scan_paths = [os.path.join(root_dir, s)
                  for s in SUBDATASETS
                  if os.path.isdir(os.path.join(root_dir, s))]
    if not scan_paths:
        scan_paths = [root_dir]

    class_names = _discover_classes(scan_paths)

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

    return paths, np.array(labels), class_names