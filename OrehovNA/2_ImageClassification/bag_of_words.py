# bag_of_words.py
import cv2
import numpy as np
from io_utils import imread_unicode
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class BagOfWords:
    def __init__(self, vocabulary_size=200, feature_extractor=None, method_name=None):
        self.vocabulary_size = vocabulary_size
        self.extractor = feature_extractor
        self.method_name = method_name
        self.kmeans = None
        self.is_fitted = False

    def fit(self, image_paths, max_descriptors=500_000, seed=42):
        rng = np.random.default_rng(seed)

        shuffled = list(image_paths)
        rng.shuffle(shuffled)

        total = 0
        chunks = []
        for path in shuffled:
            img = imread_unicode(path)
            if img is None:
                continue
            desc = self.extractor.extract(img)
            if desc.size == 0:
                continue
            chunks.append(desc)
            total += len(desc)
            if total >= max_descriptors:
                break

        if not chunks:
            raise ValueError("Не удалось извлечь ни одного дескриптора. "
                             "Проверьте, что пути корректны и изображения не битые.")

        all_desc = np.vstack(chunks)
        if len(all_desc) > max_descriptors:
            idx = rng.choice(len(all_desc), max_descriptors, replace=False)
            all_desc = all_desc[idx]

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocabulary_size,
            batch_size=10000,
            random_state=seed,
            n_init='auto',
        )
        self.kmeans.fit(all_desc)
        self.is_fitted = True
        print(f"[BoW] Словарь готов: {self.vocabulary_size} слов, "
              f"дескрипторов {len(all_desc)}")
        return self

    def _extract(self, image_input):
        """Возвращает (img_bgr, descriptors) или (None, None) если не прочитать."""
        if isinstance(image_input, str):
            img = imread_unicode(image_input)
        else:
            img = image_input
        if img is None:
            return None, None
        return img, self.extractor.extract(img)

    def transform(self, image_input, strict=False) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Сначала вызовите fit().")
        img, desc = self._extract(image_input)
        if img is None:
            if strict:
                raise FileNotFoundError(image_input)
            print(f"[BoW][skip] не удалось прочитать: {image_input}")
            return np.zeros(self.vocabulary_size)
        if desc.size == 0:
            return np.zeros(self.vocabulary_size)
        labels = self.kmeans.predict(desc)
        hist, _ = np.histogram(labels, bins=np.arange(self.vocabulary_size + 1))
        return normalize(hist.reshape(1, -1), norm='l2').flatten()
    
    def transform_batch(self, image_paths) -> np.ndarray:
        return np.array([self.transform(p) for p in image_paths])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'size': self.vocabulary_size,
                'method_name': self.method_name,
                'is_fitted': self.is_fitted,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.kmeans = data['kmeans']
        self.vocabulary_size = data['size']
        self.method_name = data.get('method_name')
        self.is_fitted = data['is_fitted']
        return self