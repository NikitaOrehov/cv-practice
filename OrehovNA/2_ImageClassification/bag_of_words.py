# bag_of_words.py
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class BagOfWords:
    def __init__(self, vocabulary_size=200, feature_extractor=None,
                 method_name=None, gem_pooling=False):
        self.vocabulary_size = vocabulary_size
        self.extractor = feature_extractor
        self.method_name = method_name
        self.gem_pooling = gem_pooling
        self.kmeans = None
        self.is_fitted = False

    def fit(self, dataset, max_descriptors=300_000, seed=42):
        """
        dataset: Dataset — читает изображения лениво через __getitem__.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(dataset))

        total = 0
        chunks = []
        for idx in order:
            img, _ = dataset[idx]
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
            raise ValueError("Не удалось извлечь ни одного дескриптора.")

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

    def _gem(self, X: np.ndarray, p: float = 3.0) -> np.ndarray:
        X = np.clip(X, 1e-6, None)
        pooled = np.power(np.mean(np.power(X, p), axis=0), 1.0 / p)
        return normalize(pooled.reshape(1, -1), norm='l2').flatten()

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Принимает уже прочитанное изображение (не путь)."""
        if not self.is_fitted:
            raise RuntimeError("Сначала вызовите fit().")
        desc = self.extractor.extract(img)
        if desc.size == 0:
            return np.zeros(self.vocabulary_size)
        if self.gem_pooling:
            return self._gem(desc)
        labels = self.kmeans.predict(desc)
        hist, _ = np.histogram(labels, bins=np.arange(self.vocabulary_size + 1))
        return normalize(hist.reshape(1, -1), norm='l2').flatten()

    def transform_dataset(self, dataset) -> np.ndarray:
        """Проходит по датасету, читая через __getitem__."""
        features = []
        for i in range(len(dataset)):
            img, _ = dataset[i]
            if img is None:
                features.append(np.zeros(self.vocabulary_size))
                continue
            features.append(self.transform(img))
        return np.array(features)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'size': self.vocabulary_size,
                'method_name': self.method_name,
                'gem_pooling': self.gem_pooling,
                'is_fitted': self.is_fitted,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.kmeans = data['kmeans']
        self.vocabulary_size = data['size']
        self.method_name = data.get('method_name')
        self.gem_pooling = data.get('gem_pooling', False)
        self.is_fitted = data['is_fitted']
        return self