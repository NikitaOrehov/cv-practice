# classifier.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle


class Classifier:
    def __init__(self, kernel='linear', C=1.0, class_weight=None):
        """
        class_weight='balanced' — компенсация дисбаланса классов.
        Крайне рекомендован для этого датасета (соотношение 19:1).
        """
        self.model = SVC(kernel=kernel, C=C, probability=True,
                         class_weight=class_weight, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[Classifier] Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, zero_division=0))
        return acc

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self