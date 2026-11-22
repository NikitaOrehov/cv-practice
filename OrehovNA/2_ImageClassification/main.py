# main.py
import argparse
import os
import pickle
import numpy as np

from feature_extractor import BaseFeatureExtractor
from bag_of_words import BagOfWords
from classifier import Classifier
from dataset import load_dataset, Dataset
from visualization import plot_histogram, draw_keypoints


def save_class_names(path, names):
    with open(path, 'wb') as f:
        pickle.dump(names, f)


def load_class_names(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'test', 'predict'], required=True)
    ap.add_argument('--data', help='Путь до trainval/ или test/')
    ap.add_argument('--model', default='model.pkl')
    ap.add_argument('--bow', default='bow.pkl')
    ap.add_argument('--classes', default='classes.pkl')
    ap.add_argument('--method', default='sift',
                    choices=BaseFeatureExtractor.available_methods())
    ap.add_argument('--vocab_size', type=int, default=200)
    ap.add_argument('--image', help='Картинка для predict')
    ap.add_argument('--plot', action='store_true')

    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--layer', default=None)
    ap.add_argument('--transform-size', type=int, default=None)
    ap.add_argument('--gem', action='store_true')
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--C', type=float, default=1.0)

    args = ap.parse_args()

    def make_extractor(method: str) -> BaseFeatureExtractor:
        return BaseFeatureExtractor.create(
            method,
            grid=args.grid,
            layer=args.layer,
            transform_size=args.transform_size,
        )

    # ---------- TRAIN ----------
    if args.mode == 'train':
        for p in (args.bow, args.model, args.classes):
            parent = os.path.dirname(p)
            if parent:
                os.makedirs(parent, exist_ok=True)

        print(f"[train] Загрузка {args.data}...")
        train_ds = load_dataset(args.data, include_other=True)
        print(f"  изображений: {len(train_ds)}, классов: {len(train_ds.class_names)}")

        extractor = make_extractor(args.method)
        bow = BagOfWords(
            vocabulary_size=args.vocab_size,
            feature_extractor=extractor,
            method_name=args.method,
            gem_pooling=args.gem,
        )

        if not args.gem:
            print("[train] Построение словаря BoW...")
            bow.fit(train_ds, max_descriptors=300_000)
        else:
            print("[train] GeM pooling: словарь не строится")
            bow.is_fitted = True

        print("[train] Преобразование trainval в признаки...")
        X = bow.transform_dataset(train_ds)

        print("[train] Обучение SVM...")
        cw = 'balanced' if args.balanced else None
        clf = Classifier(kernel='linear', C=args.C, class_weight=cw)
        clf.train(X, train_ds.labels)

        bow.save(args.bow)
        clf.save(args.model)
        save_class_names(args.classes, train_ds.class_names)
        print(f"[train] Готово: {args.bow}, {args.model}, {args.classes}")

    # ---------- TEST ----------
    elif args.mode == 'test':
        print(f"[test] Загрузка {args.data}...")
        test_ds = load_dataset(args.data, include_other=True)
        print(f"  изображений: {len(test_ds)}")

        with open(args.bow, 'rb') as f:
            method_saved = pickle.load(f).get('method_name', args.method)
        print(f"  метод извлечения (из bow.pkl): {method_saved}")

        bow = BagOfWords(feature_extractor=make_extractor(method_saved)).load(args.bow)
        clf = Classifier().load(args.model)

        print("[test] Извлечение признаков...")
        X = bow.transform_dataset(test_ds)
        print("[test] Метрики:")
        clf.evaluate(X, test_ds.labels)

    # ---------- PREDICT ----------
    else:
        if not args.image:
            raise SystemExit("Для --mode predict нужен --image")

        with open(args.bow, 'rb') as f:
            method_saved = pickle.load(f).get('method_name', args.method)
        bow = BagOfWords(feature_extractor=make_extractor(method_saved)).load(args.bow)
        clf = Classifier().load(args.model)
        class_names = load_class_names(args.classes)

        img = Dataset.read_one(args.image)

        feat = bow.transform(img).reshape(1, -1)
        proba = clf.model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))

        print(f"\nИзображение: {args.image}")
        print(f"Предсказание: {class_names[pred_idx]}")
        print("Топ-5 классов:")
        for i in np.argsort(proba)[::-1][:5]:
            print(f"  {class_names[i]:45s}  p={proba[i]:.3f}")

        if args.plot:
            out_dir = os.path.join('saved', 'predictions', method_saved)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.image))[0]
            pred_class = class_names[pred_idx]

            plot_histogram(img, feat.flatten(),
                           save_path=os.path.join(out_dir,
                                                  f"{base}__pred-{pred_class}_bow.png"),
                           title=pred_class)

            if method_saved in ('sift', 'orb', 'kaze'):
                draw_keypoints(img, make_extractor(method_saved),
                               save_path=os.path.join(out_dir,
                                                      f"{base}__pred-{pred_class}_kp.png"))

            print(f"Визуализация сохранена в {out_dir}\\")


if __name__ == '__main__':
    main()