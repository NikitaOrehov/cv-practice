# main.py
import argparse
import os
import cv2
import numpy as np
import pickle

from feature_extractor import SIFTExtractor, ORBExtractor, CNNExtractor
from bag_of_words import BagOfWords
from classifier import Classifier
from dataset import load_split
from visualization import plot_histogram, draw_keypoints
from io_utils import imread_unicode


def get_extractor(name):
    name = name.lower()
    if name == 'sift':
        return SIFTExtractor()
    if name == 'orb':
        return ORBExtractor()
    if name == 'resnet18':
        return CNNExtractor('resnet18')
    if name == 'vgg16':
        return CNNExtractor('vgg16')
    raise ValueError(f"Неизвестный метод: {name}")


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
    ap.add_argument('--classes', default='classes.pkl', help='Файл с именами классов')
    ap.add_argument('--method', default='sift',
                    help='sift | orb | resnet18 | vgg16 (игнорируется в test/predict, '
                         'если метод сохранён в bow.pkl)')
    ap.add_argument('--vocab_size', type=int, default=200)
    ap.add_argument('--image', help='Картинка для режима predict')
    ap.add_argument('--plot', action='store_true',
                    help='Сохранить визуализацию гистограммы в режиме predict')
    args = ap.parse_args()

    # ---------- TRAIN ----------
    if args.mode == 'train':
        print(f"[train] Загрузка списка файлов из {args.data}...")
        paths, labels, class_names = load_split(args.data, include_other=True)
        print(f"  всего изображений: {len(paths)}")
        print(f"  классов: {len(class_names)}")
        for i, c in enumerate(class_names):
            print(f"    [{i}] {c}")

        extractor = get_extractor(args.method)
        bow = BagOfWords(args.vocab_size, extractor, method_name=args.method)
        print("[train] Построение словаря BoW...")
        bow.fit(paths, max_descriptors=300_000)

        print("[train] Преобразование trainval в гистограммы...")
        X = bow.transform_batch(paths)

        print("[train] Обучение SVM...")
        clf = Classifier(kernel='linear', C=1.0)
        clf.train(X, labels)

        bow.save(args.bow)
        clf.save(args.model)
        save_class_names(args.classes, class_names)
        print(f"[train] Готово. Сохранено: {args.bow}, {args.model}, {args.classes}")

    # ---------- TEST ----------
    elif args.mode == 'test':
        print(f"[test] Загрузка {args.data}...")
        paths, labels, _ = load_split(args.data, include_other=True)
        print(f"  тестовых изображений: {len(paths)}")

        # Восстанавливаем экстрактор по сохранённому method_name — исключаем рассинхрон
        with open(args.bow, 'rb') as f:
            method_saved = pickle.load(f).get('method_name', args.method)
        extractor = get_extractor(method_saved)
        print(f"  метод извлечения (из bow.pkl): {method_saved}")

        bow = BagOfWords(feature_extractor=extractor).load(args.bow)
        clf = Classifier().load(args.model)

        print("[test] Извлечение признаков...")
        X = bow.transform_batch(paths)
        print("[test] Метрики:")
        clf.evaluate(X, labels)

    # ---------- PREDICT ----------
    else:
        if not args.image:
            raise SystemExit("Для --mode predict нужен --image")
        with open(args.bow, 'rb') as f:
            method_saved = pickle.load(f).get('method_name', args.method)
        extractor = get_extractor(method_saved)
        bow = BagOfWords(feature_extractor=extractor).load(args.bow)
        clf = Classifier().load(args.model)
        class_names = load_class_names(args.classes)

        img = imread_unicode(args.image)
        if img is None:
            raise FileNotFoundError(args.image)

        hist = bow.transform(img, strict=True).reshape(1, -1)
        proba = clf.model.predict_proba(hist)[0]
        pred_idx = int(np.argmax(proba))

        print(f"\nИзображение: {args.image}")
        print(f"Предсказание: {class_names[pred_idx]}")
        print("Топ-5 классов:")
        order = np.argsort(proba)[::-1][:5]
        for i in order:
            print(f"  {class_names[i]:45s}  p={proba[i]:.3f}")

        if args.plot:
            plot_histogram(img, hist.flatten(),
                           save_path=os.path.splitext(args.image)[0] + '_bow.png',
                           title=class_names[pred_idx])
            if method_saved in ('sift', 'orb'):
                draw_keypoints(img, extractor,
                               save_path=os.path.splitext(args.image)[0] + '_kp.png')
            print("Визуализация сохранена рядом с изображением.")


if __name__ == '__main__':
    main()