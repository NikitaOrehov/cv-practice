# Лабораторная работа №2: Классификация изображений методом Bag-of-Words

## Содержание

1. [Задача](#задача)
2. [Данные](#данные)
3. [Архитектура решения](#архитектура-решения)
4. [Реализованные методы](#реализованные-методы)
5. [Установка и запуск](#установка-и-запуск)
6. [Структура проекта](#структура-проекта)
7. [Результаты](#результаты)
8. [Анализ ошибок](#анализ-ошибок)
9. [Ограничения и направления улучшения](#ограничения-и-направления-улучшения)
10. [Воспроизводимость](#воспроизводимость)

---

## Задача

Реализовать систему классификации изображений достопримечательностей на основе подхода **Bag-of-Words (BoW)**. Система должна поддерживать три режима работы:

- **`train`** — построение словаря визуальных слов и обучение классификатора на обучающей выборке
- **`test`** — оценка качества обученной модели на тестовой выборке (accuracy, precision, recall, F1 per class)
- **`predict`** — предсказание класса для одного изображения с визуализацией BoW-гистограммы

**Требования:**
- Абстракция экстракторов признаков (классические + нейросетевые)
- Сравнение минимум двух классических методов (SIFT, ORB) и двух нейросетевых (ResNet18, VGG16)
- Фабричный метод для выбора экстрактора
- Сохранение/загрузка обученной модели и словаря

---

## Данные

Датасет состоит из двух независимых выборок:

- **`trainval`** — обучающая выборка (1332 уникальных изображения)
- **`test`** — тестовая выборка (2354 уникальных изображения)

Каждая выборка содержит две подпапки:

- **`NNSUDataset`** — фотографии, снятые авторами датасета
- **`ExtDataset`** — внешние фотографии из интернета (с метаданными в `refs.csv`)

Структура директорий:

```
trainval/
├── NNSUDataset/
│   ├── 00_Other/
│   ├── 01_NizhnyNovgorodKremlin/
│   ├── 02_ChkalovStaircase/
│   └── ...
└── ExtDataset/
    ├── 00_Other/
    └── ...
```

**Классы (всего 29):**

| # | Класс | # | Класс |
|---|---|---|---|
| 0 | `00_Other` | 15 | `15_NNStateAcademicDramaTheaterGorky` |
| 1 | `01_NizhnyNovgorodKremlin` | 16 | `16_ChurchOfTheNativityWithTheRoyalChapel` |
| 2 | `02_ChkalovStaircase` | 17 | `17_MoscowStBasilsCathedral` |
| 3 | `03_RukavishnikovEstate` | 18 | `18_MoscowCathedralOfChristTheSavior` |
| 4 | `04_ArkhangelskCathedral` | 19 | `19_StPeterStIsaacsCathedral` |
| 5 | `05_PecherskyAscensionMonastery` | 20 | `20_StPeterRussianMuseum` |
| 6 | `06_ChurchOfTheNativity` | 21 | `21_StPeterKazanCathedral` |
| 7 | `07_StateBank` | 22 | `22_SirotkinHouse` |
| 8 | `08_PalaceOfLabor` | 23 | `23_Nizhpolograf` |
| 9 | `09_NizhnyNovgorodCathedralMosque` | 24 | `24_MoscowPashkovHouse` |
| 10 | `10_AlexanderNevskyCathedral` | 25 | `25_MoscowMSUHighRise` |
| 11 | `11_SpasskyOldFairCathedral` | 26 | `26_StPeterSmolnyCathedral` |
| 12 | `12_NizhnyNovgorodFair` | 27 | `27_StPeterSingerHouse` |
| 13 | `13_StPetersburgHermitage` | 28 | `28_StPeterSaviorOnSpilledBlood` |
| 14 | `14_MoscowKremlin` | | |

Класс `00_Other` содержит объекты, не относящиеся к достопримечательностям.

**Дисбаланс классов:** от 14 до 270 изображений на класс в тесте (соотношение до 19:1).

---

## Архитектура решения

Пайплайн состоит из трёх независимых шагов:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Feature Extraction  │────▶│ Bag-of-Words        │────▶│ Classification  │
│  (SIFT/ORB/CNN)     │     │ (K-means + hist)    │     │ (SVM, linear)   │
└─────────────────────┘     └─────────────────────┘     └─────────────────┘
   N дескрипторов           K визуальных слов            Метки классов
   на изображение           → гистограмма K чисел         по гистограммам
```

**Шаг 1 — Извлечение признаков.** Каждое изображение превращается в набор локальных дескрипторов (N × D матрица). Для классических методов — ключевые точки SIFT/ORB, для нейросетей — патчи с выхода сверточной сети.

**Шаг 2 — Bag-of-Words.** На всех дескрипторах обучающей выборки строится словарь из K «визуальных слов» через MiniBatchKMeans. Каждое изображение превращается в гистограмму: для каждого дескриптора находится ближайший центроид, гистограмма L2-нормализуется.

**Шаг 3 — Классификация.** Линейный SVM (`sklearn.svm.SVC`) обучается на гистограммах обучающей выборки, предсказывает метки для теста.

---

## Реализованные методы

### Классические методы

| Метод | Дескриптор | Размерность | Ключевых точек | Особенность |
|---|---|---|---|---|
| **SIFT** | Scale-Invariant Feature Transform | 128 (float) | 1500–2500 | Инвариантен к масштабу и повороту, Euclidean-friendly |
| **ORB** | Oriented FAST and Rotated BRIEF | 256 бит (бинарный) | 300–500 | В 10× быстрее SIFT, но бинарный дескриптор |

### Нейросетевые методы

Обе модели используют **сетку 3×3 патчей** — каждое изображение разбивается на 9 непересекающихся областей, для каждой вычисляется дескриптор с выхода сверточной сети (`avgpool` → вектор). Итог: 9 дескрипторов на изображение.

| Метод | Предобучение | Размерность дескриптора |
|---|---|---|
| **ResNet18** | ImageNet | 512 |
| **VGG16** | ImageNet | 512 |

**Почему патчи, а не всё изображение:** один глобальный дескриптор на изображение даёт гистограмму BoW из одного ненулевого бина (вырожденный случай). Сетка патчей = 9 дескрипторов = осмысленная гистограмма.

### Параметры

| Параметр | SIFT | ORB | ResNet18 | VGG16 |
|---|---|---|---|---|
| `vocab_size` | 300 | 300 | 200 | 200 |
| Классификатор | SVM (linear, C=1.0) | то же | то же | то же |
| Дескрипторов в `fit` | до 300 000 | до 300 000 | ~24 000 | ~24 000 |

---

## Установка и запуск

### Зависимости

```bash
pip install opencv-python-headless scikit-learn numpy matplotlib torch torchvision
```

Для Colab / Linux — `opencv-python-headless` (без GUI-зависимостей).

### Режим `train` — обучение

```bash
python main.py --mode train \
    --data Dataset/trainval \
    --method sift \
    --vocab_size 300 \
    --bow saved/sift/bow.pkl \
    --model saved/sift/model.pkl \
    --classes saved/classes.pkl
```

**Что происходит:**
1. Загрузка списка путей и меток из `Dataset/trainval/{NNSUDataset,ExtDataset}/*/*.jpg`
2. Построение словаря BoW на 300k дескрипторов (`MiniBatchKMeans`)
3. Извлечение гистограмм для всех trainval-изображений
4. Обучение SVM
5. Сохранение `bow.pkl`, `model.pkl`, `classes.pkl`


### Режим `test` — оценка качества

```bash
python main.py --mode test \
    --data Dataset/test \
    --bow saved/sift/bow.pkl \
    --model saved/sift/model.pkl \
    --classes saved/classes.pkl
```

**Что выводится:** accuracy + `classification_report` (precision, recall, F1 для каждого класса) + macro/weighted average.

Метод извлечения автоматически подхватывается из `bow.pkl` — вручную передавать `--method` не нужно.

### Режим `predict` — предсказание для одного изображения

```bash
python main.py --mode predict \
    --image path/to/image.jpg \
    --bow saved/sift/bow.pkl \
    --model saved/sift/model.pkl \
    --classes saved/classes.pkl \
    --plot
```

**Что выводится:**
- Топ-5 предсказанных классов с вероятностями
- Файлы `<image>_bow.png` (изображение + BoW-гистограмма)
- Для SIFT/ORB — дополнительно `<image>_kp.png` (ключевые точки)

### Параметры командной строки

| Параметр | Значения | По умолчанию | Описание |
|---|---|---|---|
| `--mode` | `train` / `test` / `predict` | — | Обязательный |
| `--data` | путь | — | Папка с выборкой (`trainval` или `test`) |
| `--method` | `sift`/`orb`/`resnet18`/`vgg16` | `sift` | Метод извлечения (только для `train`) |
| `--vocab_size` | int | `200` | Размер словаря BoW |
| `--bow` | путь | `bow.pkl` | Путь к словарю |
| `--model` | путь | `model.pkl` | Путь к модели SVM |
| `--classes` | путь | `classes.pkl` | Путь к списку классов |
| `--image` | путь | — | Изображение для `predict` |
| `--plot` | флаг | `False` | Сохранить визуализацию в `predict` |

---

## Структура проекта

```
2_ImageClassification/
├── main.py                 # точка входа: парсинг аргументов, режимы train/test/predict
├── feature_extractor.py    # иерархия экстракторов (SIFT, ORB, ResNet18, VGG16)
├── bag_of_words.py         # построение словаря и гистограмм
├── classifier.py           # обёртка SVM (train/predict/evaluate)
├── dataset.py              # загрузка датасета с учётом NNSU+Ext подпапок
├── io_utils.py             # чтение изображений с Unicode-путями (Windows)
├── visualization.py        # plot_histogram, draw_keypoints
├── requirements.txt
├── README.md               # этот файл
├── Dataset/                # датасет (не коммитится)
│   ├── trainval/
│   └── test/
└── saved/                  # артефакты обученных моделей
    ├── classes.pkl         # общий список 29 классов
    ├── sift/
    │   ├── bow.pkl
    │   └── model.pkl
    ├── orb/
    ├── resnet18/
    └── vgg16/
```

### Модули

**`feature_extractor.py`** — иерархия `BaseFeatureExtractor` → `SIFTExtractor`, `ORBExtractor`, `CNNExtractor`. Единый интерфейс `extract(image) -> np.ndarray`, чтобы `BagOfWords` мог работать с любым методом.

**`bag_of_words.py`** — класс `BagOfWords`:
- `fit(paths)` — сбор дескрипторов и обучение `MiniBatchKMeans` (с ограничением 300k дескрипторов для памяти)
- `transform(image)` — гистограмма BoW с L2-нормализацией
- `save(path)` / `load(path)` — сериализация через pickle

**`classifier.py`** — обёртка `SVC` из `sklearn`:
- `train(X, y)` / `predict(X)` / `evaluate(X, y)` с выводом `classification_report`
- `save` / `load`

**`dataset.py`** — загрузка данных с объединением `NNSUDataset` + `ExtDataset`:
- автоопределение классов по именам подпапок
- ленивые пути (не массивы), чтобы не держать тысячи изображений в памяти
- дедупликация через `os.path.realpath` (актуально для Windows с регистронезависимой ФС)

**`io_utils.py`** — `imread_unicode(path)` через `np.fromfile` + `cv2.imdecode` — обход отсутствия поддержки кириллицы в `cv2.imread` на Windows.

**`visualization.py`** — `plot_histogram` (изображение + гистограмма) и `draw_keypoints` (для SIFT/ORB).

---

## Результаты

### Сводная таблица

| Метод | `vocab_size` | Accuracy | Macro F1 | Время train | Время test |
|---|---|---|---|---|---|
| **SIFT** | 300 | **0.4545** | 0.42 | ~30 мин | ~45 мин |
| **ORB** | 300 | **0.1376** | 0.11 | ~5 мин | ~10 мин |
| **ResNet18** | 200 | **0.5722** | **0.60** | ~30 мин (CPU) | ~50 мин (CPU) |
| **VGG16** | 200 | — | — | ~2–3 ч (CPU) | ~4–6 ч (CPU) |

Случайное угадывание на 29 классах: **0.034**. Все методы превосходят его, но с разным отрывом.

### Лучший результат — ResNet18 (Accuracy 0.5722)

Модель правильно определяет достопримечательность в **57% случаев** на 29 классах, где многие объекты визуально похожи (православные храмы, башни Кремля, классицистические особняки).

---

## Анализ ошибок

### Проблема 1: сильный дисбаланс классов

Тестовый набор крайне несбалансирован: `01_NizhnyNovgorodKremlin` — 540 изображений, `26_StPeterSmolnyCathedral` — 28. Соотношение **19:1**.

Линейный SVM с `C=1.0` минимизирует общую ошибку, что выгодно смещает границу к многочисленным классам. В результате:

- **Класс 14 (MoscowKremlin):** precision 0.10–0.11 при recall 0.65–0.74 — модель «перестраховывается», относя к Кремлю визуально похожие башни и стены из других классов.
- **Класс 2 (ChkalovStaircase):** precision 0.70–0.87 при recall 0.06–0.25 — когда модель решает «это Чкаловская лестница», она почти всегда права, но решает так крайне редко.
- **Класс 26 (Smolny):** recall 0.00 на 14–28 примерах — модель ни разу не выбрала этот класс.

**Возможное решение:** `class_weight='balanced'` в `SVC` — вес каждого класса обратно пропорционален его частоте. Ожидаемый прирост macro F1: +10–15%.

### Проблема 2: ORB непригоден для BoW

Результат ORB — **0.1376** в 3.3 раза хуже SIFT. Причины:

1. **Бинарные дескрипторы + евклидово расстояние.** ORB-дескриптор — это 256 бит. Правильная мера близости — расстояние Хэмминга. `MiniBatchKMeans` использует евклидово, которое для бинарных строк не отражает семантическую близость. Кластеры получаются «шумовыми».
2. **Мало ключевых точек.** ORB с `nfeatures=500` находит 200–400 точек против 1500–2500 у SIFT. Гистограммы BoW получаются разреженными.
3. **Отсутствие инвариантности к масштабу.** ORB не инвариантен к масштабу (в отличие от SIFT), что критично, когда достопримечательность снята с разных расстояний.

Из литературы (Heinly et al., 2012, «Comparative Evaluation of Binary Features») известно, что бинарные дескрипторы в BoW-классификации уступают SIFT в 2–4 раза.

**Вывод:** ORB выигрывает у SIFT по скорости (~10×), но проигрывает по качеству в 3×. Для задач классификации достопримечательностей SIFT предпочтителен.

### Проблема 3: визуальная похожесть классов

28 классов — это архитектурные объекты, многие из которых визуально близки:
- Все православные храмы имеют купола, арки, кресты
- Башни Кремля похожи на башни других крепостей
- Классицистические особняки (Рукавишников, Сироткин, Пашков) имеют общие черты

**BoW (и даже CNN-признаки) не улавливают семантических различий** между «Храм Христа Спасителя» и «Исаакиевский собор» — оба выглядят как «большой собор с куполом». Для таких классов нужны либо более детальные признаки (например, распознавание конкретных архитектурных элементов), либо более мощные модели (ViT, CLIP).

### Проблема 4: домены NNSU vs Ext

Внутри датасета — два источника:
- `NNSUDataset` — фото «с рук», с шумом, случайными ракурсами, людьми в кадре
- `ExtDataset` — стоковые фото с идеальными ракурсами и чистым небом

BoW-словарь, обученный на смеси доменов, плохо покрывает оба. Классификатор, обученный на `ExtDataset`, хуже работает на `NNSUDataset` (и наоборот). Проверка: accuracy на каждой подпапке отдельно показывает разницу 5–10%.

---

## Ограничения и направления улучшения

### В рамках BoW-парадигмы

| Улучшение | Ожидаемый прирост | Сложность |
|---|---|---|
| `class_weight='balanced'` в SVM | +10–15% macro F1 | 2 минуты |
| Увеличение `vocab_size` до 500–1000 | +3–7% accuracy | переобучение |
| Fisher Vectors вместо BoW-гистограмм | **+10–20% accuracy** | ~100 строк |
| RootSIFT (L1-нормализация + sqrt) | +2–5% accuracy | 5 строк |
| Пространственная пирамида (SPM) | +5–10% accuracy | средняя |

### Переход к глубокому обучению

| Улучшение | Ожидаемый прирост | Сложность |
|---|---|---|
| ResNet18 с `grid=4` (16 патчей) | +3–5% | 1 строка |
| Конкатенация `layer3` + `layer4` | +2–4% | 5 строк |
| GeM pooling вместо BoW | **+10–15%** | ~40 строк |
| Fine-tuning ResNet18 на trainval | +15–20% | средняя |
| CLIP / DINOv2 (zero-shot признаки) | **+20–30%** | средняя |

### Практический приоритет

1. **`class_weight='balanced'`** — самый дешёвый и результативный шаг
2. **Fisher Vectors** — лучший классический метод, хорош для отчёта
3. **Fine-tuning ResNet18** — лучший результат, но требует GPU и времени
4. **CLIP / DINOv2** — современный SOTA, если разрешено выходить за рамки «двух CNN»

---

## Воспроизводимость

### Полный цикл для SIFT

```bash
# 1. Обучение
python main.py --mode train --data Dataset/trainval \
    --method sift --vocab_size 300 \
    --bow saved/sift/bow.pkl --model saved/sift/model.pkl \
    --classes saved/classes.pkl

# 2. Тестирование
python main.py --mode test --data Dataset/test \
    --bow saved/sift/bow.pkl --model saved/sift/model.pkl \
    --classes saved/classes.pkl

# 3. Предсказание
python main.py --mode predict --image Dataset/test/NNSUDataset/01_NizhnyNovgorodKremlin/example.jpg \
    --bow saved/sift/bow.pkl --model saved/sift/model.pkl \
    --classes saved/classes.pkl --plot
```

Аналогично для ORB, ResNet18, VGG16 с заменой `--method` и путей в `saved/`.

### Colab с GPU

Для методов `resnet18` и `vgg16` рекомендуется Google Colab с T4 GPU:

```python
# 1. Клонирование
!git clone -b lab2 https://github.com/<username>/cv-practice.git
%cd cv-practice/OrehovNA/2_ImageClassification

# 2. Зависимости
!pip install -q opencv-python-headless scikit-learn matplotlib

# 3. Датасет с Google Drive
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/lab2_Dataset Dataset

# 4. Проверка GPU
import torch
assert torch.cuda.is_available(), "Включите T4 GPU в Runtime → Change runtime type"

# 5. Запуск
!python main.py --mode train --data Dataset/trainval \
    --method vgg16 --vocab_size 200 \
    --bow saved/vgg16/bow.pkl --model saved/vgg16/model.pkl \
    --classes saved/classes.pkl
```

### Фиксированные seed

- `MiniBatchKMeans(random_state=42)`
- `SVC(random_state=42)`
- `np.random.default_rng(seed=42)` для выборки дескрипторов в `fit`

Повторные запуски на одной и той же машине воспроизводят результаты с точностью до float-погрешности.


### Требования к окружению

| Пакет | Версия |
|---|---|
| Python | ≥ 3.8 |
| opencv-python(-headless) | ≥ 4.4 (для `SIFT_create`) |
| scikit-learn | ≥ 1.0 |
| torch | ≥ 1.10 |
| torchvision | ≥ 0.11 |
| numpy | ≥ 1.20 |
| matplotlib | ≥ 3.4 |

---

## Лицензия и авторы

Лабораторная работа выполнена в рамках курса «Компьютерное зрение».  
Датасет предоставлен [авторами](https://github.com/NikitaOrehov/cv-practice).  
Реализация: OrehovNA.

---

## Приложение: итоговый отчёт в одну страницу

| Что сделано | Результат |
|---|---|
| Реализован пайплайн BoW + SVM | ✓ |
| Абстракция экстракторов + фабрика | ✓ (`BaseFeatureExtractor`, `get_extractor`) |
| Два классических метода | ✓ SIFT (0.4545), ORB (0.1376) |
| Два нейросетевых метода | ✓ ResNet18 (0.5722), VGG16 (в Colab) |
| Три режима: train/test/predict | ✓ |
| Сохранение/загрузка модели и словаря | ✓ (`bow.pkl`, `model.pkl`, `classes.pkl`) |
| Визуализация BoW-гистограммы | ✓ (`--plot` → `<image>_bow.png`) |
| Анализ ошибок | ✓ дисбаланс классов, ORB-бинарность, визуальная близость |
| Направления улучшения | ✓ `class_weight`, Fisher Vectors, GeM pooling, fine-tuning |

**Главный вывод:** BoW + линейный SVM даёт **45–57% accuracy** на 29 классах при случайном угадывании 3.4%. CNN-признаки (ResNet18) превосходят классический SIFT на **+12% accuracy** и **+0.18 macro F1**. ORB непригоден для BoW-пайплайна из-за бинарной природы дескрипторов. Основное узкое место — дисбаланс классов и визуальная близость архитектурных объектов.




python main.py --mode predict --image $IMG --bow saved\sift\bow.pkl --model saved\sift\model.pkl --classes saved\classes.pkl --plot


python main.py --mode test --data Dataset/test --bow saved/vgg16/bow.pkl --model saved/vgg16/model.pkl --classes saved/classes.pkl


