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

Все модели используют **сетку патчей** (grid×grid): изображение разбивается на непересекающиеся области, для каждой вычисляется дескриптор с выхода сети. Итог: `grid²` дескрипторов на изображение.

| Метод | Предобучение | Размерность | Слой по умолчанию |
|---|---|---|---|
| **ResNet18** | ImageNet (supervised) | 512 | `layer3` / `avgpool` |
| **ResNet50** | ImageNet (supervised) | 1024 | `layer3` / `avgpool` |
| **VGG16** | ImageNet (supervised) | 512 | `features` / `conv4_3` |
| **ConvNeXt-Base** | ImageNet (supervised) | 1024 | `avgpool` |
| **DINOv2 (ViT-B/14)** | **Self-supervised, 142M изображений** | **768** | `norm` (CLS-токен) |

**Почему DINOv2 — качественный скачок:** DINOv2 обучен **без разметки** на огромном корпусе. Его признаки кодируют **семантику** архитектурных объектов, а не текстуры ImageNet-классификаторов. Для 29 визуально близких классов (Кремли, храмы, соборы) это решающее преимущество.

**Параметры для DINOv2:** `grid=4`, `vocab_size=500`, `C=2.0`. Для остальных — `grid=3–4`, `vocab_size=200–300`, `C=1.0`.

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

### Сводная таблица (Colab T4, уникальный тест 2354)

| Метод | Конфигурация | Accuracy | Macro F1 |
|---|---|---|---|
| **DINOv2 (ViT-B/14)** | **grid=4, vocab=500, C=2.0** | **0.9167** | **0.93** |
| ConvNeXt-Base | grid=4, vocab=500, C=2.0 | 0.6415 | 0.66 |
| ResNet18 + avgpool (baseline) | grid=3, vocab=200, C=1.0 | 0.5994 | 0.62 |
| ResNet50 + avgpool | grid=3, vocab=300, C=1.0 | 0.5365 | 0.58 |
| ResNet18 + layer3 | grid=4, vocab=300, C=1.0 | 0.5072 | 0.54 |
| SIFT | vocab=300 | 0.4545 | 0.42 |
| ORB | vocab=300 | 0.1376 | 0.11 |

**Случайное угадывание:** 0.034 (1/29).

### Лучший результат — DINOv2 (0.9167)

DINOv2 правильно определяет достопримечательность в **92% случаев** на 29 классах. Это **+32% над baseline** (ResNet18) и **+28% над ConvNeXt-Base**.

**Классы с F1 = 1.00:** 17 (StBasils), 20 (Russian Museum), 21 (Kazan), 24 (Pashkov), 25 (MSU), 26 (Smolny), 27 (Singer), 28 (Savior on Blood).

**Проблемный класс:** 14 (Moscow Kremlin) — precision 0.51 при recall 0.96. Та же «перестраховка», что и у ResNet, но в 4 раза точнее. Причина — визуальная близость Кремлей (Нижегородский, Московский) и башен других крепостей.

### Эксперименты с гиперпараметрами (ResNet18 — все хуже baseline)

| Конфигурация | Accuracy | Вывод |
|---|---|---|
| avgpool, vocab=200, C=1.0 (baseline) | **0.5994** | оптимум |
| layer3, grid=4, vocab=300, C=1.0 | 0.5072 | layer3 хуже avgpool |
| layer3, grid=4, vocab=300, balanced, C=0.5 | 0.5017 | balanced вредит |
| layer3, grid=4, vocab=300, balanced, C=1.0 | 0.5034 | balanced вредит |
| avgpool, vocab=200, balanced, C=1.0 | 0.4630 | balanced вредит (−14%) |
| avgpool, vocab=200, **rbf**, C=1.0 | 0.4486 | RBF хуже linear на разреженных BoW |

**Вывод:** стандартный BoW + avgpool + линейный SVM — оптимум для ResNet18. Дальнейший прирост — только смена архитектуры (DINOv2).

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

### Проблема 4: почему DINOv2 обошёл все CNN на +32%

DINOv2 — визуальный трансформер, обученный **self-supervised** на 142M изображений. В отличие от ResNet/VGG/ConvNeXt (supervised на ImageNet-1000), он не «заточен» под 1000 классов ImageNet, а учится **общим визуальным представлениям**. Для landmark-классификации это даёт:

1. **Семантические признаки.** DINOv2 различает «Храм Христа Спасителя» и «Исаакиевский собор» по архитектурному смыслу, а ResNet — по локальным градиентам, которые у них похожи.
2. **Устойчивость к ракурсу.** Self-supervised обучение на разнообразных изображениях → лучше генерализация на `NNSUDataset` (фото с рук) и `ExtDataset` (стоковые).
3. **CLS-токен как глобальный дескриптор.** В ViT CLS-токен агрегирует информацию со всех патчей — аналог `avgpool`, но богаче.

**Цена:** DINOv2 тяжелее ResNet18 в ~10 раз (ViT-B/14: 86M параметров против 11M). Время train в Colab T4 — ~25–35 минут против ~5 минут у ResNet18. Но результат +32% оправдывает.

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

### Полный цикл для DINOv2 (лучший результат)

```bash
# 1. Обучение (Colab T4 GPU, ~25–35 минут)
python main.py --mode train --data Dataset/trainval \
    --method dinov2 --grid 4 --vocab_size 500 --C 2.0 \
    --bow saved/dinov2/bow.pkl \
    --model saved/dinov2/model.pkl \
    --classes saved/classes.pkl

# 2. Тестирование
python main.py --mode test --data Dataset/test \
    --bow saved/dinov2/bow.pkl \
    --model saved/dinov2/model.pkl \
    --classes saved/classes.pkl

# 3. Предсказание
python main.py --mode predict \
    --image Dataset/test/ExtDataset/01_NizhnyNovgorodKremlin/16-1.jpg \
    --bow saved/dinov2/bow.pkl \
    --model saved/dinov2/model.pkl \
    --classes saved/classes.pkl --plot

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


