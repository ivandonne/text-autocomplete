# Text Autocomplete Project

## Описание проекта

Проект по автоматическому дополнению текста с использованием двух различных архитектур нейронных сетей: LSTM и Transformer (DistilGPT-2). Цель проекта - сравнение эффективности этих подходов в задаче автодополнения текста на основе датасета твитов.

### Основные задачи:
- Предобработка и очистка текстовых данных
- Реализация LSTM модели для генерации текста
- Использование предобученной Transformer модели (DistilGPT-2)
- Сравнение метрик качества (ROUGE) между двумя подходами
- Визуализация примеров автодополнения

## Структура проекта
```
text-autocomplete/
│
├── data/
│ └── tweets.txt # Тексты твитов из sentiment140 dataset
│ └── tweets_small.txt # Первые 2100 твитов из tweets.txt для эксперимента на CPU
│ └── train.csv # выборка из tweets_small для обучения модели
│ └── val.csv   # выборка из tweets_small для валидации
│ └── test.csv  # выборка из tweets_small для проверки
│
├── src/
│ ├── data_utils.py # Функции для обработки и очистки данных
│ ├── lstm_model.py # Реализация LSTM модели
│ └── models_common.py # Общие функции для обеих моделей
│
├── results/ # Директория для сохранения результатов экспериментов
|
├── models/  # веса обученных моделей
|
├── solution.ipynb # Основной Jupyter notebook с кодом
├── requirements.txt # Зависимости проекта
└── README.md # Документация проекта
```

## Установка и запуск на Linux

### 1. Установка зависимостей

```
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Запуск Jupyter Notebook
```
# Активация виртуального окружения
source venv/bin/activate

# Запуск Jupyter notebook
jupyter notebook solution.ipynb

# Альтернативно: запуск Jupyter lab
jupyter lab solution.ipynb
```
### 3. Запуск через командную строку (если нужно)
```
# Конвертация notebook в Python скрипт и запуск
jupyter nbconvert --to python solution.ipynb
python solution.py
```
## Параметры конфигурации
```
# Параметры модели
MAX_SEQUENCE_LEN = 128
VOCAB_SIZE = 50257
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2

# Параметры обучения
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# Параметры генерации
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 20
```

## Запуск на различных окружениях
### Linux с GPU
```
# Проверка доступности GPU
nvidia-smi

# Установка PyTorch с поддержкой CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Linux без GPU
```
# Установка CPU-версии PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

# Результаты эксперимента
Из-за проблем с запуском ВМ с GPU эксперимент проводился на CPU ноутбука.
Поэтому данные были сильно уменьшены до 2100 твитов (tweets_small.txt)

## Ход обучение LSTM модели
Для обучения использовалось 10 эпох.
Модель быстро вышла на плато и достигла пика обучения на данном датасете.
Функция потерь имела высокое значение порядка 6.8, ROUGE-1 порядка 0.04.
## Примеры автодополнений
### LSTM модель
```
Промпт: 'i got the i can has chezburger book from the lobo and you are not here'
Эталон: ' to look at it wif me'
Сгенерировано: '. I was there with the i can but i couldn't do it so i was'
```
### DistilGPT-2
``` 
Промпт: 'i got the i can has chezburger book from the lobo and you are not here'
Эталон: ' to look at it wif me'
Сгенерировано: ' to eat. You are here to give a reason why you should eat this food. You are here'
```
Видно, что дополнения трансформера грамматически верные, но сильно отличаются от эталона.
Дополнения LSTM модели имеют плохую грамматику

## Сравнение метрик

```
============================================================
СРАВНЕНИЕ РЕЗУЛЬТАТОВ
============================================================
LSTM ROUGE-1:     0.0393
Transformer ROUGE-1: 0.0714

LSTM ROUGE-2:     0.0000
Transformer ROUGE-2: 0.0128

LSTM ROUGE-L:     0.0376
Transformer ROUGE-L: 0.0691
```

Валидационная метрика ROUGE у обеих моделей очень низкая.
Низкая метрика трансформера скорее всего связана со слабостью модели и 
специфичностью датасета (сленг, нарушение грамматики).
Стоит ожидать, что метрики LSTM модели будут улучшаться и превыся метрики трансформера, 
если провести обучение на полном датасете, поскольку будет учтена его специфика.