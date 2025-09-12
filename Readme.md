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
│
├── src/
│ ├── data_utils.py # Функции для обработки и очистки данных
│ └── lstm_model.py # Реализация LSTM модели
│
├── results/ # Директория для сохранения результатов
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