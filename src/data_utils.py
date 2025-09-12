
import re
import datetime
import json

# функция для "чистки" текстов
def clean_string(text):
    
    text = text.lower()                             # приведение к нижнему регистру
    text = re.sub(r'http\S+|www\.\S+', '', text)    # удаление ссылок
    text = re.sub(r'@\w+', '', text)                # удаление упоминаний  
    text = re.sub(r'#\w+', '', text)                # удаление хэштегов 
    text = re.sub(r'[^a-z0-9\s]', '', text)         # удаление всего, кроме латинских букв, цифр и пробелов
    
    # Оставляем только буквы, основные знаки препинания и пробелы
    # Это упрощенное правило. Его нужно настроить под специфику данных.
    #text = re.sub(r"[^a-zA-Zа-яА-Я0-9.!?,;:'\- ]", " ", text)

    text = re.sub(r'\s+', ' ', text).strip()        # удаление дублирующихся пробелов, удаление пробелов по краям
    
    return text
    
# Функция для сохранения результатов в файл
def save_results_to_file(rouge_scores, output_file, MODEL_NAME, MAX_SEQUENCE_LEN,
                         model_type='lstm', experiment_name='', additional_info=None):
    # Создаем структуру данных для сохранения
    output_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_type': model_type,
        'model_name': MODEL_NAME,
        'max_sequence_len': MAX_SEQUENCE_LEN,
        'experiment_name': experiment_name,
        'parameters': {
            'top_p': 0.95,
            'temperature': 0.8,
            'max_new_tokens': 'reference_length + 5'
        },
        'rouge_scores': rouge_scores
        # 'dataset_info': {
        #     'total_samples': len(cleaned_tweets),
        #     'train_size': X_train.shape[0],
        #     'val_size': X_val.shape[0],
        #     'test_size': X_test.shape[0],
        #     'evaluated_samples': rouge_scores['num_samples']
        # }
    }

    # Добавляем дополнительную информацию, если есть
    if additional_info:
        output_data['additional_info'] = additional_info
    
    # Создаем директорию если не существует
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Читаем существующие результаты или создаем новый список
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []
    else:
        existing_data = []

    # Добавляем новую запись
    if isinstance(existing_data, list):
        existing_data.append(output_data)
    else:
        # Если файл был в старом формате, преобразуем в список
        existing_data = [existing_data, output_data]
    
    # Сохраняем обновленные данные
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты добавлены в файл: {output_file}")
    print(f"Всего экспериментов в файле: {len(existing_data)}")
    
    