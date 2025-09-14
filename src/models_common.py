import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from rouge_score import rouge_scorer        
    
# Функция для вычисления ROUGE (ОБЩАЯ ДЛЯ ОБЕИХ МОДЕЛЕЙ)
def calculate_rouge_batch(model, tokenizer, data_loader, device, batch_size, model_type='lstm', num_samples=10):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_masks) in enumerate(data_loader):
            if batch_idx >= num_samples // batch_size + 1:
                break
                
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            
            for i in range(len(batch_inputs)):
                input_seq = batch_inputs[i]
                attn_mask = batch_masks[i]

                # НАХОДИМ РЕАЛЬНЫЕ ТОКЕНЫ (игнорируя padding)
                real_tokens_mask = attn_mask.bool()  # True где реальные токены
                real_indices = torch.where(real_tokens_mask)[0]  # индексы реальных токенов
                
                if len(real_indices) < 10:  # Слишком короткая последовательность
                    continue
                
                # Берем только реальные токены (игнорируя padding)
                real_tokens = input_seq[real_indices]

                # Разделяем на промпт и эталон (75%/25%)
                split_point = int(len(real_tokens) * 0.75)
                prompt_ids = real_tokens[:split_point]
                reference_ids = real_tokens[split_point:]

                 # Декодируем промпт и эталон
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
                
                if not reference_text.strip() or not prompt_text.strip():
                    continue

                # Генерация в зависимости от типа модели
                if model_type == 'lstm':
                    # Для LSTM используем только реальные токены промпта
                    generated = model.generate(
                        prompt_ids.unsqueeze(0), 
                        max_new_tokens=len(reference_ids)+10, 
                        temperature=0.8
                    )
                    # Извлекаем сгенерированную часть
                    generated_tokens = generated[0, prompt_ids.shape[0]:]
                    
                else:  # transformer
                    # Для Transformer нужно воссоздать правильный формат с attention_mask
                    # Создаем корректные входные данные для Transformer
                    prompt_with_padding = prompt_ids.unsqueeze(0)
                    attention_mask_prompt = torch.ones_like(prompt_with_padding)
                    
                    generated = model.generate(
                        prompt_with_padding.to(device),
                        attention_mask=attention_mask_prompt.to(device),
                        max_new_tokens=len(reference_ids)+10,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    # Извлекаем сгенерированную часть
                    generated_tokens = generated[0, prompt_ids.shape[0]:]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # Вычисление ROUGE только если есть что сравнивать
                if generated_text.strip() and reference_text.strip():
                    scores = scorer.score(reference_text, generated_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                    
                    # Вывод для отладки (первые несколько примеров)
                    if len(rouge1_scores) <= 3:
                        print(f"\n--- Пример {len(rouge1_scores)} ---")
                        print(f"Промпт: '{prompt_text}'")
                        print(f"Эталон: '{reference_text}'")
                        print(f"Сгенерировано: '{generated_text}'")
                        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")

    # Возвращаем результаты
    result = {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0,
        'num_samples': len(rouge1_scores)  # Добавляем количество успешных примеров
    }
    
    print(f"\nОбработано {result['num_samples']} примеров")
    return result

def save_lstm_model_weights(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Веса модели сохранены в: {filepath}")


# Создаем специальную функцию для демонстрационных примеров
def show_generation_examples(model, tokenizer, data_loader, device, model_type='lstm', num_examples=3):
    """Показывает примеры генерации"""
    
    print(f"\nПримеры автодополнений {model_type.upper()}:")
    example_count = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_masks) in enumerate(data_loader):
            if example_count >= num_examples:
                break
                
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            
            for i in range(len(batch_inputs)):
                if example_count >= num_examples:
                    break
                    
                input_seq = batch_inputs[i]
                attn_mask = batch_masks[i]
                
                # Находим реальные токены (игнорируя padding)
                real_indices = torch.where(attn_mask == 1)[0]
                
                if len(real_indices) < 10:
                    continue
                    
                real_tokens = input_seq[real_indices]
                split_point = int(len(real_tokens) * 0.75)
                prompt_ids = real_tokens[:split_point]
                reference_ids = real_tokens[split_point:]
                
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
                
                if not reference_text.strip():
                    continue
                
                # Генерация
                if model_type == 'lstm':
                    generated = model.generate(
                        prompt_ids.unsqueeze(0), 
                        max_new_tokens=20, 
                        temperature=0.7
                    )
                    generated_tokens = generated[0, prompt_ids.shape[0]:]
                else:
                    # Для Transformer
                    generated = model.generate(
                        prompt_ids.unsqueeze(0).to(device),
                        attention_mask=torch.ones_like(prompt_ids).unsqueeze(0).to(device),
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated_tokens = generated[0, prompt_ids.shape[0]:]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                print(f"\n{model_type.upper()} Пример {example_count + 1}:")
                print(f"Промпт: '{prompt_text}'")
                print(f"Эталон: '{reference_text}'")
                print(f"Сгенерировано: '{generated_text}'")
                
                example_count += 1

