import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from rouge_score import rouge_scorer

# Модель LSTM
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden
    
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0):
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            hidden = None
            
            for _ in range(max_new_tokens):
                output, hidden = self.forward(generated, hidden)
                next_token_logits = output[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token], dim=1)
                
            return generated
        
    
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
                
                real_length = attn_mask.sum().item()
                if real_length < 10:
                    continue
                    
                prompt_length = int(real_length * 0.75)
                prompt_ids = input_seq[:prompt_length]
                reference_ids = input_seq[prompt_length:real_length]
                
                reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
                if not reference_text.strip():
                    continue
                
                # Генерация в зависимости от типа модели
                if model_type == 'lstm':
                    generated = model.generate(prompt_ids.unsqueeze(0), max_new_tokens=len(reference_ids)+5, temperature=0.8)
                    generated_tokens = generated[0, prompt_ids.shape[0]:]
                else:  # transformer
                    generated = model.generate(
                        prompt_ids.unsqueeze(0).to(device),
                        attention_mask=torch.ones_like(prompt_ids).unsqueeze(0).to(device),
                        max_new_tokens=len(reference_ids)+5,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated_tokens = generated[0, prompt_ids.shape[0]:]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Вычисление ROUGE
                scores = scorer.score(reference_text, generated_text)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0
    }        

def save_lstm_model_weights(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Веса модели сохранены в: {filepath}")

# Обучение модели
# def train_lstm(lstm_model, tokenizer, train_loaderloader,  learning_rate, num_of_epochs, device):
#     # Обучение модели
#     train_losses = []
#     val_rouge_scores = []
#     criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#     optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

#     print("Начинаем обучение LSTM модели...")
#     for epoch in range(num_of_epochs):
#         lstm_model .train()
#         total_loss = 0
#         progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{num_of_epochs}')
        
#         for batch_inputs, batch_masks in progress_bar:
#             batch_inputs = batch_inputs.to(device)
#             batch_masks = batch_masks.to(device)
            
#             # Подготовка данных: X = все кроме последнего токена, y = все кроме первого
#             X = batch_inputs[:, :-1]
#             y = batch_inputs[:, 1:]
            
#             optimizer.zero_grad()
#             output, _ = lstm_model (X)
            
#             # Reshape для loss function
#             loss = criterion(output.reshape(-1, VOCAB_SIZE), y.reshape(-1))
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
#         avg_loss = total_loss / len(loader)
#         train_losses.append(avg_loss)
        
#         # Валидация и вычисление ROUGE
#         rouge_scores = calculate_rouge_batch(lstm_model , tokenizer, val_loader, device, BATCH_SIZE, 
#                                             'lstm', num_samples=50)
#         val_rouge_scores.append(rouge_scores)
        
#         print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, '
#             f'ROUGE-1 = {rouge_scores["rouge1"]:.4f}, '
#             f'ROUGE-2 = {rouge_scores["rouge2"]:.4f}, '
#             f'ROUGE-L = {rouge_scores["rougeL"]:.4f}')
        

# # Функция для вычисления ROUGE для LSTM
# def calculate_rouge_batch(model, tokenizer, data_loader, device, batch_size, num_samples=10):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (batch_inputs, batch_masks) in enumerate(data_loader):
#             if batch_idx >= num_samples // batch_size + 1:
#                 break
                
#             batch_inputs = batch_inputs.to(device)
#             batch_masks = batch_masks.to(device)
            
#             for i in range(len(batch_inputs)):
#                 input_seq = batch_inputs[i]
#                 attn_mask = batch_masks[i]
                
#                 real_length = attn_mask.sum().item()
#                 if real_length < 10:
#                     continue
                    
#                 prompt_length = int(real_length * 0.75)
#                 prompt_ids = input_seq[:prompt_length]
#                 reference_ids = input_seq[prompt_length:real_length]
                
#                 reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
#                 if not reference_text.strip():
#                     continue
                
#                 # Генерация
#                 generated = model.generate(prompt_ids.unsqueeze(0), max_new_tokens=len(reference_ids)+5, temperature=0.8)
#                 generated_tokens = generated[0, prompt_ids.shape[0]:]
#                 generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
#                 # Вычисление ROUGE
#                 scores = scorer.score(reference_text, generated_text)
#                 rouge1_scores.append(scores['rouge1'].fmeasure)
#                 rouge2_scores.append(scores['rouge2'].fmeasure)
#                 rougeL_scores.append(scores['rougeL'].fmeasure)
    
#     return {
#         'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
#         'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0,
#         'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0
#     }