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
        
    