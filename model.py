import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 4
context_len = 16 # 每个batch的token块长度，即每个句子16个单词
d_model = 64 # token embeddings vector 的长度
num_layers = 8
num_heads = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 50
eval_interval = 200
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Load training data
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Using TikToken (Same as GPT3) to tokenize the source text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

# Split train and validation
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


# Define Feed Forward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.ffn(x)
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(in_features=d_model, out_features=d_model // num_heads, bias=False)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model // num_heads, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model // num_heads, bias=False)

        # apply mask
        self.register_buffer('mask', torch.tril(torch.ones(context_len, context_len))) # 临时存储


    def forward(self, x):
        B, T, C = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)


        attention = Q @ K.transpose(-2, -1) / math.sqrt(d_model // num_heads)
        attention = attention.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        
        attention = attention @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeedForwardNetwork()
    
    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))

        return x
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_len
        self.num_heads = num_heads
        self.num_blocks = num_layers
        self.dropout = dropout
        self.max_token_value = max_token_value
        self.token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)
        self.transformer_blocks = nn.Sequential(*([TransformerBlock() for _ in range(num_layers)]))
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        """
        # Set up position embedding look-up table
        # following the same approach as the original Transformer paper (Sine and Cosine functions)
        """
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # The "logits" are the output values of our model before applying softmax
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

model = Model().to(device=device)

def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_len, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_len] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_len + 1] for idx in idxs]).to(device)
    return x, y

# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')

# Generate
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')