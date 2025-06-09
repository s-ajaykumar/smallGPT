import data.vocab as vocab

import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F




# Data Collection and Preprocessing
with open('data/shakesphere.txt', 'r') as f:
    data = f.read()

processed_data = re.findall(r'\S+,?\s+|\S+,|\S+\s+', data)

encoded_data = []
for word in processed_data:
    encoded_data.append(list(word.encode('utf-8')))


def get_batch(mode):
    if mode == 'train':
        x, y = xtr, ytr
    else:
        x, y = xval, yval

    ix = [random.randint(0, len(x)) for _ in range(config.batch_size)]
    xb = pad([x[i] for i in ix])
    yb = pad([y[i] for i in ix])





class Config:
    vocab_size = len(vocab.itos) 
    n_embd = 2
    n_hidden = 4*n_embd
    n_heads = 1
    n_layers = 1
    c_block_size = 24        # The longest word in the shakesphere dataset.
    w_block_size = 10
    dropout_ratio = 0.2
config = Config()
# Attention cpu version

class CharAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)
        
    def forward(self, x):
        B, W, c, C = x.shape
        
        qkv = self.attn(x)
        q, k, v = self.qkv.split(config.n_embd, dim = -1)
        q = q.view(B, W, c, config.n_heads, C//config.n_heads).transpose(2, 3)
        k = k.view(B, W, c, config.n_heads, C//config.n_heads).transpose(2, 3)
        v = v.view(B, W, c, config.n_heads, C//config.n_heads).transpose(2, 3)
        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        out = out.transpose(2, 3).contiguous().view(B, W, c, C)
        
        out = self.c_proj(out)
        out = self.dropout(out)
        out = x + out             # Residual connection
        out = out[:, :, -1, :]    # B, W, C
        return out



class WordAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        B, W, C = x.shape
        
        qkv = self.attn(x)
        q, k, v = self.qkv.split(config.n_embd, dim = -1)
        q = q.view(B, W, config.n_heads, C//config.n_heads).transpose(1, 2)
        k = k.view(B, W, config.n_heads, C//config.n_heads).transpose(1, 2)
        v = v.view(B, W, config.n_heads, C//config.n_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        out = out.transpose(1, 2).contiguous().view(B, W, C)
        
        out = self.c_proj(out)
        out = self.dropout(out)
        return out


        
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_hidden, bias = False),
            nn.GELU(),
            nn.Linear(config.n_hidden, config.n_embd, bias = False),
            nn.Dropout(config.dropout_ratio),
        )

    def forward(self, x):
        x = self.net(x)
        return x



class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.w_attn = WordAttention()
        self.mlp = MLP()
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.w_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


        
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cte = nn.Embedding(config.vocab_size, config.n_embd)
        self.cpe = nn.Embedding(config.c_block_size, config.n_embd)
        self.wpe = nn.Embedding(config.w_block_size, config.n_embd)
        
        self.c_attn = CharAttention()
        self.h = nn.ModuleList([Block() for _ in range(config.n_layers)])
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_hidden, config.vocab_size) for _ in range(config.c_block_size)])

        
    def forward(self, x):
        c_emb = self.cte(x)
        c_pos_emb = self.cpe(x)
        x = c_emb + c_pos_emb
        x = self.c_attn(x)
        pos_emb = self.wpe(torch.arange(x.shape[1], dtype = torch.long, device = config.device))
        x = x + pos_emb
        for block in self.h:
            x = block(x)
        logits = []
        for lm_head in self.lm_heads:
            logits.append(lm_head(x))
        logits = torch.stack(logits, dim = 2)
            
        