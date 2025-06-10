import vocab

import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class Config:
    batch_size = 4
    vocab_size = len(vocab.itos) 
    n_embd = 2
    n_hidden = 4*n_embd
    n_heads = 1
    n_layers = 1
    c_block_size = 24        # The longest word in the shakesphere dataset.
    w_block_size = 10
    dropout_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_token = vocab.stoi['<pad>']


def find_max_length(x, y):
    max_len = 0
    for sample in x:
        for word in sample:
            if len(word) > max_len:
                max_len = len(word)
                
    for sample in y:
        for word in sample:
            if len(word) > max_len:
                max_len = len(word)
    return max_len

def pad(x, max_len):
    padded_samples = []
    
    for sample in x:
        padded_sample = []
        
        for word in sample:
            diff = max_len - len(word)
            if diff == 0:
                padded_sample.append(word)
            else:
                pad_seq = [config.pad_token] * diff
                word = word + pad_seq
                padded_sample.append(word)
                
        padded_samples.append(padded_sample)
    return torch.tensor(padded_samples, dtype=torch.long)

def get_batch(mode):
    if mode == 'train':
        x, y = xtr, ytr
    else:
        x, y = xval, yval

    ix = [random.randint(0, len(x)-1) for _ in range(config.batch_size)]
    xb = [x[i] for i in ix]
    yb = [y[i] for i in ix]
    max_len= find_max_length(xb, yb)
    xb, yb = pad(xb, max_len), pad(yb, max_len)
    xb, yb = xb.to(config.device), yb.to(config.device)
    return xb, yb
  
@torch.no_grad()    
def estimate_loss():
    model.eval()
    splits = ['train', 'val']
    out = {}
    for split in splits:
        losses = []
        for i in range(config.eval_iters):
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out
   
encode = lambda s: [vocab.stoi[ch] for ch in s]
decode = lambda ix: ''.join([vocab.itos[i] for i in ix])
config = Config()



# Data Collection and Preprocessing
with open('data/shakesphere.txt', 'r') as f:
    data = f.read()
    
processed_data = re.findall(r'\S+,?\s+|\S+,|\S+\s+', data)

## Encoding
encoded_data = []
for word in processed_data:
    encoded_data.append(encode(word))
 
## Shaping samples of block_size
start_idx = 0
x_samples = []
y_samples=  []
x_encoded_data = encoded_data[:-1]
y_encoded_data = encoded_data[1:]

while start_idx+config.batch_size <= len(x_encoded_data):
    x_samples.append(x_encoded_data[start_idx : start_idx+config.batch_size])
    y_samples.append(y_encoded_data[start_idx : start_idx+config.batch_size])
    start_idx += 1

## Splitting into train and validation sets
split = int(len(x_samples) * 0.9)
xtr, ytr = x_samples[:split], y_samples[:split]
xval, yval = x_samples[split:], y_samples[split:]
print(f"Xtr : {len(xtr)} samples\tYtr : {len(ytr)} samples\nXval : {len(xval)} samples\tYval : {len(yval)} samples")

## Getting a batch
xb, yb = get_batch('train')
   

