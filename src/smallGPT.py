import vocab as vocab

import re
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F




class Config:
    batch_size = 4
    vocab_size = len(vocab.itos) 
    n_embd = 32
    n_hidden = 4*n_embd
    n_heads = 2
    n_layers = 2
    c_block_size = 24        # The longest word in the shakesphere dataset.
    w_block_size = 10
    dropout_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_token = vocab.stoi['<pad>']
    lr = 4e-3
    max_iters = 1001
    eval_iters = 200
    eval_interval = 100



def pad(x):
    padded_samples = []
    
    for sample in x:
        padded_sample = []
        
        for word in sample:
            diff = config.c_block_size - len(word)
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
    xb, yb = pad(xb), pad(yb)
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
print("-"*70, "\nTOTAL SAMPLES:\n", "-"*70)
print(f"Xtr : {len(xtr)} samples\tYtr : {len(ytr)} samples\nXval : {len(xval)} samples\tYval : {len(yval)} samples\n\n")




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
        q, k, v = qkv.split(config.n_embd, dim = -1)
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

        self.c_proj.res_flag = 1

    def forward(self, x):
        B, W, C = x.shape
        
        qkv = self.attn(x)
        q, k, v = qkv.split(config.n_embd, dim = -1)
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
        self.c_fc = nn.Linear(config.n_embd, config.n_hidden, bias = False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_hidden, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.c_proj.res_flag = 1

    def forward(self, x):
        out = self.c_fc(x)
        out = self.gelu(out)
        out = self.c_proj(out)
        out = self.dropout(out)
        return out



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
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size) for _ in range(config.c_block_size)])

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'res_flag'):
                std *= ((2*config.n_layers)**-0.5)
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    def forward(self, x, targets = None):
        B, W, c = x.shape               # B, W, c   
        c_emb = self.cte(x)             # B, W, c, C
        
        c_pos_emb = self.cpe(torch.arange(c, dtype = torch.long, device = config.device))   # Character pos encoding
        x = c_emb + c_pos_emb

        x = self.c_attn(x)          # Character attention   -> returns B, W, C

        pos_emb = self.wpe(torch.arange(W, dtype = torch.long, device = config.device))     # Word pos encoding
        x = x + pos_emb

        for block in self.h:
            x = block(x)

        logits = []                       
        for lm_head in self.lm_heads:
            logits.append(lm_head(x))
        logits = torch.stack(logits, dim = 2)   # shape : B, W, c_block_size, vocab_size
        loss = None

        if targets is not None:
            B, W, c_block_size, vocab_size = logits.shape
            logits = logits.view(B*W*c_block_size, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index = config.pad_token)
        return logits, loss

    def pad(self, x):
        padded_samples = []
    
        for word in x:
            diff = config.c_block_size - len(word)
            if diff == 0:
                padded_samples.append(word)
            else:
                pad_seq = torch.full((diff,), config.pad_token)
                word = torch.cat((word, pad_seq), dim = -1)
                padded_samples.append(word)
        return torch.stack(padded_samples, dim = 0).unsqueeze(1)

    def generate(self, x, max_new_words):      # x - shape: B, W, c 
        for i in range(max_new_words):
            logits, loss = model(x)
            logits = logits[:, -1, :, :]
            B, c_block_size, vocab_size = logits.shape
            
            logits = logits.view(B*c_block_size, -1)
            probs = F.softmax(logits, dim = -1)
            ix = torch.multinomial(probs, num_samples = 1)
            ix = ix.view(B, c_block_size)

            next_word_batch = []
            for b in ix:
                stop_ix = None
                for i, element in enumerate(b):
                    if element in [77, 78]:
                        stop_ix = i
                        break
                if stop_ix:
                    next_word = b[:stop_ix+1]
                else:
                    next_word = b[:c_block_size]
                next_word_batch.append(next_word)
            next_words = self.pad(next_word_batch)
            x = torch.stack((x, next_words), dim = 1)  # Concatenate along the word dimension
            print(x.shape, x)
            return
        return x

        


# Model Initialization  
model = GPT()
model = model.to(config.device)  
print("-"*70, "\nMODEL INFO:\n", "-"*70)
print(f"model parameters : \t{sum([p.nelement() for p in model.parameters()]) / 1e6 : .3f}M parameters\n\n")   


x = torch.cat((torch.tensor([45, 7, 4], dtype = torch.long, device = config.device), torch.full((21, ), config.pad_token)), dim = -1)
out = model.generate(x, 15)
for sample in out:
    for word in sample:
        print(decode(word.tolist()), end = '')
    print("\n\n")

# Model Training
'''optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)

t1 = time.time()
for iter in range(config.max_iters):
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter}:\ttrain_loss: {losses['train']}\tval_loss: {losses['val']}")
        
    ## Forward pass
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    ## Backward pass
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    
    # Update
    optimizer.step()

t2 = time.time()
print("Time taken:\t", (t2-t1), "s\t", (t2-t1)/1000, "m")'''
    