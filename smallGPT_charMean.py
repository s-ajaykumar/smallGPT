import vocab as vocab

import re
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


class Config:
    batch_size = 8
    vocab_size = len(vocab.itos) 
    n_embd = 128
    n_hidden = 4*n_embd
    n_heads = 4
    n_layers = 4
    c_block_size = 24        # The longest word in the shakesphere dataset.
    w_block_size = 16
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

    ix = [random.randint(0, len(x)-config.w_block_size) for _ in range(config.batch_size)]
    xb = [x[i : i+config.w_block_size] for i in ix]
    yb = [y[i : i+config.w_block_size] for i in ix]
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
            xb, yb= get_batch(split)
            attention_mask = (xb != config.pad_token).int()
            logits, loss = model(xb, attention_mask, yb)
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

## Splitting into train and validation sets
split = int(len(encoded_data) * 0.9)
train_data = encoded_data[:split]
val_data = encoded_data[split:]
xtr, ytr = train_data[:-1], train_data[1:]  
xval, yval = val_data[:-1], val_data[1:]
print("-"*70, "\nTOTAL SAMPLES:\n", "-"*70)
print(f"Xtr : {len(xtr)} samples\tYtr : {len(ytr)} samples\nXval : {len(xval)} samples\tYval : {len(yval)} samples\n\n")





# Attention cpu version
class CharMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)
        
    def forward(self, x, attention_mask):
        B, W, c, C = x.shape
        out = self.v(x)
        out = out * attention_mask.unsqueeze(-1)
        
        out = out.mean(dim = 2)  # B, W, C
        out = self.c_proj(out)
        out = self.dropout(out)
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
        
        self.c_mean = CharMean()
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
        
    def forward(self, x, attention_mask, targets = None):
        B, W, c = x.shape               # B, W, c   
        c_emb = self.cte(x)             # B, W, c, C
        
        c_pos_emb = self.cpe(torch.arange(c, dtype = torch.long, device = config.device))   # Character pos encoding
        x = c_emb + c_pos_emb

        x = self.c_mean(x, attention_mask)          # Character attention   -> returns B, W, C
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

    def generate(self, x, attention_mask, in_end_ix, max_new_words):      # x - shape: B, W, c 
        for i in range(max_new_words):
            x_slided = x[:, -config.w_block_size:, :]
            attention_mask_slided = attention_mask[:, -config.w_block_size:]  # B, W, c
            logits, loss = self(x_slided, attention_mask_slided)
            logits = logits[:, -1, :, :]
            B, c_block_size, vocab_size = logits.shape
            
            logits = logits.view(B*c_block_size, -1)
            probs = F.softmax(logits, dim = -1)
            ix = torch.multinomial(probs, num_samples = 1)
            ix = ix.view(B, c_block_size)

            attention_mask_slided = []
            out_end_ix = []
            for b in ix:
                end_ix = len(b) - 1
                for j, element in enumerate(b):
                    if element in [77, 78]:
                        end_ix = j
                        break
                out_end_ix.append(end_ix)
                if end_ix == config.c_block_size - 1:
                    attn_mask_vec = torch.ones((config.c_block_size,), dtype = torch.long, device = config.device)
                else:
                    ones = torch.ones((end_ix+1,), dtype = torch.long, device = config.device)
                    zeros = torch.zeros((config.c_block_size - (end_ix + 1),), dtype = torch.long, device = config.device)
                    attn_mask_vec = torch.cat((ones, zeros), dim = 0)
                attention_mask_slided.append(attn_mask_vec) 
                
            out_end_ix = torch.tensor(out_end_ix, dtype = torch.long, device = config.device).unsqueeze(1)  # B, W
            in_end_ix = torch.cat((in_end_ix, out_end_ix), dim = 1)  # B, W+1
            attention_mask_slided = torch.stack(attention_mask_slided, dim = 0).unsqueeze(1)  # B, W, c_block_size
             
            ix = ix.unsqueeze(1)                        # B, W, c_block_size 
            x = torch.cat((x, ix), dim = 1)             # Concatenate along the word dimension
            attention_mask = torch.cat((attention_mask, attention_mask_slided), dim = 1)
        return x, in_end_ix

        


# Model Initialization  
model = GPT()
model = model.to(config.device)  
print("-"*70, "\nMODEL INFO:\n", "-"*70)
print(f"model parameters : \t{sum([p.nelement() for p in model.parameters()]) / 1e6 : .3f}M parameters\n\n")   




# Model Training
optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)

t1 = time.time()
for iter in range(config.max_iters):
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter}:\ttrain_loss: {losses['train']}\tval_loss: {losses['val']}")
        
    ## Forward pass
    xb, yb = get_batch('train')
    attention_mask = (xb != config.pad_token).int()
    logits, loss = model(xb, attention_mask, yb)
    
    ## Backward pass
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    
    # Update
    optimizer.step()

t2 = time.time()
print("Time taken:\t", (t2-t1), "s\t", (t2-t1)/60, "m")





# Inference
for i in range(2):
    x = torch.cat((torch.tensor([45, 7, 4, 77], dtype = torch.long, device = config.device), torch.full((20, ), config.pad_token)), dim = -1).unsqueeze(0).unsqueeze(0)
    in_end_ix = torch.tensor([3], dtype = torch.long, device = config.device).unsqueeze(1)  # B, W
    attention_mask = (x != config.pad_token).int()
    out, out_end_ix = model.generate(x, attention_mask, in_end_ix, 15)  # B, W, c
    
    ## Decode
    print(f"SAMPLE {i}: ")
    for m, sample in enumerate(out):
        for n, word in enumerate(sample):
            end_ix = out_end_ix[m, n]
            print(decode(word[:end_ix+1].tolist()), end = '')
    print("\n\n")
    