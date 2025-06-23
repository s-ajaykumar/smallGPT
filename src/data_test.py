'''
Data preprocessing:

Implemeted variable length word and character block size

GPT
Architecture:

Single char attention
Multilayer word attention
'''


import vocab as vocab

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


class Config:
    batch_size = 4
    vocab_size = len(vocab.itos) 
    n_embd = 32
    n_hidden = 4*n_embd
    n_heads = 2
    n_layers = 2
    c_block_size = 6        # The longest word in the shakesphere dataset.
    w_block_size = 16
    dropout_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_token = vocab.stoi['<pad>']
    lr = 4e-3
    max_iters = 1001
    eval_iters = 200
    eval_interval = 100



def get_batch(mode):
    if mode == 'train':
        x = train_data
    else:
        x = val_data

    #x_c_block_size = torch.randint(1, config.c_block_size + 1, (1,)).item()
    
    '''if x_c_block_size >= y_c_block_size:
        ixs = torch.randint(0, len(x) - ((w_block_size+1)*x_c_block_size) + 1, (config.batch_size, ))
    else:
        ixs = torch.randint(0, len(x) - ((w_block_size+1)*y_c_block_size) + 1, (config.batch_size, ))'''
    ixs = torch.randint(0, len(x) - ((config.w_block_size+1)*config.c_block_size) + 1, (config.batch_size, ))
        
    xb = [x[ix : ix + (config.w_block_size*config.c_block_size)].view(config.w_block_size, config.c_block_size) for ix in ixs]
    
    '''if x_c_block_size == y_c_block_size:
        yb = [x[ix+x_c_block_size : (ix+x_c_block_size) + (w_block_size*y_c_block_size)].view(w_block_size, y_c_block_size) for ix in ixs]
        
    else:'''
    yb = []
    for ix in ixs:
        targets = []
        target_ix = ix + config.c_block_size
        for v in range(config.w_block_size):
            target = x[target_ix]
            targets.append(target)
            target_ix += config.c_block_size
        yb.append(torch.stack(targets, dim = 0))

    xb = torch.stack(xb, dim = 0)
    yb = torch.stack(yb, dim = 0)
    
    x_diff = config.c_block_size - config.c_block_size
    
    if config.c_block_size - config.c_block_size != 0:
        pad_toks = torch.full((config.batch_size, config.w_block_size, x_diff), config.pad_token, dtype = torch.long)
        xb = torch.cat((xb, pad_toks), dim = -1)
    
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
            logits, loss = model(xb, attention_mask = attention_mask, targets = yb)
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

## Encoding
encoded_data = encode(data)
encoded_data = torch.tensor(encoded_data, dtype=torch.long)

## Splitting into train and validation sets
split = int(len(encoded_data) * 0.9)
train_data = encoded_data[:split]
val_data = encoded_data[split:]
print("-"*70, "\nTOTAL SAMPLES:\n", "-"*70)
print(f"Xtr : {len(train_data)} samples\nXval : {len(val_data)} samples\n\n")



'''
# Batch Generation Check
xb, yb = get_batch('train')  

print("-"*70, "\nBATCH INFO:\n", "-"*70)
print(f"xb shape : {xb.shape}\tyb shape : {yb.shape}\n\n")

for i in range(xb.shape[0]):
    for j in range(xb.shape[1]):
        print(f"Word {j+1} : ", end = '')
        print("Inputs:\n", decode(xb[i, j].tolist()), end = '')
        print("\n\n")
        print("Targets:\n", decode([yb[i, j].item()]), end = '')
        print("\n\n")
'''




# Model Building
class CharAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.c_proj.res_flag = 1
        
    def forward(self, x, attention_mask):
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

    def forward(self, x, attention_mask = None):
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)  
        out = self.c_fc(x)
        out = self.gelu(out)
        out = self.c_proj(out)
        out = self.dropout(out)
        return out



class W_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = WordAttention()
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP()
        

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    
        x = x + self.mlp(self.ln_2(x))
        return x



class C_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CharAttention()
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP()

    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x), attention_mask)
        return x
        
        
        
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cte = nn.Embedding(config.vocab_size, config.n_embd)
        self.cpe = nn.Embedding(config.c_block_size, config.n_embd)
        self.cpe.res_flag = 1
        self.wpe = nn.Embedding(config.w_block_size, config.n_embd)
        self.wpe.res_flag = 1
        
        self.w_h = nn.ModuleList([W_Block() for _ in range(config.n_layers)])
        self.c_h = nn.ModuleList([C_Block() for _ in range(config.n_layers)])
        
        self.final_proj = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.final_ln = nn.LayerNorm(config.n_embd)
        
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
            std = 0.02
            if hasattr(module, 'res_flag'):
                std *= (2**-0.5)
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
        
        
    def forward(self, x, attention_mask, targets = None):
        B, W, c = x.shape               # B, W, c 
        c_pos_emb = self.cpe(torch.arange(c, dtype = torch.long, device = config.device))  
        w_pos_emb = self.wpe(torch.arange(W, dtype = torch.long, device = config.device)) 

        c_emb = self.cte(x)             # B, W, c, C
        x = c_emb + c_pos_emb           # B, W, c, C


        for block in self.c_h:
            x = block(x, attention_mask)            # B, W, c, C


        # Taking last char of each word which represents the word
        x = x[:, :, -1, :]                  # B, W, C


        for block in self.w_h:
            x = block(x)                                                                        # B, W, C
        
        
        logits = self.final_proj(self.final_ln(x))                                                              #    B, W, vocab_size
        loss = None


        if targets is not None:
            B, W, vocab_size = logits.shape
            logits = logits.view(B*W, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index = config.pad_token)
        return logits, loss


    def generate(self, x, attention_mask, max_new_chars):                       # x - shape: B, W, c 
        for i in range(max_new_chars):
            x_slided = x[:, -config.w_block_size:, :]
            attention_mask_slided = attention_mask[:, -config.w_block_size:]    # B, W, c
            logits, loss = self(x_slided, attention_mask_slided)
            logits = logits[:, -1, :]                                           # B, vocab_size
            
            probs = F.softmax(logits, dim = -1)
            next_ix = torch.multinomial(probs, num_samples = 1)                 # B, 1

            last_word_last_ix = attention_mask.sum(-1)-1
            last_word_last_ix = last_word_last_ix[0][-1]
            
            if last_word_last_ix == config.c_block_size - 1:
                pad_vec = torch.full((1, config.c_block_size-1), config.pad_token, dtype = torch.long, device = config.device)
                next_ix = torch.cat((next_ix, pad_vec), dim = -1) 
                next_ix_attention_mask = (next_ix != config.pad_token).int()
                  
                x = torch.cat((x, next_ix.unsqueeze(0)), dim = 1)
                attention_mask = torch.cat((attention_mask, next_ix_attention_mask.unsqueeze(0)), dim = 1)
                 
            else:
                x[:, -1, last_word_last_ix+1] = next_ix[0]
                attention_mask[:, -1, last_word_last_ix+1] = 1
                
        return x, attention_mask

        





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
    logits, loss = model(xb, attention_mask = attention_mask, targets = yb)
    
    ## Backward pass
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    
    # Update
    optimizer.step()

t2 = time.time()
print("Time taken:\t", (t2-t1), "s\t", (t2-t1)/60, "m", "\n\n")





# Inference
for i in range(2):
    x = torch.tensor([78, 99, 99, 99, 99, 99], dtype = torch.long, device = config.device).unsqueeze(0).unsqueeze(0)
    attention_mask = (x != config.pad_token).int()
    out, out_attention_mask = model.generate(x, attention_mask, 100)  # B, W, c
    
    ## Decode
    print(f"SAMPLE {i}: ")
    for m, sample in enumerate(out):
        for n, word in enumerate(sample):
            end_ix = out_attention_mask[m, n].sum(-1)-1
            print(decode(word[:end_ix+1].tolist()), end = '')
    print("\n\n")

  