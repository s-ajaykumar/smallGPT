'''
multilayer char attention - last char of each word is attended to all the chars previous to it. Its like each word attending to all the chars of prev words.
In char attention - unidirectional attention is used
tied weights - Character embedding and lm head are tied - same matrix is used.
After the final projection, out is added with character and word positional embeddings.
'''



import vocab as vocab

import re
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

@dataclass
class Config:
    #batch_size = 2
    vocab_size = len(vocab.itos) 
    n_embd = 32
    n_hidden = 4*n_embd
    n_heads = 2
    n_layers = 2
    c_block_size = 24        # The longest word in the shakesphere dataset.
    w_block_size = 16
    dropout_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_token = vocab.stoi['<pad>']
    lr = 4e-3
    max_iters = 4001
    eval_iters = 200
    eval_interval = 400



def pad(words):
    padded_samples = []
    for word in words:
        diff = config.c_block_size - len(word)
        if diff == 0:
            padded_samples.append(word)
        else:
            pad_seq = [config.pad_token] * diff
            word = word + pad_seq
            padded_samples.append(word)      
    return torch.tensor(padded_samples, dtype=torch.long)

def get_batch(mode):
    if mode == 'train':
        x, y = xtr, ytr
    else:
        x, y = xval, yval

    ix = torch.randint(0, len(x) - config.w_block_size, (1,))
    xb = []
    yb = []
    shapes = []
    x_words = x[ix : ix+config.w_block_size]
    y_words = y[ix : ix+config.w_block_size]
    for i, word in enumerate(x_words):
        if i == 0:
            shape = len(word)-1
        else:
            shape = len(word) + shapes[-1]
        shapes.append(shape)
    xb += [ch for word in x_words for ch in word]
     
         
    xb = torch.tensor(xb, dtype = torch.long)
    yb = pad(y_words)
    attention_mask = torch.tensor(shapes, dtype = torch.long)
    
    xb, yb, attention_mask = xb.to(config.device), yb.to(config.device), attention_mask.to(config.device)
    return xb, yb, attention_mask
    
@torch.no_grad()    
def estimate_loss():
    model.eval()
    splits = ['train', 'val']
    out = {}
    for split in splits:
        losses = []
        for i in range(config.eval_iters):
            xb, yb, attention_mask = get_batch(split)
            logits, loss = model(xb, targets = yb, attention_mask = attention_mask)
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




'''print("batch_size", config.batch_size)
print("w_block_size", config.w_block_size)
print("c_block_size", config.c_block_size, "\n\n")

xb, yb, attention_mask = get_batch('train')
print("X batch shape: ", xb.shape)
print("Y batch shape: ", yb.shape)
print("Attention mask shape: ", attention_mask.shape, "\n\n")   

print("xb:\n", decode(xb.tolist()))
print("yb:\n", decode(yb.tolist()))
print("Attention mask:\n", attention_mask.tolist(), "\n\n")'''





# Model Building
class CharAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.kv_attn = nn.Linear(config.n_embd, 2*config.n_embd, bias = False)
        
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.proj.res_flag = 1
        
    def forward(self, x, attention_mask):
        x_new = x.clone()
        words = x[attention_mask]
        q = self.q_attn(words)
        
        out = []
        for i in range(len(attention_mask)):
            kv = self.kv_attn(x[:attention_mask[i]+1])      
            k, v = kv.split(config.n_embd, dim = -1)
            att_sc = q[i] @ k.T                                            
            att_sc = F.softmax(att_sc, dim = -1)
            out.append(att_sc @ v)
            x_new[attention_mask[i]] = x_new[attention_mask[i]] + out[-1]                 # Residual connection
        out = torch.stack(out, dim = 0)

        out = self.proj(out)
        out = self.dropout(out)
        out = out + words                                   # Residual connection         
        return out, x_new


        
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_hidden, bias = False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_hidden, config.n_embd, bias = False)
        self.d_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.c_proj.res_flag = 1

    def forward(self, in_words, x, attention_mask):
        x_new = x.clone()
        words = self.c_fc(in_words)
        words = self.gelu(words)
        
        words = self.c_proj(words)
        words = self.dropout(words)
        x = self.dropout(x)
        
        words = words + in_words                                            # Residual connection  
        for i in range(len(attention_mask)):
            x_new[attention_mask[i]] = x_new[attention_mask[i]] + words[i]                                # Residual connection
         
        return words, x_new


class C_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CharAttention()
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP()
        self.ln_3 = nn.LayerNorm(config.n_embd)

    def forward(self, x, attention_mask):
        words, x = self.attn(self.ln_1(x), attention_mask)
        words, x = self.mlp(self.ln_2(words), self.ln_3(x), attention_mask)
        return words, x
        
        
        
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cte = nn.Embedding(config.vocab_size, config.n_embd)
        self.cpe = nn.Embedding(config.w_block_size*config.c_block_size, config.n_embd)
        self.cpe.res_flag = 1
        self.wpe = nn.Embedding(config.w_block_size, config.n_embd)
        self.wpe.res_flag = 1
        
        self.c_h = nn.ModuleList([C_Block() for _ in range(config.n_layers)])
        
        self.proj = nn.Linear(config.n_embd, config.c_block_size*config.n_embd)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        
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
        
        
    def forward(self, x, targets = None, attention_mask = None):
        c = len(x)
        c_emb = self.cte(x)                                                                 #W*c, C
        c_pos_emb = self.cpe(torch.arange(c, dtype = torch.long, device = config.device)) 
        x = c_emb + c_pos_emb       

        for block in self.c_h:
            words, x = block(x, attention_mask)            

        W, C = words.shape
        c_pos_emb = self.cpe(torch.arange(config.c_block_size, dtype = torch.long, device = config.device))
        w_pos_emb = self.wpe(torch.arange(W, dtype = torch.long, device = config.device))
        x = self.proj(self.ln_1(words)).view(-1, config.c_block_size, config.n_embd)               
        x = x + c_pos_emb.unsqueeze(0)
        x = x + w_pos_emb.unsqueeze(1)

        x = self.ln_2(x)                
        logits = x @ self.cte.weight.T 
        #logits = self.lm_head(self.ln_2(x))                                      
        loss = None

        if targets is not None:
            W, c_block_size, vocab_size = logits.shape
            logits = logits.view(W*c_block_size, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index = config.pad_token)
        return logits, loss


    def generate(self, x, attention_mask, max_new_words):  
        x_slided = x.clone()
        attention_mask_slided = attention_mask.clone()    
        for i in range(max_new_words):
            if len(attention_mask_slided) > config.w_block_size:
                x_slided = x_slided[attention_mask_slided[0]+1:]
                length_of_1st_word = attention_mask_slided[0] + 1
                attention_mask_slided = attention_mask_slided[1:] - length_of_1st_word
            logits, loss = self(x_slided, attention_mask = attention_mask_slided)
            logits = logits[-1, :, :]
            c_block_size, vocab_size = logits.shape
            
            probs = F.softmax(logits, dim = -1)
            ix = torch.multinomial(probs, num_samples = 1)

            for j, ch in enumerate(ix):
                end_ix = c_block_size - 1
                if ch in [77, 78]:
                    end_ix = j
                    break
            next_word = ix[:end_ix+1].view(-1)
            next_word_attention_mask = torch.tensor([len(next_word) + attention_mask_slided[-1]], dtype = torch.long, device = config.device)
            x = torch.cat((x, next_word), dim = 0)
            attention_mask = torch.cat((attention_mask, next_word_attention_mask), dim = 0)
            x_slided = torch.cat((x_slided, next_word), dim = 0)
            attention_mask_slided = torch.cat((attention_mask_slided, next_word_attention_mask), dim = 0)
            
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
    xb, yb, attention_mask = get_batch('train')
    logits, loss = model(xb, targets = yb, attention_mask = attention_mask)
    
    ## Backward pass
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    
    # Update
    optimizer.step()

t2 = time.time()
print("Time taken:\t", (t2-t1), "s\t", (t2-t1)/60, "m")





# Inference
for i in range(2):
    x = torch.tensor([45, 7, 4, 77], dtype = torch.long, device = config.device)
    attention_mask = torch.tensor([len(x) - 1], dtype = torch.long, device = config.device)  
    chars, attention_mask = model.generate(x, attention_mask, 15)  
    
    ## Decode
    print(f"SAMPLE {i+1}: ")
    start_ix = 0
    for k in range(len(attention_mask)):
        end_ix = attention_mask[k] + 1
        word = chars[start_ix : end_ix].tolist()
        print(decode(word), end = '')
        start_ix = end_ix
    print("\n\n")