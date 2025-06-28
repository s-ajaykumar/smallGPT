import re 
import random
from dataclasses import dataclass
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)



@dataclass
class Config:
    batch_size = 4
    block_size = 20
    content_block_size = 4
    vocab_size = None
    stop_emb_size = None
    stop_vocab_size = None
    n_embd = 32
    n_layers = 2
    n_heads = 2
    max_iters = 1000
    eval_iters = 200
    eval_interval = 100
    lr = 1e-3
    dropout_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_mask_token = 3377
    y_mask_token = 0
    x_pad_token = 3549
    y_pad_token = -1
config = Config()



def pad(x, y):
    x_padded = []
    y_padded = []
    max_length = 0

    for sample in x:
        if len(sample) > max_length:
            max_length = len(sample)

    for x_sample, y_sample in zip(x, y):
        diff = max_length - len(x_sample)
        x_pad = torch.full((diff, ), config.x_pad_token, dtype = torch.long)
        y_pad = torch.full((diff, ), config.y_pad_token, dtype = torch.long)
        x_sample = torch.cat((x_sample, x_pad))
        y_sample = torch.cat((y_sample, y_pad))
        x_padded.append(x_sample)
        y_padded.append(y_sample)

    return torch.stack(x_padded), torch.stack(y_padded)



def get_batch(mode):
    xy_data = encoded_xy
    yy_data = encoded_yy
    
    if mode == 'train':
        data = train_data
        ixs = train_ixs
    else:
        data = val_data
        ixs = val_ixs
        
    xb = []
    yb = []
    batch_ixs = [random.randint(0, len(data)-config.content_block_size) for _ in range(config.batch_size)]   
       
    for ix in batch_ixs:    
        x = torch.cat((data[ix:ix+config.content_block_size], xy_data[ixs[ix]:ixs[ix+config.content_block_size-1]]))
        y = torch.cat((torch.full((config.content_block_size-1, ), config.y_pad_token, dtype = torch.long), yy_data[ixs[ix]:ixs[ix+config.content_block_size-1]+1]))
        xb.append(x)
        yb.append(y)
    xb, yb = pad(xb, yb)
    xb, yb = xb.to(config.device), yb.to(config.device)
    return xb, yb

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", 'val']:
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out



# Model building
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.c_proj.res_flag = 1
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(config.n_embd, dim = -1)
        q = q.view(B, T, config.n_heads, C//config.n_heads).transpose(1, 2)
        k = k.view(B, T, config.n_heads, C//config.n_heads).transpose(1, 2)
        v = v.view(B, T, config.n_heads, C//config.n_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = False)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias = False)
        self.c_proj.res_flag = 1

    def forward(self, x):
        out = self.c_fc(x)
        out = self.gelu(out)
        out = self.c_proj(out)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalSelfAttention()
        self.mlp = MLP()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.stop_emb_size, config.n_embd),
            wpe = nn.Embedding(50, config.n_embd),
            h = nn.ModuleList([Block() for _ in range(config.n_layers)]),
            ln = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.stop_vocab_size, bias = False)

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'res_flag'):
                std *= (2 * config.n_layers ** -0.5)
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    def forward(self, x, targets = None):
        B, T = x.shape

        pos = torch.arange(0, T, dtype = torch.long, device = config.device)
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln(x)
        logits = self.lm_head(x)
        loss = None
        
        if targets is not None:
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index = -1)
        return logits, loss
    
    def generate(self, ix):
        mask_count = 0
        length = 4#ix.shape[1]
        while True:
            logits, loss = self(ix)
            logits = logits[:, -1, :]
            p_dis = F.softmax(logits, dim = -1)
            next_ix = torch.multinomial(p_dis, num_samples = 1)
            ix = torch.cat((ix, next_ix), dim = -1)
            if next_ix == 0:
                mask_count += 1
            if mask_count == 4:
                return ix[0][length:]
        

    

# Data preprocessing
def remove_suffixes(word):
    if word in stop_words:
        return (word, )
    
    elif word.endswith('ing') and len(word) > 4:
        root = word[:-3]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in 'bcdfghjklmnpqrstvwxyz':
            return (root[:-1], root[-1], word[-3:])
        return (root, word[-3:])
    
    elif word.endswith('ed') and len(word) > 3:
        root = word[:-2]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in 'bcdfghjklmnpqrstvwxyz':
            return (root[:-1], root[-1], word[-2:])
        return (root, word[-2:])
    
    elif word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
        return (word[:-1], word[-1])
    
    elif word.endswith("ly") and len(word) > 3:
        root = word[:-2]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in 'bcdfghjklmnpqrstvwxyz':
            return (root[:-1], root[-1], word[-2:])
        return (root, word[-2:])
    
    return (word, )




r_expression = r'\w+|\s|[-,:?;!.\'\"]'

'''txt = "what is your name? my name is ajay. cars I have:toyato,rolls royce, cullinan!\"diwakar\" mani\'s car\nlatha val;she is running. I walked into a river"
x = re.findall(r_expression, txt.lower())
print(x)
removed_suffixes = []
for word in x:
    w = remove_suffixes(word)
    for t in w:
        removed_suffixes.append(t)
print(removed_suffixes)'''

stop_words = [
    "a", "about", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "around", "as", "at", "be", "because", "been", "before", 
    "being", "below", "between", "both", "but", "by", "can", "could", "did", 
    "do", "does", "down", "each", "else", "ever", "every", "for", "from", 
    "further", "had", "have", "he", "her", "here", "him", "himself", "his", 
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "let", "may", 
    "me", "might", "more", "most", "must", "my", "never", "no", "not", "now", 
    "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", 
    "own", "re", "s", "same", "she", "should", "so", "some", "still", "such", 
    "t", "than", "that", "the", "their", "them", "themselves", "then", "there", 
    "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "us", "very", "was", "we", "well", "were", "what", "when", "where", 
    "while", "who", "why", "will", "with", "would", "you", "your", "yours", "ed", "ing", "ly",
    "g", '!', 'a', '?', ',', '.', ':', ';', '-', '(', ')', '"', "'", '“', '”',
    '\n', '\t', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


with open("data/tiny_stories.txt", "r", encoding = 'utf-8') as f:
        text = f.read()
    
words = re.findall(r_expression, text.lower())

x = []
for word in words:
    w = remove_suffixes(word)
    for t in w:
        x.append(t)
y = x.copy()

filtered_x = [w for w in x if w not in stop_words]
filtered_x_ixs = [i for i, w in enumerate(x) if w not in stop_words]
x = filtered_x
x_ixs = filtered_x_ixs

stop_set = [word for word in set(y) if word in stop_words]
'''print(len(x), len(y))
print(len(set(x)))
print(len(stop_set))
print(x[:50], y[:x_ixs[49]+1])'''


words_vocab = set(x)
stop_words_vocab = stop_words.copy()
config.vocab_size = len(words_vocab)


stoi = {word : i for i, word in enumerate(words_vocab)}
itos = {i : word for i, word in enumerate(words_vocab)}
stop_xstoi = {word : i+3378 for i, word in enumerate(stop_words_vocab)}
stop_ystoi = {word : i+1 for i, word in enumerate(stop_words_vocab)}        # +1 for the mask token
stop_itos = {i+1 : word for i, word in enumerate(stop_words_vocab)}
stop_itos[config.y_mask_token] = '<mask>'

config.stop_emb_size = max(list(stop_xstoi.values()))+2                     # +2 length of the stop_xstoi, x_pad token
config.stop_vocab_size = len(stop_itos)                                  

encode = lambda ls: [stoi[word] for word in ls]
stop_xencode = lambda ls: [stop_xstoi[word] if word in stop_xstoi else config.x_mask_token for word in ls]
stop_yencode = lambda ls: [stop_ystoi[word] if word in stop_ystoi else config.y_mask_token for word in ls]
decode = lambda ls: ''.join([itos[ix] for ix in ls])
stop_decode = lambda ls, context: ''.join([stop_itos[ix] if ix != 0 else context.pop(0) for ix in ls])

    
encoded_x = encode(x)
encoded_xy = stop_xencode(y) 
encoded_yy =  stop_yencode(y)       

encoded_x = torch.tensor(encoded_x, dtype = torch.long)
encoded_xy = torch.tensor(encoded_xy, dtype = torch.long)
encoded_yy = torch.tensor(encoded_yy, dtype = torch.long)


n1 = int(0.9*len(encoded_x))
train_data = encoded_x[:n1]
val_data = encoded_x[n1:]

train_ixs = x_ixs[:n1]
val_ixs = x_ixs[n1:]

#print("Train data size: ", len(train_data), "Validation data size: ", len(val_data))
#print(decode(val_data[155:162].tolist()), encoded_xy[val_ixs[155] : val_ixs[161]+1], encoded_yy[val_ixs[155] : val_ixs[161]+1], stop_decode(encoded_yy[val_ixs[155] : val_ixs[161]+1].tolist()))

# Model initialization
print("-"*80)
print("Model initializing...")
print("-"*80)    
model = GPT()
model= model.to(config.device)
print(f"Total parameters : {sum([p.nelement()for p in model.parameters()])} parameters\t{sum([p.nelement()for p in model.parameters()]) / 1e6:.2f}M parameters\n")




# Model training
print("-"*80)
print("Training started...")
print("-"*80)
optimizer = torch.optim.AdamW(model.parameters(), config.lr)
for i in range(config.max_iters):
    if i%config.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}   : train_loss : {losses["train"]:.2f}    val_loss : {losses["val"]:.2f}")
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print()




#Inference
for j in range(3):
    i = torch.randint(0, len(val_data)-4, (1, ), dtype = torch.long, device = config.device)
    context = val_data[i:i+4].unsqueeze(0)
    words = [decode([k]) for k in context[0].tolist()]
    out = model.generate(context).tolist()
    out = stop_decode(out, words)
    print(f"input:\n{decode(context[0].tolist())}\n\noutput:\n{out}\n")

