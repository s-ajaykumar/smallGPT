import re 
import random
from dataclasses import dataclass
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)





def get_batch(mode):
    if mode == 'train':
        data = train_data
    else:
        data = val_data
        
    ixs = [random.randint(0, len(data)-1-config.block_size) for _ in range(config.batch_size)]
    xb = torch.stack([data[ix:ix+config.block_size] for ix in ixs])
    yb = torch.stack([data[ix+1:ix+config.block_size+1] for ix in ixs])
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
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block() for _ in range(config.n_layers)]),
            ln = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight

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
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, ix, max_new_tokens):
        for _ in range(max_new_tokens):
            ix_cond = ix[:, -config.block_size:]
            logits, loss = self(ix_cond)
            logits = logits[:, -1, :]
            p_dis = F.softmax(logits, dim = -1)
            next_ix = torch.multinomial(p_dis, num_samples = 1)
            ix = torch.cat((ix, next_ix), dim = -1)
        return ix
        

# Data collection
'''from datasets import load_dataset

train_ds = load_dataset("roneneldan/TinyStories", split = 'train')


text = ""
for i in range(1000):
    text += train_ds[i]['text'] + "\n"
    
with open("data/tiny_stories.txt", "w") as f:
    f.write(text)'''
    

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
stoi = {word : i for i, word in enumerate(words_vocab)}
itos = {i : word for i, word in enumerate(words_vocab)}
encode = lambda ls: [stoi[word] for word in ls]
decode = lambda ls: ' '.join([itos[ix] for ix in ls])

'''with open('data/words_vocab.txt', 'w') as f:
    f.write(str(stoi))'''
    
encoded_x = encode(x)
encoded_x = torch.tensor(encoded_x, dtype = torch.long)


n1 = int(0.9*len(encoded_x))
train_data = encoded_x[:n1]
val_data = encoded_x[n1:]
print("Train data size: ", len(train_data), "Validation data size: ", len(val_data))



@dataclass
class Config:
    batch_size = 8
    block_size = 8
    vocab_size = len(words_vocab)
    n_embd = 128
    n_layers = 4
    n_heads = 4
    max_iters = 10000
    eval_iters = 200
    eval_interval = 1000
    lr = 1e-2
    dropout_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model initialization
print("-"*80)
print("Model initializing...")
print("-"*80)  
config = Config()   
model = GPT()
model= model.to(config.device)
#model = torch.compile(model)
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
context = torch.full((1, 1), 63, device = config.device, dtype = torch.long)
print(f"Model response(After training)\n{decode(model.generate(context, 50).tolist()[0])}")
            
   
   
   
            
# Function to load model and generate a name
def load_model(model_path):
    global vocab_size, block_size, n_embd, n_heads, n_layers, dropout_ratio, pad_token, device, m, decode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract hyperparameters
    hyperparameters = checkpoint['hyperparameters']
    config.vocab_size = hyperparameters['vocab_size']
    config.block_size = hyperparameters['block_size']
    config.n_embd = hyperparameters['n_embd']
    config.n_heads = hyperparameters['n_heads']
    config.n_layers = hyperparameters['n_layers']
    config.dropout_ratio = hyperparameters['dropout_ratio']
    config.pad_token = hyperparameters['pad_token']

    # Load stoi and itos
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    decode = lambda ix: ''.join([itos[i] for i in ix])

    # Initialize and load model
    model = GPT()
    model.load_state_dict(checkpoint['model_state_dict'])
    m = model.to(device)
    m.eval()



def save_model(model_parameters, model_path):
    torch.save({'model_state_dict': model_parameters}, model_path)


save_model(model.state_dict(), 'model.pth')
