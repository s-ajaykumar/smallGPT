import vocab as vocab


import torch
import torch.nn as nn
import torch.nn.functional as F



# Hyperparameters
max_iters = 1001
eval_iters = 200
eval_interval = 100
lr = 4e-3                  # 1e-3
batch_size =  8         #32
block_size = 16           #256
n_embd = 32          # 384
n_blocks = 2             # 6
n_heads = 2              # 6
dropout_ratio = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)


# EDA
with open('data/shakesphere.txt', "r", encoding = "utf-8") as f:
    text = f.read()
vocab = sorted(set(text))
vocab_size = len(vocab)
print(f"Vocabulary : {vocab}")
print()
print(f"Vocab size : {vocab_size}")
print()


# Dataset Preprocessing
stoi = {ch:i+1 for i, ch in enumerate(vocab[1:])}
itos = {i+1:ch for i, ch in enumerate(vocab[1:])}
stoi['\n'] = 0
itos[0] = '\n'
encode = lambda text: [stoi[ch] for ch in text]
decode = lambda ix: ''.join([itos[i] for i in ix])
text = torch.tensor(encode(text), dtype = torch.long)
n1 = int(0.9 * len(text))
train_data = text[:n1]
val_data = text[n1:]
def get_batch(mode):
    data = train_data if mode == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# Transformer Model Building
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(n_embd, 3*n_embd, bias = False)
        self.context_proj = nn.Linear(n_embd, (block_size)*n_embd, bias = False)    
        self.proj = nn.Linear(n_embd, n_embd, bias = False)
        self.proj_2 = nn.Linear(n_embd, n_embd, bias = False)
        self.dropout = nn.Dropout(dropout_ratio)
        

    def forward(self, x, prev_decomposed):
        B, W, C = x.shape
        q, k, v = self.attn(x).split(n_embd, dim = -1)

        q = q.view(B, W, n_heads, C//n_heads).transpose(1, 2)
        k = k.view(B, W, n_heads, C//n_heads).transpose(1, 2)
        v = v.view(B, W, n_heads, C//n_heads).transpose(1, 2)

        

        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        out = out.transpose(1, 2).contiguous().view(B, W, C)

        decomposed = out[:, -1, :]                                                  # B, C
        decomposed = self.context_proj(decomposed)                                  # B, block_size*n_embd
        decomposed = decomposed.view(B, block_size, n_embd)                       
        decomposed = decomposed[:, :W, :]
        
        if prev_decomposed is not None:
            decomposed = prev_decomposed + decomposed                                           # B, W, C
            decomposed = self.proj_2(decomposed)

        out = self.proj(out)
        out = self.dropout(out)
        return out, decomposed                                                    # B, W, C


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_1 = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd, bias = False),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd, bias = False),
        nn.Dropout(dropout_ratio))

        self.net_2 = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd, bias = False),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd, bias = False),
        nn.Dropout(dropout_ratio))

    def forward(self, x, decomposed):
        out = self.net_1(x)  # B, W, C
        decomposed = self.net_2(decomposed)
        return out, decomposed


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa_head = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ln_3 = nn.LayerNorm(n_embd)
        self.ln_4 = nn.LayerNorm(n_embd)

    def forward(self, x, decomposed):
        if decomposed is not None:
            decomposed = self.ln_3(decomposed)
        out, decomposed = self.sa_head(self.ln_1(x), decomposed)
        out = x + out
        out2, decomposed = self.ffwd(self.ln_2(out), self.ln_4(decomposed))
        out2 = out + out2
        return out2, decomposed


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])

        self.proj = nn.Linear(n_embd, block_size * n_embd, bias = False)

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)

    def forward(self, x, targets = None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb

        decomposed = None
        for block in self.blocks:
            x, decomposed = block(x, decomposed)

        x = self.ln_f(decomposed)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, ix, max_new_tokens):
        for _ in range(max_new_tokens):
            ix_cond = ix[:, -block_size:]
            logits, loss = self(ix_cond)
            logits = logits[:, -1, :]
            p_dis = F.softmax(logits, dim = -1)
            next_ix = torch.multinomial(p_dis, num_samples = 1)
            ix = torch.cat((ix, next_ix), dim = -1)
        return ix


# Model Initialization
model = Transformer()
m = model.to(device)
print(f"Total parameters : {sum([p.nelement()for p in m.parameters()])} parameters\t{sum([p.nelement()for p in m.parameters()]) / 1e6:.2f}M parameters\n")
print()
print(f"Model response(Before training)\n{decode(m.generate(torch.zeros((1, 1), dtype = torch.long, device = device), 500).tolist()[0])}")
print()


# Model training + evaluation
optimizer = torch.optim.AdamW(m.parameters(), lr)
for i in range(max_iters):
    if i%eval_interval == 0:
        losses = estimate_loss()
        print(f'step {i}   : train_loss : {losses["train"]:.2f}    val_loss : {losses["val"]:.2f}')
    x, y = get_batch("train")
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print()


#Inference
context = torch.zeros(1, 1, device = device, dtype = torch.long)
print(f"Model response(After training)\n{decode(m.generate(context, 500).tolist()[0])}")