import torch
import numpy
import os
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running with {device}")

dataset_folder = "./dataset/"
batch_size = 64
block_size = 95
vocab_size = 256
epochs = 100
lr = 3e-4
nembed = 384
n_head = 6
n_layer = 6
dropout = 0.2

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, path):
        self.files = [path + f"{f}" for f in os.listdir(path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = numpy.load(self.files[idx])
        src = item[:-1]
        trt = item[1:]
        return src, trt

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(nembed, head_size, bias=False)
        self.query = torch.nn.Linear(nembed, head_size, bias=False)
        self.value = torch.nn.Linear(nembed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])
        self.proj = torch.nn.Linear(head_size * num_heads, nembed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        o = torch.cat([h(x) for h in self.heads], dim=-1)
        o = self.proj(o)
        o = self.dropout(o)
        return o

class FeedFoward(torch.nn.Module):
    def __init__(self, nembed):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(nembed, 4 * nembed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * nembed, nembed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, nembed, n_head):
        super().__init__()
        head_size = nembed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(nembed)
        self.ln1 = torch.nn.LayerNorm(nembed)
        self.ln2 = torch.nn.LayerNorm(nembed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, nembed)
        self.position_embedding_table = torch.nn.Embedding(block_size, nembed)
        self.blocks = torch.nn.Sequential(*[Block(nembed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(nembed) 
        self.lm_head = torch.nn.Linear(nembed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = torch.nn.functional.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

if __name__ == "__main__":
    # preface
    dataset = DatasetLoader(dataset_folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # train
    for epoch in range(epochs):
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            src = batch[0].to(device)
            trt = batch[1].to(device)
            logits, loss = model(src, trt)
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                print(f"{idx}::{epoch}::{loss.item()}")
                wave = model.generate(src[0,:].unsqueeze(0), 1000)[0].tolist()
                plt.plot(wave)
                plt.savefig("latest.png")
                plt.close()
