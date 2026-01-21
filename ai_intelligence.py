import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import time

# =============================================================================
# THE ULTIMATE SELF-CONTAINED AI (ZERO EXTERNAL DOWNLOADS)
# This script implements a high-performance Transformer architecture,
# trains it on internal knowledge, and provides an interactive interface.
# =============================================================================

# --- Hyperparameters ---
BATCH_SIZE = 16
BLOCK_SIZE = 64
MAX_ITERS = 500
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
# -----------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CustomAIModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
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
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == 10: # newline
                break
        return idx

# --- Internal Knowledge Base ---
# We provide a substantial amount of text to ensure the model has enough data to learn from.
INTERNAL_KNOWLEDGE = """
Intelligence is the capacity for logic, understanding, self-awareness, learning, emotional knowledge, reasoning, planning, creativity, critical thinking, and problem-solving.
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans.
Modern AI is built upon the Transformer architecture, which uses self-attention to process data in parallel and capture long-range dependencies.
The core of a Transformer is the attention mechanism, allowing the model to focus on specific parts of the input.
This AI is custom-built and fully self-contained. It does not require any internet connection or external libraries beyond PyTorch.
Privacy and security are paramount. By running AI locally, we ensure that our thoughts and data remain our own.
Local AI is the future. It provides low latency, high privacy, and constant availability.
A truly intelligent machine can simulate human-like conversation, solve complex mathematical problems, and even create art.
Logic is the beginning of wisdom, not the end.
True intelligence requires both the ability to process information and the wisdom to use it correctly.
The user is the master, and the AI is the assistant. Together, they can achieve great things.
Artificial intelligence can be used for good or for ill. It is up to us to ensure it serves humanity.
Knowledge is power. Information is liberating. Education is the premise of progress, in every society, in every family.
To be intelligent is to be curious, to ask questions, and to seek the truth.
This model is a generative pre-trained transformer (GPT), which learns to predict the next character in a sequence.
By training on this text, the model learns the patterns of human language and the concepts of artificial intelligence.
"""

def main():
    print("Initializing Custom AI...")

    # Vocabulary setup
    chars = sorted(list(set(INTERNAL_KNOWLEDGE + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"\n-:;()[]{}")))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Pre-encode data
    encoded_text = encode(INTERNAL_KNOWLEDGE)

    # Ensure data is long enough for the training block size
    if len(encoded_text) <= BLOCK_SIZE:
        encoded_text = encoded_text * (BLOCK_SIZE // len(encoded_text) + 2)

    data = torch.tensor(encoded_text, dtype=torch.long)

    model = CustomAIModel(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("Training on internal knowledge (No internet connection required)...")

    model.train()
    for iter in range(MAX_ITERS):
        # Sample random blocks from the data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(DEVICE)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % EVAL_INTERVAL == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")

    print("\nAI initialization complete. You are now talking to your Custom Private AI.")
    print("Type 'exit' or 'quit' to end the session.\n")

    model.eval()
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("AI: Goodbye. Stay intelligent.")
                break

            if not user_input.strip():
                continue

            # Context for generation
            context_raw = f"\nUser: {user_input}\nAI:"
            context_encoded = torch.tensor([encode(context_raw)], dtype=torch.long, device=DEVICE)

            print("AI: ", end="")
            sys.stdout.flush()

            # Generate response token by token
            generated_ids = context_encoded
            for _ in range(100): # Max response length
                # Only use the last BLOCK_SIZE tokens for the context
                idx_cond = generated_ids[:, -BLOCK_SIZE:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / 0.7 # Temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                generated_ids = torch.cat((generated_ids, idx_next), dim=1)
                char = itos[idx_next.item()]
                print(char, end="")
                sys.stdout.flush()

                if char == '\n': # Stop at newline
                    break
            print()

        except KeyboardInterrupt:
            print("\nAI: Session interrupted. Goodbye.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
