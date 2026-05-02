"""
GRPO (Group Relative Policy Optimization) — minimal implementation.

Architecture: tiny GPT (4 layers, 128 dim, 4 heads, vocab=50)
Task: generate sequences; reward = count of token '1' in output (dummy task)
GRPO: no value function; advantage = (r - mean_r) / std_r within group of G samples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from gui.grpo_viewer import GRPOInferenceViewer

# ── Hyperparams ──────────────────────────────────────────────────────────────
VOCAB      = 50
SEQ_LEN    = 16        # max sequence length
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
D_FF       = 256
DROPOUT    = 0.1

G          = 8         # group size: samples per prompt
EPOCHS     = 200
LR         = 3e-4
EPS_CLIP   = 0.2       # PPO-style clip
BETA_KL    = 0.01      # KL penalty weight
BATCH_SIZE = 4         # prompts per update

PAD = 0
BOS = 1
EOS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


# ── Tiny GPT ─────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att = self.drop(F.softmax(att, dim=-1))
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.ff   = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(D_MODEL)
        self.head    = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        x    = self.ln_f(x)
        return self.head(x)   # (B, T, VOCAB)

    @torch.no_grad()
    def generate(self, prompt, max_new=8, temperature=1.0):
        """Autoregressive sample. Returns (tokens, log_probs)."""
        self.eval()
        idx      = prompt.clone()          # (B, T_prompt)
        log_probs = []
        for _ in range(max_new):
            logits = self.forward(idx)[:, -1, :]   # (B, VOCAB)
            logits = logits / temperature
            probs  = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)  # (B, 1)
            log_p    = torch.log(probs.gather(1, next_tok) + 1e-8)
            log_probs.append(log_p)
            idx = torch.cat([idx, next_tok], dim=1)
            if (next_tok == EOS).all():
                break
        self.train()
        return idx[:, prompt.shape[1]:], torch.cat(log_probs, dim=1)  # generated tokens, log_probs


# ── Reward ────────────────────────────────────────────────────────────────────
def reward_fn(tokens: torch.Tensor) -> torch.Tensor:
    """
    Dummy reward: count how many tokens equal token-id 5.
    Range [0, max_new]. Higher = better.
    """
    return (tokens == 5).float().sum(dim=1)


# ── Log-prob of sequence under model ─────────────────────────────────────────
def sequence_log_prob(model, prompt, generated):
    """
    Compute sum of log-probs for generated tokens given prompt context.
    prompt:    (B, T_p)
    generated: (B, T_g)
    Returns:   (B,)
    """
    full   = torch.cat([prompt, generated], dim=1)   # (B, T_p+T_g)
    logits = model(full)                              # (B, T, VOCAB)
    T_g    = generated.shape[1]
    # logits for generated positions: shift by prompt length
    # position i in generated corresponds to logits[:, T_p - 1 + i, :]
    T_p    = prompt.shape[1]
    gen_logits = logits[:, T_p - 1: T_p - 1 + T_g, :]  # (B, T_g, VOCAB)
    log_probs  = F.log_softmax(gen_logits, dim=-1)
    tok_lp     = log_probs.gather(2, generated.unsqueeze(2)).squeeze(2)  # (B, T_g)
    return tok_lp.sum(dim=1)   # (B,)


# ── GRPO update ───────────────────────────────────────────────────────────────
def grpo_update(policy, ref_policy, optimizer, prompts):
    """
    prompts: (batch, T_p) — batch of prompt token sequences
    """
    B = prompts.shape[0]

    all_gen   = []   # list of (B, T_g)
    all_lp    = []   # list of (B,)  — old log-probs
    all_r     = []   # list of (B,)  — rewards

    # Step 1: generate G outputs per prompt
    policy.eval()
    with torch.no_grad():
        for _ in range(G):
            gen, lp = policy.generate(prompts, max_new=8)
            r = reward_fn(gen)
            all_gen.append(gen)
            all_lp.append(lp.sum(dim=1))    # sum log-probs over tokens
            all_r.append(r)

    # all_* shapes: (G, B)
    rewards    = torch.stack(all_r,  dim=0)   # (G, B)
    old_lp     = torch.stack(all_lp, dim=0)   # (G, B)

    # Step 2: group-relative advantage
    mean_r = rewards.mean(dim=0, keepdim=True)   # (1, B)
    std_r  = rewards.std(dim=0, keepdim=True).clamp(min=1e-8)
    advantages = (rewards - mean_r) / std_r       # (G, B)

    # Step 3: policy gradient with clipping + KL penalty
    policy.train()
    total_loss = torch.tensor(0.0, device=device)

    for g in range(G):
        gen = all_gen[g]                          # (B, T_g)
        adv = advantages[g]                       # (B,)

        new_lp  = sequence_log_prob(policy, prompts, gen)   # (B,)
        ref_lp  = sequence_log_prob(ref_policy, prompts, gen).detach()

        ratio   = torch.exp(new_lp - old_lp[g].detach())
        clipped = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
        pg_loss = -torch.min(ratio * adv, clipped * adv).mean()

        # KL(policy || ref) approx via log-ratio
        kl      = (new_lp - ref_lp).mean()
        loss    = pg_loss + BETA_KL * kl
        total_loss = total_loss + loss

    total_loss = total_loss / G
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    mean_reward = rewards.mean().item()
    return total_loss.item(), mean_reward


# ── Prompt generator ──────────────────────────────────────────────────────────
def random_prompts(batch_size, prompt_len=4):
    """Random prompts with BOS prefix, tokens from [3, VOCAB)."""
    tokens = torch.randint(3, VOCAB, (batch_size, prompt_len - 1), device=device)
    bos    = torch.full((batch_size, 1), BOS, device=device)
    return torch.cat([bos, tokens], dim=1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("GRPO Minimal LLM Training")
    print(f"Model: {N_LAYERS}L {D_MODEL}d {N_HEADS}H | Vocab={VOCAB}")
    print(f"Group size G={G} | Clip ε={EPS_CLIP} | KL β={BETA_KL}")
    print("=" * 50)

    policy     = TinyGPT().to(device)
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        prompts = random_prompts(BATCH_SIZE)
        loss, mean_r = grpo_update(policy, ref_policy, optimizer, prompts)
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={loss:.4f} | mean_reward={mean_r:.3f}")

    print("\nTraining done. Launching inference simulation...")

    # ── Post-training GUI simulation ──────────────────────────────────────────
    viewer = GRPOInferenceViewer(
        policy          = policy,
        reward_fn       = reward_fn,
        random_prompts_fn = lambda n: random_prompts(n, prompt_len=4),
        vocab           = VOCAB,
        n_runs          = 30,
    )
    viewer.run()


if __name__ == "__main__":
    main()
