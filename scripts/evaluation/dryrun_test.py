# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
dryrun_test.py — local CPU validation of the custom logic in the 4 tuned Model B notebooks,
BEFORE spending Colab GPU time. Validates (a) the PyTorch SupCon loss + balanced sampler +
forward_joint + a real joint train step (torch, runs directly), and (b) the TensorFlow SupCon /
focal MATH by mirroring the exact notebook formulas in NumPy and cross-checking them against the
torch implementations on identical inputs (TF can't import locally — protobuf mismatch — but it
runs on Colab; the math is what we verify here).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
ok = lambda name, cond: print(("  PASS " if cond else "  ***FAIL*** ") + name)
print("=" * 70)

# ───────────────────────── 1. PyTorch SupCon (contrastive DistilBERT) ─────────────────────────
print("1. PyTorch supcon_loss (from train_yelp_multimodal_distilbert_contrastive.ipynb)")
def supcon_loss_torch(emb, labels, temperature):
    dev = emb.device; B = emb.shape[0]
    sim = torch.matmul(emb, emb.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    self_mask = torch.eye(B, dtype=torch.bool, device=dev)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~self_mask
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_cnt = pos_mask.sum(dim=1)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_cnt.clamp(min=1)
    valid = pos_cnt > 0
    return -(mean_log_prob[valid].mean()) if valid.any() else torch.zeros((), device=dev)

B, D = 16, 128
emb = F.normalize(torch.randn(B, D, requires_grad=True), dim=1)
labels = torch.tensor([0, 1] * (B // 2), dtype=torch.float32)
L = supcon_loss_torch(emb, labels, 0.3)
ok("returns finite scalar", L.dim() == 0 and torch.isfinite(L))
ok("non-negative (it's -mean log-prob of a softmax)", L.item() >= -1e-6)
g = torch.autograd.grad(L, emb)[0]
ok("gradient flows, finite, right shape", g.shape == (B, D) and torch.isfinite(g).all())

# clustered same-label embeddings should give MUCH lower SupCon than random
e_clust = torch.zeros(B, D)
for i in range(B):
    base = torch.zeros(D); base[int(labels[i].item())] = 5.0      # class 0 -> dim0, class 1 -> dim1
    e_clust[i] = base + 0.01 * torch.randn(D)
e_clust = F.normalize(e_clust, dim=1)
e_rand = F.normalize(torch.randn(B, D), dim=1)
ok("clustered loss < random loss (pulls same-label together)",
   supcon_loss_torch(e_clust, labels, 0.3).item() < supcon_loss_torch(e_rand, labels, 0.3).item())

# all-one-class batch -> no positives for the minority... here every anchor shares label, so valid
allpos = torch.ones(B, dtype=torch.float32)
ok("single-label batch still finite (every anchor has positives)",
   torch.isfinite(supcon_loss_torch(e_rand, allpos, 0.3)))
# a batch where each label is unique -> NO positives -> must return 0, not NaN
uniq = torch.arange(B, dtype=torch.float32)
ok("no-positives batch returns finite 0 (no NaN)",
   torch.isfinite(supcon_loss_torch(e_rand, uniq, 0.3)))

# ───────────────────────── 2. BalancedBatchSampler (contrastive DistilBERT) ─────────────────────────
print("\n2. BalancedBatchSampler (50/50 batches over ~11%-fake train)")
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.asarray(labels).astype(int); self.bs = batch_size; self.half = batch_size // 2
        self.pos = np.where(self.labels == 1)[0]; self.neg = np.where(self.labels == 0)[0]
        self.n_batches = len(self.neg) // self.half
    def __iter__(self):
        neg = np.random.permutation(self.neg)
        pos = np.random.choice(self.pos, size=self.n_batches * self.half, replace=True)
        for b in range(self.n_batches):
            sl = slice(b * self.half, (b + 1) * self.half)
            idx = np.concatenate([neg[sl], pos[sl]]); np.random.shuffle(idx)
            yield idx.tolist()
    def __len__(self): return self.n_batches

lab = np.array([0] * 8900 + [1] * 1100)        # ~11% fake, like the real train
bs = BalancedBatchSampler(lab, 64)
batches = list(iter(bs))
ok("len matches n_batches", len(batches) == len(bs) == 8900 // 32)
fracs = [lab[np.array(b)].mean() for b in batches]
ok("every batch is exactly 50/50 fake", all(abs(f - 0.5) < 1e-9 for f in fracs))
sizes = [len(b) for b in batches]
ok("every batch is full size 64", all(s == 64 for s in sizes))
ok("indices in range", all(0 <= i < len(lab) for b in batches for i in b))

# ───────────────────────── 3. forward_joint + a real joint train step (torch plumbing) ─────────────────────────
print("\n3. fit_once joint CE+SupCon step (stub encoder in place of DistilBERT, same contract)")
class StubMM(nn.Module):
    """Mirrors MultiModalDistilBert's head/proj/forward_joint contract with a cheap text encoder."""
    def __init__(self, n_beh, vocab=200, d=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, 768)                       # stands in for DistilBERT [CLS]
        self.text_proj = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3))
        self.beh = nn.Sequential(nn.Linear(n_beh, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256 + 32, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))
        self.proj = nn.Sequential(nn.Linear(256 + 32, 128), nn.ReLU(), nn.Linear(128, 128))
    def _fused(self, ids, beh):
        return torch.cat([self.text_proj(self.emb(ids)), self.beh(beh)], dim=1)
    def forward(self, ids, beh):
        return self.head(self._fused(ids, beh)).squeeze(-1)
    def forward_joint(self, ids, beh):
        fused = self._fused(ids, beh)
        return self.head(fused).squeeze(-1), F.normalize(self.proj(fused), dim=1)

n_beh = 17
m = StubMM(n_beh)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss(reduction="none")
ids = torch.randint(0, 200, (B, 5))
beh = torch.randn(B, n_beh)
y = labels.clone()
lam, temp = 0.3, 0.3
losses = []
for step in range(40):
    opt.zero_grad()
    logit, e = m.forward_joint(ids, beh)
    ce = loss_fn(logit, y).mean()
    sc = supcon_loss_torch(e.float(), y, temp)
    loss = (1 - lam) * ce + lam * sc
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    losses.append(loss.item())
ok("forward_joint -> logit[B] and unit-norm emb[B,128]",
   logit.shape == (B,) and e.shape == (B, 128) and torch.allclose(e.norm(dim=1), torch.ones(B), atol=1e-5))
ok("joint loss is finite throughout", all(np.isfinite(losses)))
ok("joint loss decreases on a tiny overfit set", losses[-1] < losses[0])
# serving forward() (the exported path) must give a plain logit, no proj dependency
ok("serving forward() returns logit[B] (drop-in path)", m(ids, beh).shape == (B,))

# ───────────────────────── 4. Focal loss (torch, DistilBERT focal) ─────────────────────────
print("\n4. Focal loss (from train_yelp_multimodal_distilbert_focal.ipynb)")
def focal_loss_torch(logit, target, gamma, alpha):
    p = torch.sigmoid(logit)
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (a_t * (1 - p_t) ** gamma * bce).mean()

logit = torch.randn(B, requires_grad=True)
fl = focal_loss_torch(logit, y, 2.0, 0.25)
ok("focal finite + grad flows", torch.isfinite(fl) and torch.isfinite(torch.autograd.grad(fl, logit)[0]).all())
# gamma=0, alpha=0.5 must reduce to 0.5*BCE (focal collapses to weighted BCE)
with torch.no_grad():
    f0 = focal_loss_torch(logit, y, 0.0, 0.5)
    bce_mean = 0.5 * F.binary_cross_entropy_with_logits(logit, y, reduction="none").mean()
ok("gamma=0,alpha=.5 collapses to 0.5*BCE", torch.allclose(f0, bce_mean, atol=1e-6))
# focal must down-weight easy (confident-correct) examples vs hard ones
easy = torch.tensor([5.0]); hard = torch.tensor([0.1]); tgt = torch.tensor([1.0])
ok("focal weights hard example > easy example",
   focal_loss_torch(hard, tgt, 2.0, 0.5).item() > focal_loss_torch(easy, tgt, 2.0, 0.5).item())

# ───────────────────────── 5. TF SupCon MATH mirrored in NumPy, cross-checked vs torch ─────────────────────────
print("\n5. TF supcon_loss MATH (LSTM contrastive) mirrored in NumPy == torch value")
def supcon_np_tfstyle(emb, labels, temperature):
    # exact transcription of the TF code in train_yelp_multimodal_lstm_contrastive.ipynb cell-5
    labels = labels.reshape(-1, 1).astype("float32")
    Bn = emb.shape[0]
    sim = (emb @ emb.T) / temperature
    sim = sim - sim.max(axis=1, keepdims=True)            # stop_gradient irrelevant for value
    logits_mask = 1.0 - np.eye(Bn)
    pos_mask = (labels == labels.T).astype("float32") * logits_mask
    exp_sim = np.exp(sim) * logits_mask
    log_prob = sim - np.log(exp_sim.sum(axis=1, keepdims=True) + 1e-12)
    pos_cnt = pos_mask.sum(axis=1)
    mean_log_prob = (pos_mask * log_prob).sum(axis=1) / np.maximum(pos_cnt, 1.0)
    valid = (pos_cnt > 0).astype("float32")
    return -(mean_log_prob * valid).sum() / max(valid.sum(), 1.0)

e_np = e_clust.detach().numpy().astype("float32"); lab_np = labels.numpy()
tf_val = supcon_np_tfstyle(e_np, lab_np, 0.3)
torch_val = supcon_loss_torch(e_clust, labels, 0.3).item()
ok(f"TF-style NumPy ({tf_val:.5f}) == torch ({torch_val:.5f})", abs(tf_val - torch_val) < 1e-4)

# ───────────────────────── 6. TF focal MATH mirrored in NumPy, cross-checked vs torch ─────────────────────────
print("\n6. TF focal MATH (LSTM focal) mirrored in NumPy == torch value")
def focal_np_tfstyle(y_true, y_pred, gamma, alpha):
    # transcription of the Keras focal in train_yelp_multimodal_lstm_focal.ipynb cell-5
    y_true = y_true.astype("float32")
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    a_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return float(np.mean(a_t * (1 - p_t) ** gamma * bce))
probs = torch.sigmoid(logit).detach().numpy()
tf_f = focal_np_tfstyle(y.numpy(), probs, 2.0, 0.25)
torch_f = focal_loss_torch(logit.detach(), y, 2.0, 0.25).item()   # torch takes logits, computes sigmoid inside
ok(f"TF-style focal ({tf_f:.5f}) == torch focal ({torch_f:.5f})", abs(tf_f - torch_f) < 1e-5)

print("\n" + "=" * 70 + "\nDry-run complete.")
