# ───────────────────────────────────────────────────────────────────────────────
# 🔍 Sanity‐check script for MLP, Dialog‐RNN or TOD‐KAT‐lite
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from utils.data_processor import MELDDataModule
from models.dialog_rnn  import DialogRNNMLP
from models.mlp_fusion  import MultimodalFusionMLP     # adjust import paths as needed
from models.todkat_lite import TodkatLiteMLP

# ───────────────────────────────────────────────────────────────────────────────
# 1) Pick your config here:
config_path = "configs/colab_config_dialog_rnn.yaml"  # or _fusion.yaml, _todkat_lite.yaml
# ───────────────────────────────────────────────────────────────────────────────

cfg = OmegaConf.load(config_path)
cfg.setdefault("text_encoder_model_name",  "roberta-base")
cfg.setdefault("audio_encoder_model_name", "facebook/wav2vec2-base-960h")
cfg.setdefault("output_dim", 7)

# ---- FORCE a tiny batch for speed ----
cfg.batch_size = 1       # one dialogue at a time
# you could also reduce num_workers to 0 if debugging:
cfg.dataloader_num_workers = 0

# 2) build DataModule + DataLoader
dm = MELDDataModule(cfg)
dm.setup()
loader = dm.train_dataloader()

# 3) build model
arch = cfg.architecture.lower()
if arch == "dialog_rnn":
    model = DialogRNNMLP(cfg)
elif arch == "todkat_lite":
    model = TodkatLiteMLP(cfg)
else:
    model = MultimodalFusionMLP(cfg)
model.train()

# 4) grab & truncate
batch = next(iter(loader))
# truncate to at most 4 turns to keep the forward very cheap
for k, v in batch.items():
    # only tensors with shape (1, T, ...)
    if isinstance(v, torch.Tensor) and v.ndim >= 2:
        batch[k] = v[:, :4].clone()

# 5) print shapes
print("Batch shapes (truncated to 4 turns):")
for k,v in batch.items():
    print(f"  {k:12s} → {tuple(v.shape)}")
B, T = batch["wav"].shape[:2]
assert B==1 and T<=4

# 6) forward + backward
logits = model(batch)
mask   = batch.get("dialog_mask", torch.ones(B,T, dtype=torch.bool))
labels = batch["labels"]
flat_logits = logits[mask].view(-1, cfg.output_dim)
flat_labels = labels[mask].view(-1)
loss = F.cross_entropy(flat_logits, flat_labels)
print("Fast forward OK, loss =", loss.item())
loss.backward()

# 7) grads check
got_clf = any(p.grad is not None for p in model.classifier.parameters())
got_gru = hasattr(model, "gru_global") and any(
    p.grad is not None for p in model.gru_global.parameters()
)
print("Classifier grads? ", got_clf)
if arch == "dialog_rnn":
    print("Global-GRU grads? ", got_gru)
    assert got_gru, "No grads in your GRU—something’s still wired wrong."

# 8) GRU is doing something
if hasattr(model, "gru_global"):
    with torch.no_grad():
        # rebuild u
        wav, wav_mask = batch["wav"], batch["wav_mask"]
        txt, txt_mask = batch["txt"], batch["txt_mask"]
        a_emb = model.audio_encoder(
            input_values=wav.flatten(0,1),
            attention_mask=wav_mask.flatten(0,1)
        ).last_hidden_state[:,0,:]
        t_emb = model.text_encoder(
            input_ids=txt.flatten(0,1),
            attention_mask=txt_mask.flatten(0,1)
        ).last_hidden_state[:,0,:]
        u = torch.cat([a_emb,t_emb], dim=-1).view(B,T,-1)
        g_out,_ = model.gru_global(u)
        assert not torch.allclose(
            u[0,0,:4], g_out[0,0,:4]
        ), "GRU appears to be identity!"

print("\n✅ Fast sanity‐check passed. Your dialog-level wiring is live!")
