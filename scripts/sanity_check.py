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
#config_path = "configs/colab_config_dialog_rnn.yaml"  # or _fusion.yaml, _todkat_lite.yaml
# ───────────────────────────────────────────────────────────────────────────────
cfg = OmegaConf.load(config_path)
# backfill minimal defaults
cfg.setdefault("text_encoder_model_name",  "roberta-base")
cfg.setdefault("audio_encoder_model_name", "facebook/wav2vec2-base-960h")
cfg.setdefault("output_dim", 7)

# 2) DataModule + Model factory
dm = MELDDataModule(cfg); dm.setup()
loader = dm.train_dataloader()

arch = cfg.architecture.lower()
if arch == "dialog_rnn":
    model = DialogRNNMLP(cfg)
elif arch == "todkat_lite":
    model = TodkatLiteMLP(cfg)
else:
    model = FusionMLP(cfg)  # your MLP‐fusion class

model.train()  # so gradients will flow

# 3) Grab a single batch
raw_batch = next(iter(loader))

# 4) Auto-unpack tuple→dict if necessary
if isinstance(raw_batch, (list, tuple)):
    # your tuple‐order may vary; adjust these names/order to match your collate
    names = ["wav","wav_mask","txt","txt_mask"]
    if arch == "dialog_rnn":
        names += ["labels","speaker_id","dialog_mask"]
    elif arch == "todkat_lite":
        names += ["labels","topic_id","dialog_mask","kn_vec"]
    else:
        names += ["labels"]
    batch = dict(zip(names, raw_batch))
else:
    batch = raw_batch

# 5) Print shapes
print("\n=== Batch keys & shapes ===")
for k,v in batch.items():
    print(f"  {k:12s} → {tuple(v.shape)}")
B, T = batch["wav"].shape[:2]
assert B > 1, f"Batch dim B should be >1, got {B}"
assert T > 1, f"Time dim T should be >1, got {T}"

# 6) One forward + backward
logits = model(batch)  # should be (B,T,C) for seq models, (B,C) for MLP
# pick a loss that works generically:
mask = batch.get("dialog_mask", torch.ones(B,T, dtype=torch.bool))
labels = batch["labels"]
flat_logits = logits[mask].view(-1, cfg.output_dim)
flat_labels = labels[mask].view(-1)
loss = F.cross_entropy(flat_logits, flat_labels)
print("\nForward OK, loss =", loss.item())
loss.backward()

# 7) Check that at least one GRU & classifier param got a grad
got_gru  = any((p.grad is not None) for p in getattr(model, "gru_global", []))
got_clf  = any((p.grad is not None) for p in model.classifier.parameters())
print("GRU grads?      ", got_gru)
print("Classifier grads?", got_clf)
assert got_clf, "No grads in your classifier—optimizer groups?"
if arch == "dialog_rnn":
    assert got_gru, "No grads in your global‐GRU—check collate/masks or optimizer."

# 8) Compare raw vs context embeddings (dialog_rnn / todkat only)
if hasattr(model, "gru_global"):
    with torch.no_grad():
        # rebuild raw u
        wav, wav_mask = batch["wav"], batch["wav_mask"]
        txt, txt_mask = batch["txt"], batch["txt_mask"]
        a_emb = model.audio_encoder(
            input_values=wav.flatten(0,1), attention_mask=wav_mask.flatten(0,1)
        ).last_hidden_state[:,0,:]
        t_emb = model.text_encoder(
            input_ids=txt.flatten(0,1), attention_mask=txt_mask.flatten(0,1)
        ).last_hidden_state[:,0,:]
        u = torch.cat([a_emb,t_emb], dim=-1).view(B,T,-1)
        g_out,_ = model.gru_global(u)
        print("\nExample raw u[0,0]:", u[0,0,:4])
        print("   global g[0,0]:",   g_out[0,0,:4])
        assert not torch.allclose(u[0,0,:4], g_out[0,0,:4]), "GRU is identity!"

print("\n✅ All checks passed—you’re truly training dialogue‐level context now.")
