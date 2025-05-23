"""dialog_rnn_debug.py - ULTRA VERBOSE DEBUG VERSION
================================
DialogRNN with verbose debugging throughout.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel
from torchmetrics.classification import MulticlassF1Score

print("🔧 LOADING DIALOG_RNN_DEBUG MODULE")

def focal_loss(logits, targets, alpha, gamma=2.0):
    """Compute focal loss (multiclass, per‑class alpha)."""
    print(f"📊 FOCAL_LOSS: logits shape={logits.shape}, targets shape={targets.shape}")
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma * ce).mean()
    print(f"   Computed focal loss: {loss.item():.4f}")
    return loss

class DialogRNNMLP(LightningModule):
    """Dialog‑RNN with multimodal encoders + MLP classifier - DEBUG VERSION."""

    def __init__(self, config: Any):
        print(f"🏗️ STARTING DialogRNNMLP.__init__")
        print(f"   Config type: {type(config)}")
        print(f"   Config architecture_name: {getattr(config, 'architecture_name', 'NOT_FOUND')}")
        
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])

        # constants
        self.num_classes = int(getattr(config, "output_dim", 7))
        self.hidden_gru = int(getattr(config, "gru_hidden_size", 128))
        self.context_window = int(getattr(config, "context_window", 0))
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        
        print(f"   num_classes: {self.num_classes}")
        print(f"   hidden_gru: {self.hidden_gru}")
        print(f"   context_window: {self.context_window}")
        print(f"   bidirectional: {self.bidirectional}")

        # encoders (frozen / optional unfreeze)
        print(f"   📦 Loading audio encoder...")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        print(f"   📦 Loading text encoder...")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        
        print(f"   🔒 Freezing encoders...")
        for p in self.audio_encoder.parameters(): 
            p.requires_grad = False
        for p in self.text_encoder.parameters():  
            p.requires_grad = False
        print(f"   ✅ Encoders frozen")

        # Handle fine-tuning settings
        self.audio_lr_mul = self.text_lr_mul = 0.0
        if hasattr(config, "fine_tune"):
            print(f"   🔧 Fine-tuning settings detected")
            na = int(config.fine_tune.audio_encoder.unfreeze_top_n_layers)
            nt = int(config.fine_tune.text_encoder.unfreeze_top_n_layers)
            self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers, na)
            self._unfreeze_top_n_layers(self.text_encoder.encoder.layer, nt)
            self.audio_lr_mul = float(config.fine_tune.audio_encoder.lr_mul)
            self.text_lr_mul = float(config.fine_tune.text_encoder.lr_mul)
            print(f"   Unfroze {na} audio layers, {nt} text layers")
        else:
            print(f"   No fine-tuning settings")

        # dimensionalities
        self.enc_dim = (
            self.audio_encoder.config.hidden_size
          + self.text_encoder.config.hidden_size
        )
        print(f"   enc_dim: {self.enc_dim}")

        # 2-layer bidirectional GRUs
        print(f"   🔧 Creating GRU layers...")
        self.gru_global = nn.GRU(self.enc_dim, self.hidden_gru,
                                  num_layers=2, batch_first=True,
                                  bidirectional=self.bidirectional)
        self.gru_speaker = nn.GRU(self.enc_dim, self.hidden_gru,
                                  num_layers=2, batch_first=True,
                                  bidirectional=self.bidirectional)
        self.gru_emotion = nn.GRU(self.enc_dim, self.hidden_gru,
                                  num_layers=2, batch_first=True,
                                  bidirectional=self.bidirectional)
        print(f"   ✅ Created 3 GRU layers")

        # compute total hidden dim
        total_dim = self.enc_dim + 3 * (self.hidden_gru * self.num_directions)
        print(f"   total_dim: {total_dim}")

        # LayerNorm over the time-step representations
        self.layer_norm = nn.LayerNorm(total_dim)
        print(f"   ✅ Created LayerNorm")

        # classification head
        mlp_hidden = int(getattr(config, "mlp_hidden_size", 2048))
        print(f"   mlp_hidden: {mlp_hidden}")
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, mlp_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, self.num_classes),
        )
        print(f"   ✅ Created classifier")

        # Metrics / focal‑loss alpha
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")

        alpha = torch.tensor(
            getattr(config, "class_weights", [1.0] * self.num_classes), dtype=torch.float
        )
        self.register_buffer("alpha", alpha)
        print(f"   ✅ Created metrics and alpha weights")
        
        # Final verification
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   📊 FINAL DialogRNN STATS:")
        print(f"      Total parameters: {total_params:,}")
        print(f"      Trainable parameters: {trainable_params:,}")
        print(f"   ✅ DialogRNNMLP.__init__ COMPLETED")

    @staticmethod
    def _unfreeze_top_n_layers(layers: List[nn.Module], n: int) -> None:
        print(f"   Unfreezing top {n} layers")
        for layer in layers[-max(0, n):]:
            for p in layer.parameters():
                p.requires_grad = True

    @staticmethod
    def _mask_same_speaker(u: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Return tensor where each position *t* contains only features spoken by the
        same speaker as at *t* (others are zero), keeping temporal order."""
        spk = speaker.clone()
        spk[spk < 0] = 10_000
        same = spk.unsqueeze(2) == spk.unsqueeze(1)
        u_masked = torch.einsum("bij,bjd->bid", same.float(), u)
        return u_masked

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        print(f"🔄 DialogRNN.forward() called")
        wav, wav_mask = batch["wav"], batch["wav_mask"]
        txt, txt_mask = batch["txt"], batch["txt_mask"]
        speaker_id = batch["speaker_id"]

        B, T, _ = wav.shape
        print(f"   Batch shape: B={B}, T={T}")
        print(f"   wav shape: {wav.shape}")
        print(f"   txt shape: {txt.shape}")
        print(f"   speaker_id shape: {speaker_id.shape}")

        # encode each utterance
        print(f"   🔧 Encoding audio...")
        a = self.audio_encoder(input_values=wav.flatten(0,1),
                               attention_mask=wav_mask.flatten(0,1)
                              ).last_hidden_state[:,0]
        print(f"   Audio encoded shape: {a.shape}")
        
        print(f"   🔧 Encoding text...")
        t = self.text_encoder(input_ids=txt.flatten(0,1),
                              attention_mask=txt_mask.flatten(0,1)
                             ).last_hidden_state[:,0]
        print(f"   Text encoded shape: {t.shape}")
        
        u = torch.cat([a,t], dim=-1).view(B, T, -1)
        print(f"   Combined features shape: {u.shape}")

        # optional context window
        if self.context_window > 0 and T > self.context_window:
            print(f"   Applying context window: {self.context_window}")
            u = u[:,-self.context_window:,:]
            speaker_id = speaker_id[:,-self.context_window:]

        # run all three GRUs
        print(f"   🔧 Running GRUs...")
        g_out, _ = self.gru_global(u)
        print(f"   Global GRU output: {g_out.shape}")
        
        s_out, _ = self.gru_speaker(self._mask_same_speaker(u, speaker_id))
        print(f"   Speaker GRU output: {s_out.shape}")
        
        e_out, _ = self.gru_emotion(u)
        print(f"   Emotion GRU output: {e_out.shape}")

        # concat + norm + classify
        h = torch.cat([u, g_out, s_out, e_out], dim=-1)
        print(f"   Concatenated features: {h.shape}")
        
        h = self.layer_norm(h)
        print(f"   After layer norm: {h.shape}")
        
        logits = self.classifier(h)
        print(f"   Final logits: {logits.shape}")
        print(f"✅ DialogRNN.forward() completed")
        return logits

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        print(f"📊 DialogRNN._shared_step({stage})")
        mask = batch["dialog_mask"].bool()
        labels = batch["labels"]
        print(f"   mask shape: {mask.shape}, sum: {mask.sum()}")
        print(f"   labels shape: {labels.shape}")

        logits = self(batch)
        logits_flat = logits[mask]
        labels_flat = labels[mask]
        print(f"   logits_flat shape: {logits_flat.shape}")
        print(f"   labels_flat shape: {labels_flat.shape}")

        loss = focal_loss(logits_flat, labels_flat, self.alpha,
                         gamma=float(getattr(self.config, "focal_gamma", 2.0)))
        preds = logits_flat.argmax(dim=-1)

        if stage == "train":
            self.log("train_loss", loss, prog_bar=True)
        elif stage == "val":
            self.val_f1.update(preds, labels_flat)
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", (preds == labels_flat).float().mean(), prog_bar=True)
        else:
            self.test_f1.update(preds, labels_flat)
            self.log("test_loss", loss)
            self.log("test_acc", (preds == labels_flat).float().mean())
        
        print(f"   {stage} loss: {loss.item():.4f}")
        return loss

    def training_step(self, batch, batch_idx):
        print(f"🏋️ DialogRNN.training_step({batch_idx})")
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        print(f"✅ DialogRNN.validation_step({batch_idx})")
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        print(f"🧪 DialogRNN.test_step({batch_idx})")
        self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        print(f"📈 DialogRNN.on_validation_epoch_end()")
        f1 = self.val_f1.compute()
        self.log("val_f1", f1, prog_bar=True)
        self.val_f1.reset()
        print(f"   Validation F1: {f1:.4f}")

    def on_test_epoch_end(self):
        print(f"🏁 DialogRNN.on_test_epoch_end()")
        f1 = self.test_f1.compute()
        self.log("test_f1", f1)
        self.test_f1.reset()
        print(f"   Test F1: {f1:.4f}")

    def configure_optimizers(self):
        print(f"⚙️ DialogRNN.configure_optimizers()")
        base_lr = float(self.config.learning_rate)
        wd = float(getattr(self.config, "weight_decay", 1e-4))
        print(f"   base_lr: {base_lr}, weight_decay: {wd}")

        groups = []
        if any(p.requires_grad for p in self.audio_encoder.parameters()):
            groups.append({"params": [p for p in self.audio_encoder.parameters() if p.requires_grad],
                           "lr": base_lr * self.audio_lr_mul})
            print(f"   Added audio encoder group with lr_mul: {self.audio_lr_mul}")
        if any(p.requires_grad for p in self.text_encoder.parameters()):
            groups.append({"params": [p for p in self.text_encoder.parameters() if p.requires_grad],
                           "lr": base_lr * self.text_lr_mul})
            print(f"   Added text encoder group with lr_mul: {self.text_lr_mul}")

        ctx_params = list(self.gru_global.parameters()) + \
                     list(self.gru_speaker.parameters()) + \
                     list(self.gru_emotion.parameters()) + \
                     list(self.classifier.parameters())
        groups.append({"params": ctx_params, "lr": base_lr})
        print(f"   Added context + classifier group")

        optimizer = optim.AdamW(groups, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.num_epochs),
            eta_min=float(getattr(self.config, "eta_min", 1e-7)),
        )
        print(f"   ✅ Created optimizer and scheduler")
        return [optimizer], [scheduler]

print("✅ DIALOG_RNN_DEBUG MODULE LOADED") 