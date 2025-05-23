"""dialog_rnn.py
================================
Speaker‑aware Dialog‑RNN for multimodal emotion recognition (MELD).

* **Encoders**  : Wav2Vec 2.0 + RoBERTa (frozen by default, optional top‑N unfreeze).
* **Context**    : Three GRUs — global, speaker‑specific, emotion — as in Majumder et al. (2019).
* **Fusion**     : `[u_t | g_t | s_t | e_t] → 2‑layer MLP → logits` (identical head dims to baseline).
* **Loss**       : Focal loss with per‑class weights.

Required batch keys
-------------------
```
wav         (B, T, L_a)      float32   – raw audio     (16 kHz)
wav_mask    (B, T, L_a)      int64     – padding mask for audio encoder (1 = keep)
txt         (B, T, L_t)      int64     – token IDs
txt_mask    (B, T, L_t)      int64     – text padding mask (1 = keep)
speaker_id  (B, T)           int64     – speaker index, −1 for PAD
dialog_mask (B, T)           int64     – 1 = valid utterance, 0 = PAD
labels      (B, T)           int64     – gold emotions
```
All shapes are *dialogue‑level* (time‑axis **T**).  Utterance‑only batches still work with `T=1`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel
from torchmetrics.classification import MulticlassF1Score

################################################################################
# Loss helper
################################################################################

def focal_loss(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    alpha: torch.FloatTensor,
    gamma: float = 2.0,
) -> torch.FloatTensor:
    """Compute focal loss (multiclass, per‑class alpha)."""
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()

################################################################################
# Dialog‑RNN model
################################################################################

class DialogRNNMLP(LightningModule):
    """Dialog‑RNN with multimodal encoders + MLP classifier."""

    # ---------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])

        # constants
        self.num_classes    = int(getattr(config, "output_dim", 7))
        self.hidden_gru     = int(getattr(config, "gru_hidden_size", 128))
        self.context_window = int(getattr(config, "context_window", 0))
        self.bidirectional  = True
        self.num_directions = 2 if self.bidirectional else 1

        # encoders (frozen / optional unfreeze)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_encoder  = RobertaModel.from_pretrained("roberta-base")
        for p in self.audio_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters():  p.requires_grad = False

        # Optional encoder fine-tuning - SIMPLIFIED  
        self.audio_lr_mul = self.text_lr_mul = 0.0
        
        unfreeze_audio = int(getattr(config, "unfreeze_audio_layers", 0))
        unfreeze_text = int(getattr(config, "unfreeze_text_layers", 0))
        
        if unfreeze_audio > 0 or unfreeze_text > 0:
            self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers, unfreeze_audio)
            self._unfreeze_top_n_layers(self.text_encoder.encoder.layer, unfreeze_text)
            self.audio_lr_mul = float(getattr(config, "audio_lr_mul", 1.0))
            self.text_lr_mul = float(getattr(config, "text_lr_mul", 1.0))

        # dimensionalities
        self.enc_dim = (
            self.audio_encoder.config.hidden_size
          + self.text_encoder.config.hidden_size
        )

        # 2-layer bidirectional GRUs
        self.gru_global  = nn.GRU(self.enc_dim, self.hidden_gru,
                                  num_layers=2, batch_first=True,
                                  bidirectional=self.bidirectional)
        self.gru_speaker = nn.GRU(self.enc_dim, self.hidden_gru,
                                  num_layers=2, batch_first=True,
                                  bidirectional=self.bidirectional)
        self.gru_emotion = nn.GRU(self.enc_dim, self.hidden_gru,
                                  num_layers=2, batch_first=True,
                                  bidirectional=self.bidirectional)

        # compute total hidden dim: enc_dim + 3 * (hidden_gru * num_directions)
        total_dim = self.enc_dim + 3 * (self.hidden_gru * self.num_directions)

        # LayerNorm over the time-step representations
        self.layer_norm = nn.LayerNorm(total_dim)

        # classification head (same depth, bigger MLP)
        mlp_hidden = int(getattr(config, "mlp_hidden_size", 2048))
        self.classifier = nn.Sequential(
            nn.Linear(total_dim,      mlp_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden,     mlp_hidden // 2),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, self.num_classes),
        )

        # Metrics / focal‑loss alpha ---------------------------------------
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")

        alpha = torch.tensor(
            getattr(config, "class_weights", [1.0] * self.num_classes), dtype=torch.float
        )
        self.register_buffer("alpha", alpha)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def _unfreeze_top_n_layers(layers: List[nn.Module], n: int) -> None:
        for layer in layers[-max(0, n) :]:
            for p in layer.parameters():
                p.requires_grad = True

    @staticmethod
    def _mask_same_speaker(u: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Return tensor where each position *t* contains only features spoken by the
        same speaker as at *t* (others are zero), keeping temporal order.
        Vectorised: avoids Python loops.  Shape preserved (B,T,D).
        """
        # speaker: (B,T) with –1 pads → set pads to large unique id so comparison yields False.
        spk = speaker.clone()
        spk[spk < 0] = 10_000  # unlikely high id
        # Broadcast compare – result (B,T,T)
        same = spk.unsqueeze(2) == spk.unsqueeze(1)  # (B,T,T)
        u_masked = torch.einsum("bij,bjd->bid", same.float(), u)
        return u_masked

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        wav, wav_mask = batch["wav"], batch["wav_mask"]
        txt, txt_mask = batch["txt"], batch["txt_mask"]
        speaker_id    = batch["speaker_id"]

        B, T, _ = wav.shape
        # encode each utterance
        a = self.audio_encoder(input_values=wav.flatten(0,1),
                               attention_mask=wav_mask.flatten(0,1)
                              ).last_hidden_state[:,0]
        t = self.text_encoder(input_ids=txt.flatten(0,1),
                              attention_mask=txt_mask.flatten(0,1)
                             ).last_hidden_state[:,0]
        u = torch.cat([a,t], dim=-1).view(B, T, -1)

        # optional context window
        if self.context_window>0 and T>self.context_window:
            u = u[:,-self.context_window:,:]
            speaker_id = speaker_id[:,-self.context_window:]

        # run all three GRUs
        g_out, _ = self.gru_global(u)
        s_out, _ = self.gru_speaker(self._mask_same_speaker(u, speaker_id))
        e_out, _ = self.gru_emotion(u)

        # concat + norm + classify
        h = torch.cat([u, g_out, s_out, e_out], dim=-1)  # (B,T,total_dim)
        h = self.layer_norm(h)
        logits = self.classifier(h)                    # (B,T,C)
        return logits

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        mask = batch["dialog_mask"].bool()
        labels = batch["labels"]

        logits = self(batch)
        
        # If context window was applied in forward(), we need to truncate mask and labels too
        if self.context_window > 0 and mask.shape[1] > self.context_window and logits.shape[1] == self.context_window:
            mask = mask[:, -self.context_window:]
            labels = labels[:, -self.context_window:]
        
        logits_flat = logits[mask]
        labels_flat = labels[mask]

        loss = focal_loss(
            logits_flat,
            labels_flat,
            self.alpha,
            gamma=float(getattr(self.config, "focal_gamma", 2.0)),
        )
        preds = logits_flat.argmax(dim=-1)

        if stage == "train":
            self.log("train_loss", loss, prog_bar=True)
        elif stage == "val":
            self.val_f1.update(preds, labels_flat)
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", (preds == labels_flat).float().mean(), prog_bar=True)
        else:  # test
            self.test_f1.update(preds, labels_flat)
            self.log("test_loss", loss)
            self.log("test_acc", (preds == labels_flat).float().mean())
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.log("test_f1", self.test_f1.compute())
        self.test_f1.reset()

    # ------------------------------------------------------------------
    #  Optimiser & scheduler (mirrors baseline)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        base_lr = float(self.config.learning_rate)
        wd = float(getattr(self.config, "weight_decay", 1e-4))

        groups: List[dict[str, Any]] = []

        if any(p.requires_grad for p in self.audio_encoder.parameters()):
            groups.append({"params": [p for p in self.audio_encoder.parameters() if p.requires_grad],
                           "lr": base_lr * self.audio_lr_mul})
        if any(p.requires_grad for p in self.text_encoder.parameters()):
            groups.append({"params": [p for p in self.text_encoder.parameters() if p.requires_grad],
                           "lr": base_lr * self.text_lr_mul})

        # Context + classifier always train
        ctx_params = list(self.gru_global.parameters()) + \
                     list(self.gru_speaker.parameters()) + \
                     list(self.gru_emotion.parameters()) + \
                     list(self.classifier.parameters())
        groups.append({"params": ctx_params, "lr": base_lr})

        optimizer = optim.AdamW(groups, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.num_epochs),
            eta_min=float(getattr(self.config, "eta_min", 1e-7)),
        )
        return [optimizer], [scheduler]
