"""MLP Fusion model for multimodal emotion classification.

Encoders: Wav2Vec 2.0 + DistilRoBERTa
Fusion:   [audio CLS | text CLS] → 2‑layer MLP
Loss:     Focal loss with class weights

This version fixes ⚠️ issues spotted in review:
  • test_step now uses focal_loss consistently
  • safely handles LR multipliers when encoders stay frozen
  • pools audio using the CLS/output‑0 vector instead of mean‑pooling
"""

from __future__ import annotations

# ‑‑ python stdlib
from typing import Any, List

# ‑‑ third‑party
import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel
from torchmetrics.classification import MulticlassF1Score

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    alpha: torch.FloatTensor,
    gamma: float = 2.0,
) -> torch.FloatTensor:
    """Focal loss for class‑imbalance handling.

    Args:
        logits:  (B, C) raw outputs
        targets: (B,)   gold labels
        alpha:   (C,)   per‑class weights (≈ inverse frequency)
        gamma:   focusing parameter (default 2.0)
    """
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)  # model‑estimated prob for the gold class
    return ((1.0 - pt) ** gamma * ce).mean()

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class MultimodalFusionMLP(LightningModule):
    """Wav2Vec2 + RoBERTa → feature concat → MLP classifier."""

    def __init__(self, config: Any, **kwargs: Any):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])  # full YAML is huge

        # ------------------------------------------------------------------
        # 1)   Encoders (frozen by default)
        # ------------------------------------------------------------------
        self.audio_encoder: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.text_encoder: RobertaModel = RobertaModel.from_pretrained("roberta-base")

        # Freeze everything first; we may unfreeze selectively below
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Optional fine‑tune policy ------------------------------------------------
        if hasattr(self.config, "fine_tune"):
            n_audio = int(getattr(self.config.fine_tune.audio_encoder, "unfreeze_top_n_layers", 0))
            n_text = int(getattr(self.config.fine_tune.text_encoder, "unfreeze_top_n_layers", 0))
            self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers, n_audio)
            self._unfreeze_top_n_layers(self.text_encoder.encoder.layer, n_text)

            self.audio_lr_mul = float(
                getattr(self.config.fine_tune.audio_encoder, "lr_mul", 1.0)
            )
            self.text_lr_mul = float(
                getattr(self.config.fine_tune.text_encoder, "lr_mul", 1.0)
            )
        else:
            # Keep encoders frozen but still register a param‑group with LR 0 ➜
            # avoids PyTorch warnings about unused parameters in DDP.
            self.audio_lr_mul = 0.0
            self.text_lr_mul = 0.0

        # ------------------------------------------------------------------
        # 2)   Fusion MLP
        # ------------------------------------------------------------------
        hidden_dim = self.audio_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        mlp_hidden = int(getattr(self.config, "mlp_hidden_size", 512))
        out_dim = int(getattr(self.config, "output_dim", 7))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, out_dim),
        )

        # ------------------------------------------------------------------
        # 3)   Metrics & loss buffers
        # ------------------------------------------------------------------
        self.val_f1 = MulticlassF1Score(num_classes=out_dim, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=out_dim, average="macro")

        alpha = torch.tensor(getattr(self.config, "class_weights", [1.0] * out_dim), dtype=torch.float)
        self.register_buffer("alpha", alpha)

    # ------------------------------------------------------------------
    #  Helper ‑ unfreeze top‑N transformer blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _unfreeze_top_n_layers(layer_list: List[nn.Module], n_layers: int) -> None:
        if n_layers <= 0:
            return
        for layer in layer_list[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        wav_inputs: torch.FloatTensor,
        wav_attention: torch.LongTensor,
        text_inputs: torch.LongTensor,
        text_attention: torch.LongTensor,
    ) -> torch.FloatTensor:
        # Audio CLS (token 0) ---------------------------------------------
        a_out = self.audio_encoder(
            input_values=wav_inputs, attention_mask=wav_attention
        ).last_hidden_state  # (B, L_a, H_a)
        a_emb = a_out[:, 0, :]  # CLS‑like summary vector

        # Text CLS ---------------------------------------------------------
        t_out = self.text_encoder(
            input_ids=text_inputs, attention_mask=text_attention
        ).last_hidden_state  # (B, L_t, H_t)
        t_emb = t_out[:, 0, :]

        # Fusion -----------------------------------------------------------
        fused = torch.cat([a_emb, t_emb], dim=-1)
        logits = self.fusion_mlp(fused)
        return logits  # (B, C)

    # ------------------------------------------------------------------
    #  Lightning steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: list[torch.Tensor], stage: str):
        wav, wav_mask, txt, txt_mask, labels = batch
        logits = self(wav, wav_mask, txt, txt_mask)
        loss = focal_loss(logits, labels, alpha=self.alpha, gamma=getattr(self.config, "focal_gamma", 2.0))
        preds = logits.argmax(dim=-1)

        if stage == "val":
            self.val_f1.update(preds, labels)
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", (preds == labels).float().mean(), prog_bar=True)
        elif stage == "test":
            self.test_f1.update(preds, labels)
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", (preds == labels).float().mean())
        else:  # train
            self.log("train_loss", loss, prog_bar=True)
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
    #  Optimiser & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        base_lr = float(self.config.learning_rate)
        wd = float(getattr(self.config, "weight_decay", 1e-4))

        param_groups: List[dict[str, Any]] = []

        # Audio encoder params (may stay frozen; LR 0 keeps scheduler happy)
        audio_params = [p for p in self.audio_encoder.parameters() if p.requires_grad]
        if audio_params:
            param_groups.append({"params": audio_params, "lr": base_lr * self.audio_lr_mul})

        # Text encoder params
        text_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
        if text_params:
            param_groups.append({"params": text_params, "lr": base_lr * self.text_lr_mul})

        # Fusion head always trains
        param_groups.append({"params": self.fusion_mlp.parameters(), "lr": base_lr})

        optimizer = optim.AdamW(param_groups, weight_decay=wd)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.num_epochs),
            eta_min=float(getattr(self.config, "eta_min", 1e-7)),
        )
        return [optimizer], [scheduler]
