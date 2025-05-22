"""MLP Fusion model for emotion classification.

A simplified implementation that fuses audio and text features using a simple MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel


class MultimodalFusionMLP(LightningModule):
    """Wav2Vec2 + RoBERTa → feature concat → MLP classifier.

    Now supports:
    1. Class‑imbalance weights via `config.class_weights`.
    2. Partial fine‑tuning of the last *N* transformer blocks with an LR multiplier.
    3. CosineAnnealingLR scheduler configured from YAML.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])  # saves CLI args automatically
        self.config = config

        # ------------------------------------------------------------------
        # 1) Encoders
        # ------------------------------------------------------------------
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")

        # Freeze all params first
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Unfreeze top‑N layers as per YAML
        self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers,
                                    n_layers=self.config.fine_tune.audio_encoder.unfreeze_top_n_layers)
        self._unfreeze_top_n_layers(self.text_encoder.encoder.layer,
                                    n_layers=self.config.fine_tune.text_encoder.unfreeze_top_n_layers)

        # Store LR multipliers
        self.audio_lr_mul = float(self.config.fine_tune.audio_encoder.lr_mul)
        self.text_lr_mul = float(self.config.fine_tune.text_encoder.lr_mul)

        # ------------------------------------------------------------------
        # 2) Fusion head
        # ------------------------------------------------------------------
        hidden_dim = 768 + 768  # wav2vec2 + roberta hidden sizes
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7)  # 7 emotion classes
        )

        # ------------------------------------------------------------------
        # 3) Loss with class weights
        # ------------------------------------------------------------------
        class_wts = torch.tensor(self.config.class_weights, dtype=torch.float)
        self.register_buffer("class_weights", class_wts)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    # --------------------------- helper methods ------------------------- #
    @staticmethod
    def _unfreeze_top_n_layers(layer_list, n_layers):
        """Unfreeze the last *n_layers* in a list of transformer blocks."""
        if n_layers == 0:
            return
        for layer in layer_list[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ------------------------------ forward ----------------------------- #
    def forward(self, wav_inputs, wav_attention, text_inputs, text_attention):
        # Audio
        a_out = self.audio_encoder(input_values=wav_inputs,
                                   attention_mask=wav_attention).last_hidden_state  # (B, T, 768)
        a_emb = a_out.mean(dim=1)  # simple pooling

        # Text
        t_out = self.text_encoder(input_ids=text_inputs,
                                   attention_mask=text_attention).last_hidden_state
        t_emb = t_out[:, 0, :]  # use CLS token

        # Concatenate
        fused = torch.cat([a_emb, t_emb], dim=-1)
        logits = self.fusion_mlp(fused)
        return logits

    # ----------------------------- training ----------------------------- #
    def training_step(self, batch, batch_idx):
        wav_in, wav_mask, txt_in, txt_mask, labels = batch
        logits = self(wav_in, wav_mask, txt_in, txt_mask)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav_in, wav_mask, txt_in, txt_mask, labels = batch
        logits = self(wav_in, wav_mask, txt_in, txt_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return {"preds": preds, "targets": labels}

    # ------------------------- optimizer setup -------------------------- #
    def configure_optimizers(self):
        base_lr = float(self.config.optimizer.lr)
        wd      = float(self.config.optimizer.weight_decay)

        # Parameter groups with LR multipliers
        audio_params = {
            "params": [p for p in self.audio_encoder.parameters() if p.requires_grad],
            "lr": base_lr * self.audio_lr_mul,
        }
        text_params = {
            "params": [p for p in self.text_encoder.parameters() if p.requires_grad],
            "lr": base_lr * self.text_lr_mul,
        }
        other_params = {
            "params": [p for p in self.fusion_mlp.parameters()],
            "lr": base_lr,
        }

        optimizer = optim.AdamW([
            audio_params,
            text_params,
            other_params,
        ], weight_decay=wd)

        # Cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.scheduler.T_max),
            eta_min=float(self.config.scheduler.eta_min),
        )

        return [optimizer], [scheduler]

