"""Dialog‑RNN–style context model for multimodal ERC (MELD).

Encoders  : Wav2Vec 2.0 + DistilRoBERTa  (frozen or partially fine‑tuned)
Context   : Three GRUs tracking global, speaker, and emotion states
Fusion    : [utterance_emb | global | speaker | emotion] → MLP → logits
Loss      : Focal loss with class‑imbalance weights

Expected batch dict keys (shapes in brackets):
    wav            – (B, T, L_a)  raw audio waveforms (float32)
    wav_mask       – (B, T, L_a)  attention mask for audio encoder
    txt            – (B, T, L_t)  token IDs (int64)
    txt_mask       – (B, T, L_t)  text attention mask
    speaker_id     – (B, T)       integer speaker IDs, pad=-1
    dialog_mask    – (B, T)       1 = valid utterance, 0 = pad
    labels         – (B, T)       gold emotion labels (int64)

If you feed isolated utterances (T = 1) the model degrades gracefully and
behaves like the MLP baseline.
"""

from __future__ import annotations

from typing import Any, List
import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel
from torchmetrics.classification import MulticlassF1Score

# ---------------------------------------------------------------------------
# Focal‑loss helper
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    alpha: torch.FloatTensor,
    gamma: float = 2.0,
) -> torch.FloatTensor:
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()

# ---------------------------------------------------------------------------
# Dialog‑RNN model
# ---------------------------------------------------------------------------

class DialogRNNMLP(LightningModule):
    """Speaker‑aware DialogRNN with the same encoders/head as the baseline."""

    # ------------------------------------------------------------------
    #  init
    # ------------------------------------------------------------------
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])

        out_dim = int(getattr(self.config, "output_dim", 7))
        hid_gru = int(getattr(self.config, "gru_hidden_size", 128))

        # Encoders ----------------------------------------------------------------
        self.audio_encoder: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.text_encoder: RobertaModel = RobertaModel.from_pretrained("roberta-base")

        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Optional fine‑tuning -----------------------------------------------------
        if hasattr(self.config, "fine_tune"):
            n_audio = int(getattr(self.config.fine_tune.audio_encoder, "unfreeze_top_n_layers", 0))
            n_text = int(getattr(self.config.fine_tune.text_encoder, "unfreeze_top_n_layers", 0))
            self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers, n_audio)
            self._unfreeze_top_n_layers(self.text_encoder.encoder.layer, n_text)
            self.audio_lr_mul = float(getattr(self.config.fine_tune.audio_encoder, "lr_mul", 1.0))
            self.text_lr_mul = float(getattr(self.config.fine_tune.text_encoder, "lr_mul", 1.0))
        else:
            self.audio_lr_mul = 0.0
            self.text_lr_mul = 0.0

        # Context GRUs -------------------------------------------------------------
        enc_dim = self.audio_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        self.gru_global  = nn.GRU(enc_dim, hid_gru, batch_first=True)
        self.gru_speaker = nn.GRU(enc_dim, hid_gru, batch_first=True)
        self.gru_emotion = nn.GRU(enc_dim, hid_gru, batch_first=True)

        # Classification head ------------------------------------------------------
        mlp_hidden = int(getattr(self.config, "mlp_hidden_size", 512))
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim + 3 * hid_gru, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, out_dim),
        )

        # Metrics & loss buffers ---------------------------------------------------
        self.val_f1  = MulticlassF1Score(num_classes=out_dim, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=out_dim, average="macro")

        alpha = torch.tensor(getattr(self.config, "class_weights", [1.0] * out_dim), dtype=torch.float)
        self.register_buffer("alpha", alpha)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unfreeze_top_n_layers(layer_list: List[nn.Module], n_layers: int) -> None:
        for layer in layer_list[-max(0, n_layers):]:
            for p in layer.parameters():
                p.requires_grad = True

    @staticmethod
    def _mask_by_speaker(x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Organise utterance embeddings so that each speaker has a contiguous sequence.
        The simple trick from the original DialogRNN: duplicate the global sequence
        and zero out positions spoken by other speakers.
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)
        for b in range(B):
            for t in range(T):
                spk = speaker[b, t]
                if spk >= 0:  # skip padded positions
                    out[b, t, :] = x[b, t, :]
        return out

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.FloatTensor:
        wav, wav_mask = batch["wav"], batch["wav_mask"]   # (B,T,L_a)
        txt, txt_mask = batch["txt"], batch["txt_mask"]   # (B,T,L_t)

        B, T, _ = wav.shape
        wav = wav.flatten(0, 1)          # (B*T, L_a)
        wav_mask = wav_mask.flatten(0, 1)
        txt = txt.flatten(0, 1)          # (B*T, L_t)
        txt_mask = txt_mask.flatten(0, 1)

        # Encoders -------------------------------------------------------
        a_out = self.audio_encoder(input_values=wav, attention_mask=wav_mask).last_hidden_state  # (B*T, L_a', H_a)
        a_emb = a_out[:, 0, :]  # CLS‑like summary

        t_out = self.text_encoder(input_ids=txt, attention_mask=txt_mask).last_hidden_state  # (B*T, L_t', H_t)
        t_emb = t_out[:, 0, :]

        u = torch.cat([a_emb, t_emb], dim=-1).view(B, T, -1)  # (B, T, D)

        # Context GRUs ---------------------------------------------------
        global_out, _  = self.gru_global(u)
        speaker_inp = self._mask_by_speaker(u, batch["speaker_id"])
        speaker_out, _ = self.gru_speaker(speaker_inp)
        emotion_out, _ = self.gru_emotion(u)

        h = torch.cat([u, global_out, speaker_out, emotion_out], dim=-1)
        logits = self.classifier(h)  # (B, T, C)
        return logits

    # ------------------------------------------------------------------
    #  Lightning steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str):
        dialog_mask = batch["dialog_mask"]  # (B, T)
        labels = batch["labels"]

        logits = self(batch)  # (B, T, C)

        # Flatten valid positions only
        mask = dialog_mask.bool()
        logits_flat = logits[mask]
        labels_flat = labels[mask]

        loss = focal_loss(logits_flat, labels_flat, alpha=self.alpha, gamma=getattr(self.config, "focal_gamma", 2.0))
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
