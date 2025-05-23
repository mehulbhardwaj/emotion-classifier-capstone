"""todkat_lite.py
========================================
A *lite* adaptation of TOD‑KAT (Topic‑Driven & Knowledge‑Aware Transformer)
for multimodal emotion recognition on MELD.

This implementation follows the original TOD-KAT approach of sequence-to-sequence
emotion prediction: given utterances x₁, x₂, ..., xₙ, predict emotion labels 
y₁, y₂, ..., yₙ one by one, using only past context at each step.

The model uses:
* **Topic embedding** (default 100‑d, from an integer `topic_id`).
* **Optional commonsense vector** (50‑d) if `use_knowledge: true` in YAML.
* **Causal Transformer encoder** (d_model ≈ 768+768+topic+kn) with masking
  to ensure each position only attends to previous positions.
* **Sequence-to-sequence prediction** for all utterances in dialogue.

All other hyper‑params (batch‑size, LR, scheduler) remain constant, ensuring an
apples‑to‑apples comparison with the MLP baseline and Dialog‑RNN.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel
from torchmetrics.classification import MulticlassF1Score

################################################################################
# focal‑loss helper
################################################################################

def focal_loss(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    alpha: torch.FloatTensor,
    gamma: float = 2.0,
) -> torch.FloatTensor:
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()

################################################################################
# Lite TOD‑KAT model
################################################################################

class TodkatLiteMLP(LightningModule):
    """Topic‑aware + optional knowledge vector + Transformer context."""

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])

        # ----- constants from cfg -----
        self.num_classes: int = int(getattr(config, "output_dim", 7))
        self.topic_dim: int = int(getattr(config, "topic_embedding_dim", 100))
        self.use_knowledge: bool = bool(getattr(config, "use_knowledge", False))
        self.kn_dim: int = int(getattr(config, "knowledge_dim", 16)) if self.use_knowledge else 0
        
        # NEW: SOTA TOD-KAT features
        self.use_topic_mlps: bool = bool(getattr(config, "use_topic_mlps", False))
        self.use_knowledge_attention: bool = bool(getattr(config, "use_knowledge_attention", False))

        # ----- encoders (frozen) -----
        self.audio_encoder: Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_encoder:  RobertaModel  = RobertaModel.from_pretrained("roberta-base")
        for p in self.audio_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters():  p.requires_grad = False

        # optional partial unfreeze
        self.audio_lr_mul = self.text_lr_mul = 0.0
        if hasattr(config, "fine_tune"):
            n_audio = int(getattr(config.fine_tune.audio_encoder, "unfreeze_top_n_layers", 0))
            n_text  = int(getattr(config.fine_tune.text_encoder,  "unfreeze_top_n_layers", 0))
            self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers, n_audio)
            self._unfreeze_top_n_layers(self.text_encoder.encoder.layer,  n_text)
            self.audio_lr_mul = float(getattr(config.fine_tune.audio_encoder, "lr_mul", 1.0))
            self.text_lr_mul = float(getattr(config.fine_tune.text_encoder,  "lr_mul", 1.0))

        # ----- topic & knowledge embeddings -----
        n_topics = int(getattr(config, "n_topics", 50))
        self.topic_emb = nn.Embedding(n_topics, self.topic_dim)

        # ----- SOTA TOD-KAT COMPONENTS -----
        # Topic MLPs (fₙᵤ and fₛ from SOTA paper)
        if self.use_topic_mlps:
            audio_hidden = self.audio_encoder.config.hidden_size
            text_hidden = self.text_encoder.config.hidden_size
            self.topic_mlp_audio = nn.Sequential(
                nn.Linear(audio_hidden, audio_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(audio_hidden, self.topic_dim)
            )
            self.topic_mlp_text = nn.Sequential(
                nn.Linear(text_hidden, text_hidden), 
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(text_hidden, self.topic_dim)
            )
        
        # Knowledge attention mechanism (Bahdanau-style)
        if self.use_knowledge_attention:
            self.knowledge_attention = nn.MultiheadAttention(
                embed_dim=self.kn_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.knowledge_query_proj = nn.Linear(768+768, self.kn_dim)  # [audio;text] -> knowledge query

        # ----- ADD PROJECTION LAYERS TO REDUCE d_model -----
        # Project high-dimensional features to smaller dimensions
        projection_dim = int(getattr(config, "projection_dim", 128))
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, projection_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        
        # final per‑token rep dim entering Transformer (much smaller!)
        self.d_model = (
            projection_dim  # projected audio
            + projection_dim  # projected text  
            + self.topic_dim
            + self.kn_dim
        )

        n_layers = int(getattr(config, "rel_transformer_layers", 2))
        n_heads  = int(getattr(config, "rel_heads", 4))
        dim_feedforward = int(getattr(config, "transformer_dim_feedforward", 256))
        self.rel_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

        # ----- classifier (same depth as baseline) -----
        mlp_hidden = int(getattr(config, "mlp_hidden_size", 512))
        cls_input_dim = (
            projection_dim  # projected audio
            + projection_dim  # projected text
            + self.d_model  # transformer context
        )
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, self.num_classes),
        )

        # ----- metrics / alpha -----
        self.val_f1  = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        
        # Handle class_weights properly - config might have None value
        class_weights = getattr(config, "class_weights", None)
        if class_weights is None:
            class_weights = [1.0] * self.num_classes
        alpha = torch.tensor(class_weights, dtype=torch.float)
        self.register_buffer("alpha", alpha)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unfreeze_top_n_layers(layers: List[nn.Module], n: int):
        for layer in layers[-max(0, n):]:
            for p in layer.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        wav, wav_mask = batch["wav"], batch["wav_mask"]
        txt, txt_mask = batch["txt"], batch["txt_mask"]
        topic_id = batch["topic_id"]
        kn_vec   = batch.get("kn_vec")  # may be None or zeros

        B, T, _ = wav.shape

        # ---- encoders CLS ----
        a_emb = self.audio_encoder(input_values=wav.flatten(0,1), attention_mask=wav_mask.flatten(0,1)).last_hidden_state[:,0,:]
        t_emb = self.text_encoder(input_ids=txt.flatten(0,1), attention_mask=txt_mask.flatten(0,1)).last_hidden_state[:,0,:]
        a_emb = a_emb.view(B, T, -1)
        t_emb = t_emb.view(B, T, -1)

        # ---- PROJECT TO SMALLER DIMENSIONS ----
        a_proj = self.audio_proj(a_emb)  # (B, T, projection_dim)
        t_proj = self.text_proj(t_emb)   # (B, T, projection_dim)

        # ---- ENHANCED TOPIC PROCESSING ----
        if self.use_topic_mlps:
            # Use topic MLPs like SOTA (fₙᵤ, fₛ)
            topic_from_audio = self.topic_mlp_audio(a_emb)  # (B, T, topic_dim)
            topic_from_text = self.topic_mlp_text(t_emb)    # (B, T, topic_dim)
            # Combine: learned topics + lookup topics
            topic_lookup = self.topic_emb(topic_id)
            topic_emb = topic_lookup + topic_from_audio + topic_from_text
        else:
            topic_emb = self.topic_emb(topic_id)  # (B,T,topic_dim)
        
        # ---- ENHANCED KNOWLEDGE PROCESSING ----
        if self.use_knowledge_attention and kn_vec is not None:
            # Use attention mechanism for knowledge (like SOTA)
            query = self.knowledge_query_proj(torch.cat([a_emb, t_emb], dim=-1))  # (B,T,kn_dim)
            attended_knowledge, _ = self.knowledge_attention(
                query, kn_vec, kn_vec  # Q, K, V
            )
            knowledge_features = attended_knowledge
        else:
            knowledge_features = kn_vec if (self.use_knowledge and kn_vec is not None) else None

        # ---- CONCATENATE FEATURES ----
        if self.use_knowledge and knowledge_features is not None:
            x = torch.cat([a_proj, t_proj, topic_emb, knowledge_features], dim=-1)
        else:
            x = torch.cat([a_proj, t_proj, topic_emb], dim=-1)

        # padding mask for TransformerEncoder (True = pad token)
        pad_mask = (~batch["dialog_mask"].bool())
        
        # Create causal mask for autoregressive prediction
        # Each position can only attend to previous positions (including itself)
        seq_len = T
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Apply context encoding with causal masking
        ctx = self.rel_enc(x, src_key_padding_mask=pad_mask, mask=causal_mask)  # (B,T,d_model)

        # Predict emotion for each utterance position using context + original embeddings
        # Concatenate: [audio_emb, text_emb, context] for each position
        fused = torch.cat([a_proj, t_proj, ctx], dim=-1)  # (B,T,cls_input_dim)
        logits = self.classifier(fused)  # (B,T,C)
        return logits

    # ------------------------------------------------------------------
    # lightning steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        # TOD-KAT predicts emotions for all utterances in sequence-to-sequence manner
        mask = batch["dialog_mask"].bool()  # (B, T) - True for valid utterances
        labels = batch["labels"]  # (B, T)
        logits = self(batch)  # (B, T, C)
        
        # Flatten valid utterances only (remove padding)
        logits_flat = logits[mask]  # (N_valid, C)
        labels_flat = labels[mask]  # (N_valid,)
        
        # Check for invalid labels in the flattened valid utterances
        if torch.any(labels_flat < 0) or torch.any(labels_flat >= self.num_classes):
            print(f"⚠️  Invalid labels detected in {stage}:")
            print(f"   Labels range: {labels_flat.min().item()} to {labels_flat.max().item()}")
            print(f"   Expected range: 0 to {self.num_classes-1}")
            print(f"   Invalid labels: {labels_flat[torch.logical_or(labels_flat < 0, labels_flat >= self.num_classes)]}")
            
            # Filter out invalid labels among the valid utterances
            valid_label_mask = torch.logical_and(labels_flat >= 0, labels_flat < self.num_classes)
            if not torch.any(valid_label_mask):
                print("❌ No valid labels in batch, skipping...")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            labels_flat = labels_flat[valid_label_mask]
            logits_flat = logits_flat[valid_label_mask]
            print(f"   Using {valid_label_mask.sum().item()}/{valid_label_mask.shape[0]} valid utterances")
        
        loss = focal_loss(logits_flat, labels_flat, self.alpha, gamma=float(getattr(self.config, "focal_gamma", 2.0)))
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
    # optimiser & scheduler (mirrors baseline)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
            lr_base = float(self.config.learning_rate)
            wd      = float(getattr(self.config, "weight_decay", 1e-4))
    
            groups = []
            if any(p.requires_grad for p in self.audio_encoder.parameters()):
                groups.append({"params": [p for p in self.audio_encoder.parameters() if p.requires_grad],
                               "lr": lr_base * self.audio_lr_mul})
            if any(p.requires_grad for p in self.text_encoder.parameters()):
                groups.append({"params": [p for p in self.text_encoder.parameters() if p.requires_grad],
                               "lr": lr_base * self.text_lr_mul})
    
            # topic + Transformer + classifier always train
            groups.append({"params": list(self.topic_emb.parameters()) +
                                      list(self.rel_enc.parameters()) +
                                      list(self.classifier.parameters()),
                           "lr": lr_base})
    
            optimizer = optim.AdamW(groups, weight_decay=wd)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.config.num_epochs),
                eta_min=float(getattr(self.config, "eta_min", 1e-7)),
            )
            return [optimizer], [scheduler]
