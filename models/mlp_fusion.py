"""MLP Fusion model for emotion classification.

A simplified implementation that fuses audio and text features using a simple MLP.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, RobertaModel
from torchmetrics.classification import MulticlassF1Score

class MultimodalFusionMLP(LightningModule):
    """Wav2Vec2 + RoBERTa → feature concat → MLP classifier.

    Supports:
    1. Class-imbalance weights via `config.class_weights`.
    2. Optional fine-tuning of last *N* transformer blocks via `config.fine_tune`.
    3. CosineAnnealingLR scheduler over epochs.
    4. Adjustable MLP hidden size via config.mlp_hidden_size.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.val_f1 = MulticlassF1Score(num_classes=config.output_dim, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.config.output_dim, average="macro")
        # ------------------------------------------------------------------
        # 1) Encoders
        # ------------------------------------------------------------------
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_encoder  = RobertaModel.from_pretrained("roberta-base")

        # Freeze all encoder params by default
        for p in self.audio_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters():  p.requires_grad = False

        # Optional fine-tuning: unfreeze top-N and set LR multipliers
        if hasattr(self.config, 'fine_tune'):
            n_audio = getattr(self.config.fine_tune.audio_encoder, 'unfreeze_top_n_layers', 0)
            n_text  = getattr(self.config.fine_tune.text_encoder, 'unfreeze_top_n_layers', 0)
            self._unfreeze_top_n_layers(self.audio_encoder.encoder.layers, n_audio)
            self._unfreeze_top_n_layers(self.text_encoder.encoder.layer, n_text)
            self.audio_lr_mul = float(getattr(self.config.fine_tune.audio_encoder, 'lr_mul', 1.0))
            self.text_lr_mul  = float(getattr(self.config.fine_tune.text_encoder, 'lr_mul', 1.0))
        else:
            self.audio_lr_mul = 0.0
            self.text_lr_mul  = 0.0

        # ------------------------------------------------------------------
        # 2) Fusion head
        # ------------------------------------------------------------------
        hidden_dim = self.audio_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        mlp_hidden = getattr(self.config, 'mlp_hidden_size', 512)
        out_dim = len(self.config.class_weights) if hasattr(self.config, 'class_weights') else 7
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
        # 3) Loss with class weights
        # ------------------------------------------------------------------
        if hasattr(self.config, 'class_weights'):
            w = torch.tensor(self.config.class_weights, dtype=torch.float)
            self.register_buffer('class_weights', w)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _unfreeze_top_n_layers(layer_list, n_layers: int):
        """Unfreeze the last *n_layers* in a list of transformer blocks."""
        for layer in layer_list[-max(0, n_layers):]:
            for p in layer.parameters(): p.requires_grad = True

    def forward(self, wav_inputs, wav_attention, text_inputs, text_attention):
        # Audio pooling
        a_out = self.audio_encoder(input_values=wav_inputs, attention_mask=wav_attention).last_hidden_state
        a_emb = a_out.mean(dim=1)

        # Text CLS
        t_out = self.text_encoder(input_ids=text_inputs, attention_mask=text_attention).last_hidden_state
        t_emb = t_out[:, 0, :]

        logits = self.fusion_mlp(torch.cat([a_emb, t_emb], dim=-1))
        return logits

    def training_step(self, batch, batch_idx):
        wav, wav_mask, txt, txt_mask, labels = batch
        logits = self(wav, wav_mask, txt, txt_mask)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav, wav_mask, txt, txt_mask, labels = batch
        logits = self(wav, wav_mask, txt, txt_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        self.val_f1.update(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", (preds == labels).float().mean(), prog_bar=True)
        #self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)
        return {'preds': preds, 'targets': labels}

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        self.log("val_f1", f1, prog_bar=True)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        wav, wav_mask, txt, txt_mask, labels = batch
        logits = self(wav, wav_mask, txt, txt_mask)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc  = (preds == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc",  acc)
        self.test_f1.update(preds, labels)
        return {"preds": preds, "targets": labels}

    def on_test_epoch_end(self):
        f1 = self.test_f1.compute()
        self.log("test_f1", f1)
        self.test_f1.reset()
    
    def configure_optimizers(self):
        # Base LR from config
        base_lr = float(self.config.learning_rate)
        # Weight decay, default if missing
        wd = float(getattr(self.config, 'weight_decay', 1e-4))

        # Build parameter groups
        params = []
        if self.audio_lr_mul > 0:
            params.append({
                'params': [p for p in self.audio_encoder.parameters() if p.requires_grad],
                'lr': base_lr * self.audio_lr_mul,
            })
        if self.text_lr_mul > 0:
            params.append({
                'params': [p for p in self.text_encoder.parameters() if p.requires_grad],
                'lr': base_lr * self.text_lr_mul,
            })
        params.append({
            'params': self.fusion_mlp.parameters(),
            'lr': base_lr,
        })

        optimizer = optim.AdamW(params, weight_decay=wd)

        # Cosine scheduler over epochs
        t_max = int(self.config.num_epochs)
        eta  = float(getattr(self.config, 'eta_min', 1e-7))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta,
        )
        return [optimizer], [scheduler]
