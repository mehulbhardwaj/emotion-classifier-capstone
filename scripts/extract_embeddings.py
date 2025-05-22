#!/usr/bin/env python
"""
Extract and cache audio‐ and text‐embeddings from a trained MultimodalFusionMLP checkpoint,
for each split in {train, dev, test}.
"""

import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

# make sure your project root is on PYTHONPATH, or adjust the imports accordingly:
from models.mlp_fusion import MultimodalFusionMLP
from utils.data_processor import MELDDataModule

def parse_args():
    p = argparse.ArgumentParser(description="Extract embeddings from a trained checkpoint")
    p.add_argument("--ckpt",    type=str, required=True, help="path to <your>.ckpt")
    p.add_argument(
        "--config", type=str, default="configs/colab_config.yaml",
        help="path to yaml config (must define data_root, batch_size, etc.)"
    )
    p.add_argument(
        "--splits", nargs="+",
        default=["train", "dev", "test"],
        help="which splits to extract (subset of train/dev/test)"
    )
    p.add_argument(
        "--out_dir", type=str, required=True,
        help="where to write {split}_audio.npy, {split}_text.npy, {split}_labels.npy"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="random seed (for reproducibility)"
    )
    return p.parse_args()

@torch.no_grad()
def extract_split(model, dataloader, device):
    """
    returns (audio_embeddings, text_embeddings, labels) each as a [N x D] numpy array
    """
    aud_list, txt_list, lbl_list = [], [], []
    model.eval()
    for batch in dataloader:
        wav, wav_mask, txt, txt_mask, labels = batch
        wav, wav_mask = wav.to(device), wav_mask.to(device)
        txt, txt_mask = txt.to(device), txt_mask.to(device)

        # forward through each encoder
        a_out = model.audio_encoder(input_values=wav, attention_mask=wav_mask).last_hidden_state
        a_emb = a_out.mean(dim=1)                             # [B, A_hidden]

        t_out = model.text_encoder(input_ids=txt, attention_mask=txt_mask).last_hidden_state
        t_emb = t_out[:, 0, :]                                # [B, T_hidden]

        aud_list.append(a_emb.cpu().numpy())
        txt_list.append(t_emb.cpu().numpy())
        lbl_list.append(labels.numpy())

    audio_feats = np.concatenate(aud_list, axis=0)
    text_feats  = np.concatenate(txt_list, axis=0)
    labels      = np.concatenate(lbl_list, axis=0)
    return audio_feats, text_feats, labels

def main():
    args = parse_args()
    seed_everything(args.seed)

    # load config
    cfg = OmegaConf.load(args.config)

    # instantiate model
    model = MultimodalFusionMLP.load_from_checkpoint(args.ckpt, config=cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # instantiate datamodule
    dm = MELDDataModule(cfg)

    # map split names to DM methods & setup stages
    loaders = {
        "train": dm.train_dataloader,
        "dev":   dm.val_dataloader,
        "test":  dm.test_dataloader
    }

    os.makedirs(args.out_dir, exist_ok=True)

    # first do train+dev under stage="fit"
    if any(s in ["train","dev"] for s in args.splits):
        dm.setup("fit")
        for split in ["train","dev"]:
            if split not in args.splits: continue
            print(f"→ extracting {split} …")
            loader = loaders[split]()
            a, t, y = extract_split(model, loader, device)
            np.save(os.path.join(args.out_dir, f"{split}_audio.npy"), a)
            np.save(os.path.join(args.out_dir, f"{split}_text.npy"),  t)
            np.save(os.path.join(args.out_dir, f"{split}_labels.npy"), y)

    # then test under stage="test"
    if "test" in args.splits:
        dm.setup("test")
        print("→ extracting test …")
        loader = loaders["test"]()
        a, t, y = extract_split(model, loader, device)
        np.save(os.path.join(args.out_dir, "test_audio.npy"),  a)
        np.save(os.path.join(args.out_dir, "test_text.npy"),   t)
        np.save(os.path.join(args.out_dir, "test_labels.npy"), y)

    print("Done. saved to", args.out_dir)

if __name__ == "__main__":
    main()
