from pytorch_lightning.callbacks import ModelCheckpoint

def get_checkpoint_cb(dirpath: str):
    return ModelCheckpoint(
        dirpath=dirpath,
        filename='{epoch:02d}-{val_wf1:.3f}',
        monitor="val_wf1",
        mode="max",
        save_last=True,
        save_top_k=3,
        every_n_train_steps=500,
    ) 