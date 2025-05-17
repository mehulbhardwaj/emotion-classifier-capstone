def setup(self, stage=None):
    if hasattr(self.cfg.paths, "processed_root") and self.cfg.paths.processed_root:
        # Use cached datasets
        self.train_ds = load_from_disk(f"{self.cfg.paths.processed_root}/train")
        self.val_ds = load_from_disk(f"{self.cfg.paths.processed_root}/dev")
        self.test_ds = load_from_disk(f"{self.cfg.paths.processed_root}/test")
    else:
        # Original on-the-fly processing
        # ... existing code ... 