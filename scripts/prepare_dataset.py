def prepare_cached_dataset(cfg):
    # Load CSVs and WAVs
    # Process audio to mel spectrograms
    # Tokenize text
    # Save as HF datasets
    datasets = {
        'train': train_ds,
        'dev': dev_ds,
        'test': test_ds
    }
    
    cache_dir = Path(cfg.paths.processed_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for split, dataset in datasets.items():
        dataset.save_to_disk(cache_dir / split) 