# utils/sampler.py
# ---------------------------------------------------------------------
# Two complementary samplers:
#   • DialogueBatchSampler – batches = K full dialogues  (context models)
#   • ClassBalancedSampler – batches = single utterances (baseline MLP)
# ---------------------------------------------------------------------
from __future__ import annotations

import random
from typing import Dict, List, Iterator

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


# ──────────────────────────────────────────────────────────────────────
# 1.  Dialogue-level batching
# ──────────────────────────────────────────────────────────────────────
class DialogueBatchSampler(Sampler[List[int]]):
    """
    Yields a *list of dataset indices* that together form one batch.
    Each batch contains `batch_size` complete dialogues; every utterance
    that shares the same Dialogue_ID stays in the same mini-batch.

    Args
    ----
    dialogue_to_indices : mapping {dialogue_id: [idx0, idx1, …]}
    batch_size          : number of *dialogues* per batch (≠ utterances)
    shuffle             : shuffle dialogues each epoch
    """
    def __init__(
        self,
        dialogue_to_indices: Dict[int, List[int]],
        batch_size: int,
        shuffle: bool = True,
    ):
        super().__init__(None)
        self.dialogue_lists: List[List[int]] = list(dialogue_to_indices.values())
        self.batch_size = batch_size
        self.shuffle = shuffle

    # list[int] → DataLoader will send those utterances to the collate_fn
    def __iter__(self) -> Iterator[List[int]]:
        order = list(range(len(self.dialogue_lists)))
        if self.shuffle:
            random.shuffle(order)

        for i in range(0, len(order), self.batch_size):
            batch_dialogues = [
                self.dialogue_lists[j] for j in order[i : i + self.batch_size]
            ]
            # Flatten utterance indices from all selected dialogues
            yield [utt_idx for dial in batch_dialogues for utt_idx in dial]

    def __len__(self) -> int:
        return (len(self.dialogue_lists) + self.batch_size - 1) // self.batch_size


# ──────────────────────────────────────────────────────────────────────
# 2.  Utterance-level class-balanced sampling
# ──────────────────────────────────────────────────────────────────────
class ClassBalancedSampler(WeightedRandomSampler):
    """
    Classic inverse-frequency sampler for *utterance*-level training.
    Extends `torch.utils.data.WeightedRandomSampler` so it can be passed
    straight to a DataLoader.
    """
    def __init__(self, labels: torch.Tensor):
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)

        class_count  = torch.bincount(labels)
        class_weight = 1.0 / class_count.float().clamp(min=1)

        # weight for each sample = weight of its class
        sample_weights = class_weight[labels]

        # We call the parent constructor → it handles __iter__ / __len__
        super().__init__(weights=sample_weights,
                         num_samples=len(sample_weights),
                         replacement=True)
