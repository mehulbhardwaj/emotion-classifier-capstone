# utils/sampler.py
class ClassBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, labels):
        class_count = torch.bincount(labels)
        weights = 1. / class_count[labels]
        self.weights = weights / weights.sum()
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
                  self.weights, len(self.weights), replacement=True))
