import torch
import numpy as np


class TorchStandardScaler:
    def __init__(self) -> None:
        self.mean = 0.
        self.std = 0.
        self.n_sampled_seen = 0
        
    def fit(self, x):
        assert not x.isnan().any()
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def partial_fit(self, x: torch.tensor):
        assert not x.isnan().any()
        last_sum = self.mean * self.n_sampled_seen
        new_sum = x.sum(0, keepdim=True)
        new_sample_count = len(x)
        updated_sample_count = self.n_sampled_seen + new_sample_count
        
        self.mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = x - T
        correction = temp.sum(0, keepdim=True)
        temp **=2
        new_unnormalized_variance = temp.sum(0, keepdim=True)
        new_unnormalized_variance -= correction ** 2 / new_sample_count
        last_unnormalized_variance = self.std ** 2 * self.n_sampled_seen
        last_over_new_count = self.n_sampled_seen / new_sample_count
        if last_over_new_count > 0:
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum) ** 2
            )
        else:
            updated_unnormalized_variance = new_unnormalized_variance
        updated_variance = updated_unnormalized_variance / updated_sample_count
        self.std = updated_variance ** 0.5
        self.n_sampled_seen = updated_sample_count

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x
    
    def inverse_transform(self, x):
        x *= (self.std + 1e-7)
        x += self.mean
        return x
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self
    
    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self
