import unittest
from torch_scaler.standard_scaler import TorchStandardScaler
import torch
import numpy as np

class TestTorchScaler(unittest.TestCase):
    def test_torch_standard_scaler(self):
        data = torch.normal(2, 3, size=(100000, 3)).float()
        foo = TorchStandardScaler()
        foo.fit(data)
        # has to be close to 2, 3
        print(foo.mean)
        print(foo.std)
        self.assertTrue(torch.allclose(foo.mean, torch.tensor([2., 2., 2.]).float(), atol=1e-1))
        self.assertTrue(torch.allclose(foo.std, torch.tensor([3., 3., 3.]).float(), atol=1e-1))

        foo1 = TorchStandardScaler()
        for i in range(0, 100000, 2):
            foo1.partial_fit(data[i:i + 2])
        # has to be close to 2, 3
        print(foo1.mean)
        print(foo1.std)
        self.assertTrue(torch.allclose(foo1.mean, torch.tensor([2., 2., 2.]).float(), atol=1e-1))
        self.assertTrue(torch.allclose(foo1.std, torch.tensor([3., 3., 3.]).float(), atol=1e-1))
        new_data = foo.transform(data)
        foo2 = TorchStandardScaler()
        foo2.fit(new_data)
        # has to be close to 0, 1
        print(foo2.mean)
        print(foo2.std)
        self.assertTrue(torch.allclose(foo2.mean, torch.tensor([0., 0., 0.]).float(), atol=1e-1))
        self.assertTrue(torch.allclose(foo2.std, torch.tensor([1., 1., 1.]).float(), atol=1e-1))


if __name__ == "__main__":
    unittest.main()