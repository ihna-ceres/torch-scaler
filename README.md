# torch-scaler

A scaler similar to the scikit-learn scaler that works on pytorch tensors.

## Installation

```
pip install torch-scaler
```

## How to use

- Currently, we support standard scaler only.

- We support `fit(x: torch.tensor)`, `partial_fit(x: torch.tensor)`, `transform(x: torch.tensor)`, `inverse_transform(x: torch.tensor)` similar to sklearn.preprocessing.StandardScaler.
  - x has to have shape of (batch_size, feature_dim)
- Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- You can see how-to-use example in `test.py`
