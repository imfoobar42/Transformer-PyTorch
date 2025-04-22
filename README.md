# Transformer-PyTorch
# 🧠 Transformer from Scratch with PyTorch

This repository presents a clear and modular implementation of the Transformer architecture using **PyTorch**, built entirely from first principles. It serves as a pedagogical toolkit for understanding and experimenting with the foundational components of Transformers.

## 🔍 Modules Implemented

- **Positional Encoding** — Enables the model to incorporate sequence order information without recurrence.
- **Scaled Dot-Product Attention** — Computes attention weights using queries, keys, and values.
- **Self-Attention** — Allows each token to attend to every other token in the sequence.
- **Multi-Head Attention** — Runs multiple attention mechanisms in parallel for richer representations.



## 📦 Setup

```bash
git clone https://github.com/your-username/transformer-pytorch.git
cd transformer-pytorch
pip install torch numpy
```

## 🚀 Quick Example

```python
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
import torch

x = torch.rand(4, 10, 512)  # batch_size=4, seq_len=10, d_model=512
x = PositionalEncoding(d_model=512)(x)

mha = MultiHeadAttention(embed_dim=512, num_heads=8)
out = mha(x, x, x)

print(out.shape)  # torch.Size([4, 10, 512])
```

## 📚 References

- Vaswani et al., *Attention Is All You Need* (2017)
- PyTorch Official Documentation
- Jay Alammar's Illustrated Transformer Blog

## 📝 License

This project is released under the MIT License.
