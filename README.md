# Attention Is All You Need — From Scratch

**Building and Understanding Transformers from the Ground Up**

---

In 2017, the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) revolutionized the field of Natural
Language Processing (NLP) by introducing the Transformer architecture. This model removed the recurrent and
convolutional structures previously dominating sequence-to-sequence tasks, relying entirely on a powerful mechanism
called "attention."

Ever since reading Jay Alammar’s
insightful ["The Illustrated Transformer"](http://jalammar.github.io/illustrated-transformer/) blog, I've been
fascinated by the elegance and effectiveness of attention mechanisms. Driven by curiosity and a desire to deeply
understand the inner workings of Transformers, I decided to implement the entire encoder-decoder Transformer
architecture from scratch using PyTorch.

## What’s in This Repository?

This repository contains a modular, from-scratch implementation of the original Transformer architecture, broken down
for clarity and learning:

### `modules/blocks/`

- `encoder_block.py`: Implements a single Transformer encoder block — includes multi-head self-attention, FFN,
  residuals, and layer norm.
- `decoder_block.py`: Implements a Transformer decoder block — includes masked self-attention, cross-attention, and
  position-wise FFNs.

### `modules/core/`

- `embeddings.py`: Sinusoidal positional encodings and token embeddings.
- `scaled_dot_product_attention.py`: The core attention mechanism (Q · Kᵗ / √d).
- `multi_head_attention.py`: Efficient parallel computation of multiple attention heads.
- `transformer.py`: Combines encoder and decoder into a full TransformerEncoderDecoder model.

### Root Files

- `demo.ipynb`: Demonstrates the full model on a toy sentence like `"hello llms"` to `"start"`.
- `Transformer_module_tests.ipynb`: Unit tests for each module (attention, encoder, decoder, etc.).
- `requirements.txt`: Dependencies to set up your Python environment.

## Why Did I Do This?

This project was purely driven by fun and the urge to learn. Implementing attention from scratch helped me to:

- **Deeply understand** the mathematics and mechanics behind self-attention and cross-attention.
- **Bridge the gap** between theory (papers and blogs) and practical implementation (writing actual code).
- **Strengthen my PyTorch skills**, from tensor manipulation to building modular deep-learning architectures.

I built this Transformer implementation as part of my learning journey .

## References and Inspirations

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

Feel free to explore the code, provide feedback, or fork the repository to experiment further!

