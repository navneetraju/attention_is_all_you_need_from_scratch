# Attention Is All You Need — From Scratch

**Building and Understanding Transformers from the Ground Up**

---

In 2017, the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) revolutionized the field of Natural Language Processing (NLP) by introducing the Transformer architecture. This model removed the recurrent and convolutional structures previously dominating sequence-to-sequence tasks, relying entirely on a powerful mechanism called "attention."

Ever since reading Jay Alammar’s insightful ["The Illustrated Transformer"](http://jalammar.github.io/illustrated-transformer/) blog, I've been fascinated by the elegance and effectiveness of attention mechanisms. Driven by curiosity and a desire to deeply understand the inner workings of Transformers, I decided to implement the entire encoder-decoder Transformer architecture from scratch using PyTorch.

## What’s in This Repository?

This repository contains my implementation of the original Transformer architecture, including:

- **Token and Positional Embeddings:** To represent words and their positions within sequences.
- **Scaled Dot-Product Attention:** The fundamental building block powering Transformer models.
- **Multi-Head Attention:** Allowing the model to attend to different positions and feature subspaces simultaneously.
- **Encoder and Decoder Blocks:** Complete with residual connections, feed-forward networks, and layer normalization.
- **Full Transformer Encoder-Decoder Model:** Ready for basic experimentation and further exploration.

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

## About Me

Hi, I'm Navneet Raju. I'm passionate about machine learning, deep learning, and exploring cutting-edge architectures. This repository is part of my journey into applied science and engineering roles at leading tech companies. Let's connect and talk about Transformers, NLP, and beyond!

GitHub: [navneetraju](https://github.com/navneetraju)

---

Happy coding and learning!

– Navneet

