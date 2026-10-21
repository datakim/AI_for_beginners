# References

[← Part 8](08-objectives-and-training-methods.md) · [Contents](README.md)

After reading the plain explanations, you can go back to the original sources to check the formulas and their conditions. The list below is a path for verifying the ideas in this guide and reading further. It does not mean that any single paper presents all of the examples and explanations used here.

| | Source | What to read it for |
|---|---|---|
| S1 | Goodfellow, Bengio, and Courville (2016), [*Deep Learning*](https://www.deeplearningbook.org/) | Foundational math, how learning works, and optimization. Good for going deeper on the basics in Q01–Q20. |
| S2 | Kingma and Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980) | The adaptive update in Q19: the exact formulas for the exponential moving averages and the bias correction. |
| S3 | He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) | Residual connections in Q21: the design of adding a correction to the input, and its effect on training. |
| S4 | [PyTorch `LayerNorm` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) | Q22: which axis gets normalized, and what the learnable scale and bias mean. |
| S5 | Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | Q24–Q27: Q, K, V, scaled dot-product attention, and the multi-head structure. |
| S6 | Hugging Face, [*How caching works*](https://huggingface.co/docs/transformers/cache_explanation) | Q31: autoregressive generation and how keys and values are reused. |
| S7 | Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) | Q32: training by predicting noise, and the reverse generation process. |
| S8 | OpenAI, [*Spinning Up in Deep RL*](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) | Q34–Q35: return, policy gradients, and baselines. |
