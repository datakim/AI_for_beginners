# Part 7. Generation and Inference

[← Previous part](06-transformers-and-llms.md) · [Contents](README.md) · [Next part →](08-objectives-and-training-methods.md)

- [Q29. What exactly does temperature change?](#q29)
- [Q30. How is top-p different from temperature?](#q30)
- [Q31. What does the KV cache remember to make things faster?](#q31)
- [Q32. Why does a diffusion model predict noise instead of the image?](#q32)

---

<a id="q29"></a>
## Q29. What exactly does temperature change?

For every candidate next token, the model produces a score called a logit. Temperature is a positive number that those scores are divided by before they are turned into probabilities. Lowering it gives high scores an even bigger advantage; raising it shrinks the probability gaps between candidates.

```math
p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}, \qquad T > 0
```

**In plain words:** Divide the score of every candidate token by the temperature T. Turn the divided scores into positive numbers and divide by their total to get new probabilities. When T is small, the gaps between scores widen and more probability piles up on the high-scoring side; when T is large, the gaps shrink and the distribution flattens.

### Reading the symbols

- **zᵢ** (z sub i: the logit of candidate i): The score before it is turned into a probability. It can be negative or larger than 1. It is the value the model computed from the current context.
- **T > 0** (a positive temperature): Here T is the temperature setting, not the number of tokens from [Q28](06-transformers-and-llms.md#q28). Dividing by 0.5 doubles a score; dividing by 2 halves it.
- **e^(zᵢ/T)** (e to the zᵢ over T: divide first, then exponentiate): The whole of zᵢ/T inside the parentheses is the exponent. This is different from computing eᶻ first and then dividing by T.
- **pᵢ(T) and Σⱼ** (p sub i of T, and the sum over j: the probability at that temperature, and the total): The numerator is for candidate i, and the denominator is the sum of the positive scores of all candidates. pᵢ(T) means the probability of candidate i that you get when T is set.

### Step by step

1. With scores [2, 1, 0] and T = 0.5, they become [4, 2, 0]. The ranking of the scores stays the same, but the gap between first and second place grows from 1 to 2.
2. Applying softmax to these values gives about [0.867, 0.117, 0.016]. That is more concentrated on first place than the T = 1 result of [0.665, 0.245, 0.090].
3. In sampling, where a token is drawn according to its probability, this change shifts how often each candidate is picked. It does not modify the model's weights, and it does not flip the ranking of the original logits.

### Following the numbers

If the scores are [2, 1, 0], the probabilities at T = 1 are about [0.665, 0.245, 0.090]. At T = 0.5 they are [0.867, 0.117, 0.016], and at T = 2 they are [0.506, 0.307, 0.186]. At a low T, more probability piles onto first place.

### Where it shows up in models

When a language model samples from these probabilities during inference, temperature controls how random the next token is. If the choice of one token changes, the context that follows changes too, so the whole sentence can end up very different.

### What changes if you tweak it

With the same logits, raising T from 0.5 to 2 moves first place's probability from about 86.7% to 50.6%, and third place's from 1.6% to 18.6%. If you keep drawing from this fixed distribution, lower-ranked candidates come up more often. If you always pick only first place, changing T to another positive value does not, on its own, change the choice.

### Watch out

Plugging T = 0 directly into this formula means dividing by 0. Treating it as an "always pick first place" mode is a separate rule of the system. A low temperature does not guarantee a correct answer either. If the candidate with the highest original score was wrong, probability can pile up even more on that wrong answer.

> **Takeaway:** Temperature does not change what the model has learned; it changes how concentrated the sampling distribution is.

**Check question:** If the ranking of the scores stays the same, why can the actual response still change?

<details>
<summary>Answer</summary>

Because tokens are drawn according to probability, not by rank. The rate at which each candidate is picked changes, and so does the context that follows.

</details>

---

<a id="q30"></a>
## Q30. How is top-p different from temperature?

Temperature changes how concentrated the probabilities are, while top-p limits the set of candidates for sampling by gathering candidates starting from the most probable. The number of candidates needed varies with the context. When the model is confident, only a few may remain; when things are ambiguous, more may remain.

```math
\sum_{i \in S} p_i \ge p_{\text{cut}}, \qquad p'_i = \frac{p_i}{\sum_{j \in S} p_j} \quad (i \in S)
```

**In plain words:** Gather candidates in order, starting from the highest probability. Stop the moment the cumulative probability reaches the threshold or more. Divide the probabilities of the kept candidates by their sum so the total becomes 1, and do not choose any of the other candidates.

### Reading the symbols

- **S and i∈S** (S, and i in S: the set of kept candidates, and a candidate in it): S is a set. ∈ means "belongs to." S is defined as the shortest list of candidates gathered in order of highest probability.
- **Σᵢ∈S pᵢ ≥ p_cut** (the sum of pᵢ over S is at least p_cut: the sum reaches the threshold): ≥ means "greater than or equal to." The candidate that first pushes the sum over the threshold is included too. This inequality alone does not fix the order of selection, so the rule above is also needed.
- **p′ᵢ** (p sub i prime: the probability after the change): Here the prime is not a derivative but a label meaning "the changed value." It is added to tell it apart from the original probability pᵢ.
- **pᵢ / Σⱼ∈S pⱼ** (pᵢ divided by the sum over S: dividing up the remaining probability again): The denominator is the sum of the original probabilities of the kept candidates. Candidates outside S are set to 0, and those inside S are divided by this sum so the total becomes 1 again.

### Step by step

1. The probabilities [0.5, 0.3, 0.15, 0.05] are already sorted from largest to smallest. With a threshold of 0.8, the first candidate alone gives a sum of 0.5, which is not enough.
2. Adding the second brings the sum to 0.8, so we stop. Dividing each of the two remaining candidates' probabilities by 0.8 gives 0.625 and 0.375.
3. Temperature adjusts the gaps between scores. Top-p keeps or drops candidates based on cumulative probability. The two settings perform different calculations.

### Following the numbers

If the probabilities are [0.5, 0.3, 0.15, 0.05] and the threshold is 0.8, the first two candidates are kept. Dividing by their sum, 0.8, gives new probabilities of [0.625, 0.375, 0, 0]. With a threshold of 0.9, three candidates would be needed.

### Where it shows up in models

It is used in LLM response generation to keep candidates with extremely low probability from being chosen. Top-k fixes the number of candidates, while top-p changes that number based on cumulative probability.

### What changes if you tweak it

With the same distribution, raising the threshold from 0.8 to 0.9 keeps candidates up to the third one. Their original probabilities add up to 0.95, and the new probabilities are about [0.526, 0.316, 0.158, 0]. Raising the threshold does not always add one more candidate; the set changes when the threshold crosses a boundary of the cumulative sum.

### Watch out

Once you cut down the candidates, the excluded answers cannot come out at that step. If the correct answer was among them, that chance is gone too. When using top-p together with temperature, first check which distribution top-p is applied to. Detailed rules, such as how ties are handled, can also differ between implementations.

> **Takeaway:** Top-p trims the candidates by cumulative probability, then rescales the remaining candidates' probabilities so they add up to 1 again.

**Check question:** With the same top-p = 0.9, does every context keep the same number of candidates?

<details>
<summary>Answer</summary>

No. The number of candidates depends on how concentrated the probability is.

</details>

---

<a id="q31"></a>
## Q31. What does the KV cache remember to make things faster?

In a causal Transformer, the representations of past tokens can be used as they are even when new tokens are appended after them. If you store the keys and values that each layer has already computed, you do not have to repeat the computation for all the past tokens when producing the next token. This is different from a cache that memorizes answers.

```math
K_{1:t} = [K_{1:t-1};\, k_t], \qquad V_{1:t} = [V_{1:t-1};\, v_t]
```

**In plain words:** Keep the list of keys already computed from the earlier tokens. Append the new token's key to the end, making the list one slot longer. Do the same for the list of values. The next computation uses both the stored past values and the new ones.

### Reading the symbols

- **K₁:ₜ and K₁:ₜ₋₁** (K from 1 to t, and K from 1 to t minus 1: the list up to now and the list up to the previous step): The colon marks a range. 1:t means from the first position to the t-th, and 1:t−1 means up to the position just before. It is not a division sign.
- **kₜ and vₜ** (k sub t and v sub t: the new token's key and value): Read the lowercase letters as the vector for just this one token, and the uppercase letters as a list stacking many tokens.
- **[existing list; new value]** (append to the end): In this formula, the semicolon ; marks concatenation. It does not add the two values slot by slot; it extends the list along the token direction.
- **The two equals signs** (update K and V separately): The key list and the value list are managed separately. In a real Transformer, each layer usually has its own cache.

### Step by step

1. Suppose the K and V for 100 past tokens are stored. While processing a new token, compute k and v for this position.
2. Append them to the end of the existing lists, making each one 101 tokens long. There is less need to recompute K and V for the past 100 tokens.
3. The new query still compares itself with the stored keys and mixes the values. The cache is a device for cutting repeated computation, not a device that removes the attention computation altogether.

### Following the numbers

If the preceding context is 100 tokens and you process one new token, you can add one token's worth without rebuilding K·V for the past 100 tokens. But the attention computation, where the new query looks at 101 keys, does not disappear.

### Where it shows up in models

It is used to speed up autoregressive inference in LLMs. The cache is reused with the same model and the same, still-valid preceding context. Exact caching removes duplicates of mathematically identical computation; approximate compression or quantization is a separate change.

### What changes if you tweak it

With the same model and the same context, using an exact cache or recomputing the same values is intended to give mathematically the same result. If you double the number of stored tokens, the K·V memory also roughly doubles, all else being equal. The longer the context, the larger the storage cost.

### Watch out

If the preceding context or the model weights change, an old cache may no longer be correct. This explanation covers the basic case of causal generation, where past representations do not depend on future tokens. Adding cache compression, quantization, or length limits can change the accuracy or the range the model can look back at.

> **Takeaway:** The KV cache reuses the intermediate computations for past tokens to cut the cost of generation.

**Check question:** Does using a KV cache completely remove the cost of looking at a long context?

<details>
<summary>Answer</summary>

No. Recomputing past K·V is reduced, but the things the new query has to look at and the memory needed to store them still grow.

</details>

_Further reading: [S6 · Hugging Face, *How caching works*](references.md)_

---

<a id="q32"></a>
## Q32. Why does a diffusion model predict noise instead of the image?

If you mix noise that you know into a clean image, you can create a training problem along with its answer. If the model learns to estimate the mixed-in noise at many noise levels, then at generation time it can use those predictions to start from random noise and gradually build up structure.

```math
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad L = \mathbb{E}\!\left[\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\right]
```

**In plain words:** The first formula says: multiply the original by a signal scale, multiply random noise by a noise scale, and add the two. The second formula says: the model looks at that mixed input and the step number and predicts the noise; compare that prediction with the noise actually added, and average the squared error.

### Reading the symbols

- **x₀, xₜ, t** (x zero, x sub t, t: the original, the noisy input, the step): Here t is not a word position but a step that indicates the noise level. You can think of these as image pixels or a compressed representation of the image.
- **ᾱₜ** (alpha bar sub t): A value between 0 and 1 that controls how much signal remains. Here the bar on top is notation for a value accumulated from the coefficients of the earlier steps. Do not read it as a simple average.
- **√ᾱₜ and √(1−ᾱₜ)** (square root of alpha bar t, and square root of 1 minus alpha bar t: the two scales): The numbers that the original and the noise are multiplied by. For example, if ᾱₜ = 0.64, the scales are 0.8 and 0.6. The two scales do not need to add up to 1.
- **ε and ε_θ(xₜ, t)** (epsilon, and epsilon theta of x sub t and t: the actual noise and the predicted noise): Here ε is not a tiny constant but randomly drawn noise. The function with θ attached at the bottom is the model's prediction, and it takes two inputs.
- **𝔼[‖difference‖²]** (the expected value of the squared norm of the difference: the mean of the squared error): ‖difference‖² squares the difference in each slot and adds them up. 𝔼 means the average over many draws of originals, noise, and steps.

### Step by step

1. Multiply one slot of the original, 1, by the scale 0.8. Multiply one slot of the noise, 0.5, by the scale 0.6 and add it, and the input is 1.1.
2. The model takes the input 1.1 and the step t and predicts the noise. If it predicts 0.4, the squared error against the actual 0.5 is 0.01.
3. Because we know the noise we put in, we can build training problems without a person separately attaching answers. By repeating this at many levels, the model learns how to deal with noise.

### Following the numbers

Looking at just one number slot: with x₀ = 1, ᾱₜ = 0.64, and ε = 0.5, xₜ = 0.8 × 1 + 0.6 × 0.5 = 1.1. If the model predicts the noise as 0.4, the squared error for this slot is (0.5 − 0.4)² = 0.01.

### Where it shows up in models

It is used in the standard noise-prediction training of DDPM. During generation, the learned prediction and a fixed reverse update rule are applied over and over. Adding a text condition lets you train the model to generate images that match a description.

### What changes if you tweak it

With the same original and noise, lowering ᾱₜ from 0.64 to 0.25 changes the original's scale from 0.8 to 0.5 and the noise scale from 0.6 to about 0.866. This does not mean the final number in a slot must get bigger; it means the noise takes up a larger share relative to the original signal.

### Watch out

Generation is not a matter of finding one fixed original image and reversing back to it. It starts from noise and creates a new sample by following the distribution the model learned. Real generation updates need step-specific coefficients, so do not read it as "subtract the predicted noise once and you are done." The formula above is one example of the noise-prediction approach.

> **Takeaway:** Using noise it knowingly added as the training signal, the model learns structure across many noise levels.

**Check question:** If you give only the same blurry input and do not tell the model the noise step t, what becomes difficult?

<details>
<summary>Answer</summary>

It can become hard to apply a different restoration rule suited to how damaged the input is.

</details>

_Further reading: [S7 · Ho et al., *Denoising Diffusion Probabilistic Models*](references.md)_

---
