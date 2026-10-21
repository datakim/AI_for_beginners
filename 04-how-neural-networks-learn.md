# Part 4. How Neural Networks Learn

[← Previous part](03-classical-ml-and-evaluation.md) · [Contents](README.md) · [Next part →](05-when-training-works-or-breaks.md)

- [Q14. Why do we need activation functions when stacking layers?](#q14)
- [Q15. How does backpropagation work out each parameter's influence?](#q15)
- [Q16. Why does picking exactly one option cut off the learning signal?](#q16)
- [Q17. What else does a neural network learn besides the answer?](#q17)

---

<a id="q14"></a>
## Q14. Why do we need activation functions when stacking layers?

If you only do matrix multiplication and addition over and over, it can all be combined into a single matrix multiplication and addition in the end. Putting a nonlinear transformation in between breaks this limit. The best-known one, ReLU, turns negative values into 0 and lets positive values pass through unchanged.

```math
h = \max(0,\, wx + b), \qquad \hat{y} = vh + c
```

**In plain words:** First multiply the input by a weight and add a base value. Compare that result with 0 and let only the larger one through. Multiply the resulting intermediate value h by the next layer's weight and add a base value to produce the final prediction.

### Reading the symbols

- **wx+b** ("w x plus b"): The first layer's calculation. w is the value multiplied by the input, and b is the value added. Both are parameters that can be learned.
- **max(0, wx+b)** ("max of zero and w x plus b"): The larger of the two values. max is a function that picks the largest value. If the first calculation is positive, it passes through as is. If it is negative, it becomes 0. This is ReLU.
- **h**: The intermediate value. It is the name given to the first layer's result so it can be passed to the next layer. It is not new input data but a feature made by calculation.
- **vh+c** ("v h plus c"): The next layer's calculation. v and c are the next layer's parameters. This example uses single numbers, but with many slots it extends to a matrix calculation.

### Step by step

1. Let w = 2, b = 1, v = 3, c = 0. If x = −1, the first calculation gives −1.
2. Since max(0, −1) = 0, h = 0. The next layer outputs 3×0 = 0. If x = 1, then h = 3 and the prediction is 9.
3. As the input changes, the rule for what passes through is different in different ranges, so the relationship gets a bend. This is what sets it apart from simply chaining straight-line calculations together.

### Following the numbers

If the first calculation is 2x+1 and the next one takes 3 times that result, then without an activation the whole thing is 6x+3. With ReLU, at x = −1 the first result −1 becomes 0, and the final value is also 0. At x = 1, the 3 passes through and becomes 9. The relationship now has a bend.

### Where it shows up in models

It is used in the hidden layers of neural networks. The feed-forward part of a Transformer also uses nonlinear transformations such as GELU or SiLU. They provide the expressive power needed to transform features step by step as layers are stacked.

### What changes if you tweak it

Take ReLU out and the result is 3(2x+1) = 6x+3, a single straight line. With ReLU, the result at x = −1 is 0 instead of −3. Because of the operation that cuts off the negative range, the layers do not merge into one straight line.

### Watch out

In ReLU's negative range, the rate of change is also 0. If a neuron's input always sits in that range, no learning signal may get through along that path. This is a property of ReLU. Not every activation function cuts off negative values the same way.

> **Takeaway:** Only with a nonlinear transformation can multiple layers go beyond what a single linear layer can do.

**Check question:** If you stack 100 layers that only compute weighted sums, can they represent a complex curve?

<details>
<summary>Answer</summary>

If there is no other nonlinear operation in between, the whole thing merges into a single affine transformation. Just adding more layers does not solve this.

</details>

---

<a id="q15"></a>
## Q15. How does backpropagation work out each parameter's influence?

The final loss is produced through many calculations. Changing an earlier parameter a little changes an intermediate value, and that change in turn affects the loss. The chain rule is the rule for multiplying the rates of change along this path, and backpropagation is the method that applies it efficiently, starting from the end.

```math
\frac{dL}{dw} = \frac{dL}{dh}\,\frac{dh}{dw}
```

**In plain words:** If w changes a little, the intermediate value h changes, and if h changes, the loss L changes. Multiply "the rate at which w changes h" by "the rate at which h changes L," and you get the total rate at which w changes the loss.

### Reading the symbols

- **w → h → L**: The chain of calculations. w is the parameter, h is an intermediate value made from that parameter, and L is the final loss. Values are computed in the direction of the arrows.
- **dh/dw** ("the derivative of h with respect to w"): Shows how much h goes up or down when you increase w a little.
- **dL/dh** ("the derivative of L with respect to h"): Shows how much the loss changes when you increase h a little.
- **dL/dw** ("d L d w"): The final rate of change we want to know. You get it by multiplying the two rates of change on the right. This linking rule is called the chain rule. Don't memorize it as canceling the d symbols like ordinary numbers.

### Step by step

1. If h = 2w at the current point, then dh/dw = 2. If w increases by 0.01, h increases by 0.02.
2. If dL/dh = −4 at that point, an increase of 0.02 in h reduces the loss by roughly 0.08. So the rate with respect to w is −4×2 = −8.
3. Backpropagation works backward through the calculation from the loss side and links these influences together. If there are several paths from one parameter to the loss, you add up the influence of each path.

### Following the numbers

Let h = 2w and L = (h−6)². If w = 2, then h = 4. Here dL/dh = 2(h−6) = −4 and dh/dw = 2, so dL/dw = −8. So nudging w up a little is the direction that reduces the current loss.

### Where it shows up in models

In neural network training, backward computes the gradients, and the optimizer uses those gradients to update the parameters. The two have different roles. Automatic differentiation tools handle this process by recording the derivative rules of the operations used and how the calculations connect.

### What changes if you tweak it

Change one path's intermediate rate of change from 2 → 0, and the gradient through that path goes from −4×2 = −8 to −4×0 = 0. Even if an output value exists, if small input changes along that path do not reach the output, the signal for fixing the earlier parameters disappears.

### Watch out

Backpropagation is the process of computing gradients. The actual parameter update is done by the optimizer. Also, this gradient is the sensitivity of the current calculation. It is not a causal analysis proving that the parameter is the real-world cause of a wrong answer.

> **Takeaway:** Backpropagation is a way of tracing back along the calculation path and collecting each parameter's rate of change.

**Check question:** What happens if you only compute the gradients and don't run the optimizer's update step?

<details>
<summary>Answer</summary>

In general, the parameters do not change. Keep backward and update separate in your head.

</details>

---

<a id="q16"></a>
## Q16. Why does picking exactly one option cut off the learning signal?

argmax, which picks only the highest-scoring item, often keeps the same choice even when the scores change a little. Since the selected result does not change, the derivative through that choice is 0 almost everywhere. At the boundary, the choice suddenly flips, so it usually cannot be differentiated there.

```math
\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \qquad o = \sum_i \alpha_i v_i
```

**In plain words:** Turn each candidate's score into a positive number. Divide each positive number by the total to get a weight. Multiply each candidate's value by its weight and add them all up. Then, when a score changes a little, the mixing ratio can also change a little.

### Reading the symbols

- **sᵢ and eˢⁱ** ("s sub i" and "e to the s sub i"): The score, and the score turned positive. sᵢ is candidate i's original score. Raising e, which is about 2.718, to the power of that score makes even a negative score positive.
- **Σⱼ in the denominator** ("sum over j"): The sum of all the candidates' positive scores. You add while letting j run over every candidate. The i in the numerator is the candidate whose weight you are computing now, and the j in the denominator is the index that sweeps over all candidates.
- **αᵢ** ("alpha sub i"): The candidate's weight. It is its own share divided by the total. All the αᵢ add up to 1. This score → weight conversion is softmax.
- **vᵢ and o** ("v sub i" and "o"): The value to fetch, and the mixed result. αᵢvᵢ is the portion of value i, scaled by its weight. Σᵢ adds up the portions of all candidates.

### Step by step

1. Exponentiating the scores [2, 1] gives roughly [7.389, 2.718]. The sum is 10.107.
2. Dividing each by the sum gives weights of about [0.731, 0.269]. Mixing the candidate values [10, 0] gives 0.731×10 + 0.269×0 ≈ 7.31.
3. Raising a score slightly changes the weights too, which can affect the result. In contrast, if you pick only the top one, the selected result stays the same until the ranking flips, so the rate of change along that selection path is 0.

### Following the numbers

With scores [2, 1], if you pick only the first candidate, the result stays the same even when the first score becomes 2.01. Softmax gives about [0.731, 0.269], and with values [10, 0] the output is about 7.31. Raise a score a little and the weights and the output change a little too.

### Where it shows up in models

This property matters for learning the scores in attention. The gradient can be passed from the loss on the mixed output all the way back to the scores. However, even with argmax, a separate gradient path can remain for the parameters that produce the selected value itself.

### What changes if you tweak it

Raise only the first score from 2 → 2.01, and the softmax weight moves from about 0.731 → 0.733, and the output from about 7.31 → 7.33. The output of the pick-only-the-top method stays at 10. The difference is whether a small change in the score shows up in the result.

### Watch out

Mixing smoothly does not mean the gradient is always large enough. If the candidate values are the same, or the weights pile up at an extreme, the useful signal can be weak. Discrete choices can also be learned with separate estimation methods such as policy gradients, so this does not mean "if it can't be differentiated, no learning is possible at all."

> **Takeaway:** A soft choice lets small changes in the scores turn into changes in the output.

**Check question:** Is using argmax at inference time always a problem too?

<details>
<summary>Answer</summary>

No. At inference time, you may not need to learn the selection scores through backpropagation. Keep the training path and the usage stage separate.

</details>

---

<a id="q17"></a>
## Q17. What else does a neural network learn besides the answer?

Intermediate layers turn the input into a form that the next calculation can handle more easily. This is called representation learning. With images, the calculation that looks for small patterns at many positions can be shared. Convolution in a CNN applies the same filter while shifting it from position to position.

```math
h_t = \sigma\!\left(\sum_{j=0}^{k-1} a_j x_{t+j} + b\right)
```

**In plain words:** Take a few input slots near the current position. Multiply each slot by the filter's number, add them up, and add a base value. Pass that result through an activation function to make hₜ, the new feature at this position.

### Reading the symbols

- **t and xₜ₊ⱼ** ("t" and "x sub t plus j"): The current position and the nearby inputs. t is the position you are looking at. j = 0 means xₜ, and j = 1 means the very next slot, xₜ₊₁.
- **k and Σⱼ₌₀ᵏ⁻¹** ("k" and "sum from j = 0 to k − 1"): The range to look at, and the sum. You look at k slots. Since numbering starts from 0, the last index is k−1. If k = 2, you use j = 0, 1.
- **aⱼ** ("a sub j"): The filter's j-th weight. It decides how much to add or subtract each slot of the nearby input. The same filter is reused at many positions.
- **b and σ( )** ("b" and "sigma of"): The base value and the activation. Add b to the sum, then apply σ. Here σ is a general name for an activation function. Unlike in [Q09](03-classical-ml-and-evaluation.md#q09), it does not always mean sigmoid.
- **hₜ** ("h sub t"): The output feature at position t. It is the value obtained by calculating over the neighborhood of position t. It is not limited to features that people labeled in advance. It can be a feature that became useful through training.

### Step by step

1. Take the filter [1, −1] and the input [3, 3, 8]. At the first position, read the first two slots to get 1×3 − 1×3 = 0.
2. Shift one slot to the next position and use the same filter: 1×3 − 1×8 = −5. With b = 0, these are the values just before they go into the activation.
3. The filter [1, −1] responds to the difference between neighboring slots. Actual training changes these filter numbers too, and builds the representations it needs with many filters and layers.

### Following the numbers

Set the filter to [1, −1], the input to [3, 3, 8], and the bias to 0. The values before activation are 3−3 = 0 at the first position and 3−8 = −5 at the second. That is a signal that there is a big difference in the second stretch. The final output depends on which activation you apply to these values.

### Where it shows up in models

CNNs capture local patterns in images and time series. Transformers also learn intermediate representations, but the structure they use to mix information is different. Transfer learning, which reuses learned intermediate vectors for other tasks, is also possible.

### What changes if you tweak it

Change the filter from [1, −1] to [0.5, 0.5], and the values before activation at the two positions go from [0, −5] to [3, 5.5]. A calculation that looked for differences has turned into one that averages the neighbors.

### Watch out

If ReLU comes afterward, −5 becomes 0, so you have to tell apart the calculation that "responded to a difference" and the final output. It is also not true that a single real neuron always corresponds to one meaning, such as an edge, an eye, or grammar. A meaning can be spread across many numbers.

> **Takeaway:** While a neural network produces answers, it also learns intermediate representations that are useful for those answers.

**Check question:** What would you lose if a CNN learned a separate filter for every position?

<details>
<summary>Answer</summary>

You lose the weight-sharing effect of finding the same pattern at different positions, and the number of parameters needed goes up.

</details>

---
