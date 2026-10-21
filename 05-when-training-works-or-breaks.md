# Part 5. Why Training Works or Breaks

[← Previous part](04-how-neural-networks-learn.md) · [Contents](README.md) · [Next part →](06-transformers-and-llms.md)

- [Q18. Why can a large learning rate make the loss go up?](#q18)
- [Q19. Why learn in small batches, and why use Adam?](#q19)
- [Q20. Why does the training score improve while the real-world score gets worse?](#q20)
- [Q21. Why do residual connections help deep models learn?](#q21)
- [Q22. What does LayerNorm keep steady?](#q22)

---

<a id="q18"></a>
## Q18. Why can a large learning rate make the loss go up?

The gradient tells you the direction near the current point. Gradient descent moves the opposite way, but if it goes too far, it can overshoot the low point and land on high ground on the other side. The learning rate is the multiplier applied to the gradient, and the actual step is the product of the two.

```math
\mathbf{w}_{\text{new}} = \mathbf{w} - \eta\,\nabla L(\mathbf{w})
```

**In plain words:** For each parameter, find the gradient, which points toward where the loss increases. Multiply that gradient by the learning rate to set the size of the move. Subtract that amount from the current parameters, so you move in the direction where the loss goes down within a nearby range.

### Reading the symbols

- $\mathbf{w}$ and $\mathbf{w}_{\text{new}}$ (w and w-new: the current values and the new values): The bold w is a vector that bundles many parameters together. The subscript "new" means the value after the update.
- **L(w)** ("L of w", the loss at the current parameters): The parentheses show which parameter settings the loss was computed with.
- **∇L(w)** ("gradient of L at w", the bundle of gradients at the current position): Each entry is the rate at which the loss changes when you raise only that parameter a little. It is computed with the other parameters held fixed.
- **−η∇L** ("minus eta times the gradient of L", the change that actually gets added): η is a positive number. If the gradient is positive, the value goes down; if it is negative, you subtract a negative number, so the value goes up. In both cases, the aim is to reduce the loss within a small range.

### Step by step

1. With L = (w − 2)² and w = 3, the loss is 1 and the gradient is 2. First, compute the gradient at this current position.
2. Multiplying by the learning rate η = 0.1 gives an update of size 0.2. Subtracting it from the current value gives $w_{\text{new}} = 3 - 0.2 = 2.8$.
3. Compute the new loss, (2.8 − 2)² = 0.64, and compare it with the previous 1. Computing the gradient and checking the result after the actual move are separate steps.

### Following the numbers

With L = (w − 2)² and w = 3, the gradient is 2. With η = 0.1, the new value is 2.8 and the loss drops to 0.64. With η = 1, the new value is 1 and the loss stays at 1. With η = 1.5, the new value is 0 and the loss grows to 4.

### Where it shows up in models

This is the basic principle for training neural networks. In practice, it is often used together with an optimizer such as Adam and a learning rate schedule. Warmup, which raises the learning rate gradually at the start, is used to reduce the instability caused by large updates in the early phase.

### What changes if you tweak it

From the same starting point w = 3: with η = 0.1, $w_{\text{new}} = 2.8$ and the loss is 0.64. With η = 1, $w_{\text{new}} = 1$ and the loss is 1; with η = 1.5, $w_{\text{new}} = 0$ and the loss is 4. Even if the initial direction is right, going too far can overshoot the good point and make things worse.

### Watch out

A smaller learning rate shrinks each move, but the model may not learn enough within a fixed amount of time. A minibatch gradient can differ from the gradient over the full data, so even a small learning rate does not guarantee that the overall loss goes down at every step.

> **Takeaway:** Knowing a good direction is not the same as knowing a safe distance to move.

**Check question:** If the gradient doubles and the learning rate stays the same, what happens to the size of the move in plain gradient descent?

<details>
<summary>Answer</summary>

It doubles. If you only memorize the learning rate as "the actual step size itself," this is easy to miss.

</details>

---

<a id="q19"></a>
## Q19. Why learn in small batches, and why use Adam?

Computing the gradient over the full data every time is expensive in computation and memory. A minibatch estimates the average gradient from a subset of examples, so the model can be updated more often. Adam is a method that keeps separate running records of the direction and the size of recent gradients, and uses them to adjust the update for each parameter.

```math
g_B = \frac{1}{|B|}\sum_{i \in B} \nabla \ell_i, \qquad w_{t+1} = w_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

**In plain words:** The first formula says: average the gradients of the examples picked this time. The second says: take the direction information of recent gradients, scale it according to the size of recent gradients, then subtract it from the parameter. Adam decides how far to move while also remembering past gradients.

### Reading the symbols

- **B, |B|, i ∈ B** ("i in B": the batch and the examples in it): B is the set of examples looked at together this time. |B| is how many there are, and ∈ means "belongs to that set." The vertical bars here are not a sign to take an absolute value.
- $g_B$ and **∇ℓᵢ** ("g sub B" and "gradient of ell i": the batch gradient and the per-example gradient): Add up the gradients of each example's loss ℓᵢ and divide by the number of examples to get the average. An actual implementation may instead compute the average loss first and then differentiate it.
- **wₜ, wₜ₊₁** ("w sub t" and "w sub t plus 1": the value at this step and at the next step): Here t is the number of the training update. You can read the right-hand formula as the update of a single parameter entry.
- **m̂ₜ** and **v̂ₜ** ("m-hat t" and "v-hat t": two corrected averages): m̂ is an average of recent gradients, so it keeps their sign. v̂ is an average of the squared recent gradients. Both give more weight to recent values than to old ones.
- **√v̂ₜ + ε** ("square root of v-hat t, plus epsilon", the denominator that adjusts the size): The square root turns the squared size back into the original scale, and that is used as the denominator. ε is a small positive number that prevents division by zero. The hat ^ here marks that the bias of the averages in the early steps has been corrected.

### Step by step

1. If the two example gradients in this batch are 2 and 6, then $g_B = (2 + 6)/2 = 4$. This gradient is used to update the averages that Adam keeps in memory.
2. Suppose that after the update, m̂ = 2 and v̂ = 4. These values are not determined by the 4 above alone; they are a separate example that also reflects past history.
3. Ignoring ε, the ratio is 2/√4 = 1. Multiply by η = 0.1 and subtract 0.1 from the current parameter. The formula above does not show every detail of how the averages are updated.

### Following the numbers

In the batch gradient example, if the two examples have gradients 2 and 6, the average is 4. As a separate Adam update example, assume m̂ = 2 and v̂ = 4 (which already reflect past history) and η = 0.1. Ignoring ε, the move is 0.1 × 2/2 = 0.1.

### Where it shows up in models

This is used to train large neural networks. A minibatch is a way of splitting up the computation, and Adam is an update rule, so the two can be used together. The actual values also reflect the history built up over previous batches.

### What changes if you tweak it

Keep m̂ = 2 and η = 0.1 fixed and change only v̂ from 4 to 16, and the amount subtracted shrinks from 0.1 to 0.05. The denominator gets bigger, so the move gets smaller. In real training, the two averages change together, so you should not assume that only one of them grows independently.

### Watch out

A bigger batch usually averages over more examples, which can reduce the sampling variation in the gradient, but it also increases the memory and compute load. A large batch or Adam does not always improve final performance. Adam still requires you to choose a base learning rate.

> **Takeaway:** A batch is the unit for estimating the gradient, and Adam is a way of moving using that gradient.

**Check question:** If you only double the batch size, does the average gradient necessarily double too?

<details>
<summary>Answer</summary>

No. If you use the average rather than the sum, the size does not automatically double.

</details>

_Further reading: [S2 · Kingma and Ba, *Adam*](references.md)_

---

<a id="q20"></a>
## Q20. Why does the training score improve while the real-world score gets worse?

If a model fits even the accidental patterns that exist only in the training examples, it may not fit new data well. This is overfitting. Regularization is a way of constraining the training objective so that it prefers certain kinds of complexity or sensitivity less.

```math
L_{\text{total}} = L_{\text{data}} + \lambda \sum_j w_j^2
```

**In plain words:** Add a penalty for large weights to the penalty for wrong predictions. The weight penalty squares each value, adds them up, and counts the result in proportion to λ. Training proceeds to reduce the total loss, which combines the two penalties.

### Reading the symbols

- $L_{\text{total}}$ and $L_{\text{data}}$ ("L total" and "L data": the total loss and the data loss): The data loss is the value from comparing predictions with the correct answers. "total" is the name for the overall score with the regularization added on top.
- **wⱼ** and **Σⱼ wⱼ²** ("w j" and "sum over j of w j squared": the sum of squared weights): j is the index of a weight being regularized. Because each value is squared before adding, large weights get a large penalty regardless of their sign.
- **λ** (lambda, the strength of regularization): A setting that is 0 or greater. At 0, this regularization has no effect; the larger it gets, the more strongly the size of the weights is penalized.
- **+** (plus, combining two goals): The goal of predicting well and the goal of keeping the weights small are evaluated together as a single score.

### Step by step

1. First, compute the prediction loss. Then square all the weights subject to regularization and add them up.
2. Multiply the sum of squared weights by λ and add it to the first loss. With a data loss of 1, a sum of squares of 100, and λ = 0.1, you get 1 + 10 = 11.
3. The model may choose different settings than it would if it minimized the data loss alone. This is the channel through which L2 regularization affects training.

### Following the numbers

Suppose A has a data loss of 1 and a sum of squared weights of 100, and B has a loss of 2 and a sum of squares of 4. With λ = 0.1, the total loss is 11 for A and 2.4 for B. Regularization can lead to picking different settings, even if they fit the training data a little less well.

### Where it shows up in models

This is used in training regression models and neural networks. Early stopping (stopping early based on validation performance), dropout (randomly switching off some activations), and data augmentation that preserves meaning are also ways to reduce overfitting.

### What changes if you tweak it

Say A has a data loss of 1 and a sum of squared weights of 100, and B has a loss of 2 and a sum of squares of 4. With λ = 0, A comes out ahead. With λ = 0.1, the total losses become A = 11 and B = 2.4, so B comes out ahead. The strength of regularization changes which model settings are preferred.

### Watch out

Making the weights small does not necessarily improve performance on new data. If regularization is too strong, the model cannot learn even the relationships it needs. Choose the strength using validation performance. Also, weight decay in AdamW is generally not the same as adding an L2 term to the loss under Adam.

> **Takeaway:** Fitting the training data better is no guarantee of fitting new data better.

**Check question:** If you keep making regularization stronger, will test performance keep improving?

<details>
<summary>Answer</summary>

No. Past a certain point, the model may fail to learn the patterns it needs and get worse. You have to check with validation data.

</details>

---

<a id="q21"></a>
## Q21. Why do residual connections help deep models learn?

With many layers, the path along which rates of change get multiplied together becomes long. Keep multiplying small numbers and the signal weakens; multiply large numbers and it can grow. A residual connection adds the input directly to the output, so the new transformation learns a correction, and the signal also gets a direct path to pass through.

```math
y = x + F(x), \qquad \frac{dy}{dx} = 1 + F'(x)
```

**In plain words:** Take F(x), the transformed input, and add the original input x to it unchanged. So the output becomes "original value + correction." The overall rate of change when you nudge the input is likewise the sum of 1 from the pass-through path and the rate of change of the transformation path.

### Reading the symbols

- **F(x)** ("F of x", the value of F with x plugged in): The result of the transformation computed by several layers. In a residual connection, it plays the role of the correction added to the original input.
- **y = x + F(x)** (the sum of two paths): Adds the path where the input comes through unchanged and the path that goes through the transformation. The two values must have shapes and sizes that can be added.
- **dy/dx** ("d y over d x", the rate of change of the output with respect to the input): This is the derivative for the simple case where each is a single number. With real vectors, you deal with rates of change between many entries.
- **1 + F′(x)** (the rates of change of the direct path and the transformation path): F′ is read "F prime" and means the derivative of F. The rate of change of x itself is 1, so adding the two paths gives this expression.

### Step by step

1. If x = 10 and the transformation gives F(x) = 2, the output is 12. The transformation can add just a correction instead of having to rebuild the whole input from scratch.
2. If F′(x) = 0.1 at the moment, then when x goes up by 0.01, the direct path adds 0.01 and the transformation path adds about 0.001.
3. Combining the two changes, the output increases by about 0.011. Dividing by the input change of 0.01 gives an overall rate of change of 1.1.

### Following the numbers

If x = 10 and F(x) = 2, the output is 12. If F′(x) = 0.1 right now, the overall rate of change is 1.1. By contrast, passing through five layers that each have a rate of change of 0.1, with no direct path, gives a product of 0.00001.

### Where it shows up in models

Residual connections are used in the deep blocks of ResNet and the Transformer. Instead of having a block build all the information anew from scratch, they let the block add the needed change to the existing representation. This helps make optimization easier.

### What changes if you tweak it

Remove the direct path x, and the overall rate of change in the example above drops from 1.1 to 0.1. Pass through only five transformations in a row, each with a rate of change of 0.1, and the product is 0.00001. A residual connection provides a direct path for passing things through, in addition to the path where only the transformations get multiplied one after another.

### Watch out

This does not mean the gradient is always preserved. If F′ = −1, then 1 + F′ = 0. Real models also have many dimensions and other operations, so this scalar example alone cannot guarantee stable training.

> **Takeaway:** A residual connection is a path that preserves the input while adding the needed correction.

**Check question:** If F outputs 0 for every input, what does the residual block do?

<details>
<summary>Answer</summary>

It passes the input through unchanged. It is a structure that can preserve the input when no new transformation is needed.

</details>

_Further reading: [S3 · He et al., *Deep Residual Learning*](references.md)_

---

<a id="q22"></a>
## Q22. What does LayerNorm keep steady?

If the size of the values in intermediate representations swings wildly, the later computations and training can become difficult. LayerNorm readjusts the values using the mean and spread along a specified feature axis. In Transformers, it is usually computed within the feature vector of a single token.

```math
y_j = \gamma_j\,\frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_j
```

**In plain words:** From one feature entry, subtract the mean of that group of features. Divide how far it is from the mean by the size of the group's spread. Finally, multiply by a learned scale γ and add a shift β to make the final output.

### Reading the symbols

- **xⱼ** and **yⱼ** ("x j" and "y j": the j-th input and output): j is the index of a feature entry within the same group. In Transformers, the mean and spread are usually computed within the feature vector of a single token.
- **μ** (mu, the mean): Add up the values of the entries being normalized and divide by the number of entries. xⱼ − μ is how far above or below the mean that entry is.
- **σ²** (sigma squared, the variance): The average of the squared differences between each entry and the mean. √σ² is the standard deviation, which expresses the size of the spread in the same units as the original values.
- **ε** (epsilon): A small positive number added so that the denominator does not become 0 even when the variance is 0. In this formula, it is added first and then the square root is taken, as in √(σ² + ε).
- **γⱼ, βⱼ** (gamma j, beta j): Learned parameters that are multiplied with and added to each entry after normalization. After the size has first been tidied up, they let the model learn back the scale and reference point it needs.

### Step by step

1. The mean μ of [2, 4] is 3. Subtracting the mean gives [−1, 1].
2. Squaring the differences and averaging them gives a variance of (1 + 1)/2 = 1. Ignoring ε for the moment and dividing by the standard deviation of 1 gives [−1, 1].
3. If γ = [1, 1] and β = [0, 0], that is the final output. If γ or β is changed through learning, the mean and spread of the final output can change again too.

### Following the numbers

If x = [2, 4], the mean is 3 and the variance is 1. Ignoring ε, with γ = [1, 1] and β = [0, 0], the result is [−1, 1]. Even if x is [20, 40], the same conditions give the same result.

### Where it shows up in models

LayerNorm is used in each Transformer block to handle intermediate representations. The mean and variance are computed from the current input, while γ and β are learned. RMSNorm is a different method that does not subtract the mean, so you should not treat the two names as the same thing.

### What changes if you tweak it

Even if you scale the input up tenfold, from [2, 4] to [20, 40], the normalized result (ignoring ε) is the same [−1, 1]. That is because the mean (3 → 30) and the standard deviation (1 → 10) grow together and cancel out. If you then change γ from 1 to 2, the output becomes [−2, 2].

### Watch out

It is wrong to say the final output always has mean 0 and variance 1. γ and β change it again, and ε also has an effect. Also, unlike BatchNorm, which usually pools the mean across neighboring examples, LayerNorm computes within the specified feature axis.

> **Takeaway:** LayerNorm adjusts the center and spread of a group of features, then learns back the scale it needs.

**Check question:** Does LayerNorm for a given token usually depend directly on the values of other examples in the batch?

<details>
<summary>Answer</summary>

Not in a typical Transformer setup. The statistics are computed within that token's feature axis.

</details>

_Further reading: [S4 · PyTorch `LayerNorm` docs](references.md)_

---
