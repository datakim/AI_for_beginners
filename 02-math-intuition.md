# Part 2. The Math Intuition You Need

[← Previous part](01-models-and-learning.md) · [Contents](README.md) · [Next part →](03-classical-ml-and-evaluation.md)

- [Q05. Why use a vector instead of a single number?](#q05)
- [Q06. Why does a dot product work as a relevance score?](#q06)
- [Q07. What does matrix multiplication actually change?](#q07)
- [Q08. How does a derivative tell us which way to adjust?](#q08)

---

<a id="q05"></a>
## Q05. Why use a vector instead of a single number?

If several features describe one thing, you need several numbers too. A vector is these numbers bundled together in order. A vector can also be seen as an arrow in space, but at first it is fine to think of it as a list of numbers with several slots.

```math
\mathbf{x} = (x_1, x_2), \qquad \lVert\mathbf{x}\rVert = \sqrt{x_1^2 + x_2^2}
```

**In plain words:** x is a bundle holding two numbers in order. To find the length of that bundle when you think of it as an arrow, square each number, add them up, and then take the square root of the sum.

### Reading the symbols

- **Bold x** ($\mathbf{x}$, read "vector x"): Not a single number but a bundle of numbers in several slots. The bold type makes this difference visible.
- **x₁, x₂** ("x one", "x two"): The values in the first and second slots. The 2 written below is not a square. x₂² would mean "square the value in the second slot."
- **‖x‖** ("the norm of x"): Here it means the length of the vector. The double bars on each side are the symbol for finding the length; don't confuse them with the single bars of absolute value.
- **√ and squaring**: √25 is 5, the answer to "which number, 0 or greater, gives 25 when squared?" In the length formula, it turns the sum of the squared values (each slot's distance from 0, squared) back into the size of a length.

### Step by step

1. First, spread the vector out slot by slot. For [3, 4], x₁=3 and x₂=4.
2. Square each slot and add them up: 3²+4²=9+16=25. Finally, compute √25=5.
3. In two dimensions, this is the length of the hypotenuse of a right triangle that is 3 wide and 4 tall. Even with more slots, the order of computation, "square each slot → sum → square root," stays the same.

### Following the numbers

If you write one seller's features as [visits 3, purchases 4], the length is √(9+16)=5. But if you switch visits from units of a thousand people to single people, the length changes a lot. This is why scaling is needed when comparing features that have different units.

### Where it shows up in models

Vectors are used for model inputs, for embeddings in document search, and for the internal representations of images. An embedding is a representation that turns something into a vector. A person can decide what each slot means, or a model can build a useful representation through learning.

### What changes if you tweak it

Multiplying every slot by ten turns [3, 4] into [30, 40], and the length goes from 5 to 50. The direction stays the same. Multiplying just one slot by a thousand changes the direction too, so changing the units of an input affects distance and relevance calculations.

### Watch out

A larger length does not mean the content is more important or more accurate. Especially when you bundle numbers with different units, such as visit counts and amounts of money, you need to check that the raw size of the numbers is not dominating the comparison.

> **Takeaway:** A vector is a bundle of numbers that lets you compute with several features together.

**Check question:** For [3, 4] and [30, 40], which one is the same: the direction or the length?

<details>
<summary>Answer</summary>

The direction is the same, and the lengths differ by a factor of ten. This is the starting point for understanding the difference between the dot product and cosine similarity.

</details>

---

<a id="q06"></a>
## Q06. Why does a dot product work as a relevance score?

Multiply the matching slots of two vectors and add up the results, and you get the dot product. When the two vectors point in similar directions and their matching features are large together, the score can come out high. However, for the dot product to represent meaningful relevance, the vectors have to be built or learned for that purpose.

```math
\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d} q_i k_i = \lVert\mathbf{q}\rVert\,\lVert\mathbf{k}\rVert\cos\phi
```

**In plain words:** Multiply the slots with the same number in the two vectors. The sum of all those products is the dot product. The expression on the right means this score is determined by the lengths of the two vectors and by how closely their directions match.

### Reading the symbols

- **q·k** ("q dot k": the dot product of q and k): The dot in the middle stands for a computation that compares vectors. The result is a single number, not a vector.
- **Σᵢ₌₁ᵈ qᵢkᵢ** ("sum from i = 1 to d of q sub i times k sub i": multiply matching slots and add): d is the number of slots. Two values written side by side, like qᵢkᵢ, are multiplied. Add them up while changing i from 1 to d.
- **‖q‖, ‖k‖** ("norm of q", "norm of k": the lengths of the two vectors): The length you learned in [Q05](#q05). If either one gets bigger, the dot product can grow even when the direction is the same.
- **cos φ** ("cosine phi"): φ is the angle between the two vectors. cos φ expresses how well the directions match, on a scale from −1 to 1. The same direction gives 1, a right angle gives 0, and opposite directions give −1.

### Step by step

1. For computing, the middle expression is easier. For [1, 2] and [2, 1], multiply the matching slots to get 1×2 and 2×1, then add them.
2. For interpreting the meaning, look at the expression on the right. A high score could be because the directions match well, or because the vectors are long.
3. To compare only the direction, divide the dot product by the product of the two lengths. For vectors that are not the zero vector, what remains is the cosine similarity.

### Following the numbers

With q=[1, 2], the dot product with k=[2, 1] is 1×2+2×1=4. If k=[−2, −1], it is −4. If k=[20, 10], the direction is the same as the first candidate, but the length is larger, so the score becomes 40.

### Where it shows up in models

In attention, it is used for the score that compares a query with a key. It is also used in embedding search. Taking the dot product of vectors whose lengths have been set to 1 gives the same result as cosine similarity, which compares only direction.

### What changes if you tweak it

Changing k=[2, 1] to [20, 10] changes its dot product with q from 4 to 40, but the cosine similarity stays the same. If you want to remove the effect of length in search, you can set the lengths to 1 before taking the dot product.

### Watch out

Taking the dot product of arbitrary bundles of numbers does not automatically produce semantic relevance. You need vectors that fit the purpose of the comparison. In attention, too, a high score is a learned, computational relevance, and it is not necessarily the same as the similarity a person would have in mind.

> **Takeaway:** The dot product lines up the signals in matching slots and compares them, but it is also affected by length.

**Check question:** If the directions of two vectors stay the same but one of them is made twice as long, what happens to the dot product?

<details>
<summary>Answer</summary>

It doubles. The cosine similarity stays the same, as long as neither vector is the zero vector.

</details>

---

<a id="q07"></a>
## Q07. What does matrix multiplication actually change?

A matrix gathers several weighted-sum rules into a single table. Multiplying an input vector by a matrix mixes the original features in different proportions to make a new list of features. A linear layer in a neural network adds a bias to this computation.

```math
\begin{bmatrix} h_1 \\ h_2 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
```

**In plain words:** Mix the two input slots in different proportions to make two output slots. The first output is x₁+2x₂, and the second output is −x₁+x₂. Each row of the matrix is a recipe for making one output slot.

### Reading the symbols

- **x₁, x₂** (the two slots of the input): The vertical list on the right is the input vector. If the order of the numbers changes, the result can change too.
- **The 2×2 grid in the middle** (the matrix): Rows run horizontally and columns run vertically. The first row [1, 2] means "use the first input once and the second input twice."
- **h₁, h₂** (the two slots of the output): The results once the computation is done. The number of outputs equals the number of rows in the matrix.
- **The matrix and the vector written side by side** (matrix multiplication): This does not mean multiplying any slots you like. You pair one row of the matrix with the input entry in the same position, multiply each pair, and then add.

### Step by step

1. Read the first row, [1, 2]. Line up its slots with the input [3, 4] and compute 1×3+2×4=11.
2. Compute the second row, [−1, 1], with the same input: −1×3+1×4=1. Stack the results as [11, 1].
3. The number of columns in the matrix must equal the number of input slots so that the pairs for multiplication line up. A typical neural network learns these proportions, and after the computation it also adds a bias, which acts as a base value.

### Following the numbers

If the input is [3, 4], the first output is 3+8=11 and the second output is −3+4=1. The result [11, 1] re-expresses the same input as features seen from different perspectives.

### Where it shows up in models

It is used when making Q, K, and V in a Transformer, when transforming representations between neural network layers, and when turning the final representation into a score for each class. Gathering many inputs into a matrix for the computation lets the hardware process many multiplications together.

### What changes if you tweak it

Changing the 2 in the first row to 0 changes the first output from 11 to 3. The first output no longer looks at the second input. Adding a new row [1, 1] creates a third output slot with the value 7.

### Watch out

Even if you chain several matrix multiplications, they can be combined into a single matrix multiplication as long as there is no other operation in between. Using many matrices does not by itself always make the relationships a model can express more complex.

> **Takeaway:** Matrix multiplication is a computation that mixes existing features in fixed proportions to make new features.

**Check question:** To increase the output slots from two to three, what would you add to this matrix?

<details>
<summary>Answer</summary>

Add one row, which serves as the rule for the new output. The input dimension stays at two slots.

</details>

---

<a id="q08"></a>
## Q08. How does a derivative tell us which way to adjust?

A derivative measures how much the result changes when you move the input by a very small amount. Because it is the rate of change around the current point, it tells you nothing about the terrain far away. If there are several parameters, you find the rate of change for each one by moving it on its own.

```math
\frac{dL}{dw} \approx \frac{L(w + \Delta w) - L(w)}{\Delta w}
```

**In plain words:** Raise w a little. Find the difference in loss before and after the change, and divide it by the amount you raised w. That gives you an estimate of how much the loss changes when w moves by one unit. The value you get by shrinking the change down to something extremely small is the derivative.

### Reading the symbols

- **L(w)** ("L of w": the loss at w): The loss you get when computing with the parameter value w. It does not mean multiply L by w.
- **Δw** ("delta w"): The amount by which w was changed. Δ marks an amount of change. w+Δw is the parameter value after the change.
- **Numerator and denominator**: The numerator, L(w+Δw)−L(w), is the change in the loss, and the denominator, Δw, is the change in the parameter. You get the rate of change as "change in the result ÷ the change you put in as the cause."
- **dL/dw** ("dL by dw": the derivative of L with respect to w): The rate at which the loss changes when w is changed by a tiny amount. ≈ means the two sides are not exactly equal, because we computed with a finite amount of change.

### Step by step

1. First, compute the current loss and the loss after the change. With L=(w−2)², the loss is 1 at w=3 and 1.0201 at w=3.01.
2. Divide the change in loss, 0.0201, by the change in the parameter, 0.01, to get 2.01. If you make the change smaller, this ratio gets closer to 2.
3. Read the gradient 2 as "if I raise w by 0.001 right now, the loss goes up by about 0.002." Conversely, lowering w a little is the direction in which the loss goes down.

### Following the numbers

With L(w)=(w−2)² and a current value of w=3, the loss is 1. Raising w to 3.01 gives 1.0201, so the rate of change is 0.0201/0.01=2.01. For a very small change, it gets close to 2. In other words, right now, increasing w makes the loss bigger.

### Where it shows up in models

Backpropagation efficiently computes this kind of rate of change for every parameter of a neural network. If the gradient is positive, lowering the value is the direction that locally reduces the loss; if it is negative, raising the value is.

### What changes if you tweak it

When the current gradient is +2, raising w a little increases the loss. At a different point where the gradient is −2, raising it a little decreases the loss. The same "raise it" has a different effect depending on the sign of the gradient where you are now.

### Watch out

This prediction is an approximation that holds over a nearby range. If you move far away, the gradient itself changes. Also, a gradient of 0 is no guarantee that there is nowhere better to go. You can get 0 at a bottom, at a peak, and on a flat stretch.

> **Takeaway:** A derivative tells you how the result changes if you change this value a little, right now.

**Check question:** If the gradient is −3, roughly what happens to the loss when you raise w a little?

<details>
<summary>Answer</summary>

It goes down. For example, if you raise w by 0.01, a first-order approximation says the loss drops by about 0.03.

</details>

---
