# Part 3. Classical Machine Learning and Evaluation

[← Previous part](02-math-intuition.md) · [Contents](README.md) · [Next part →](04-how-neural-networks-learn.md)

- [Q09. How are linear regression and logistic regression connected?](#q09)
- [Q10. Why take the log of the correct answer's probability to make a loss?](#q10)
- [Q11. What does a tree learn, and what does boosting add?](#q11)
- [Q12. Why can a highly accurate model be useless?](#q12)
- [Q13. How can a model learn structure without any answers?](#q13)

---

<a id="q09"></a>
## Q09. How are linear regression and logistic regression connected?

Linear regression predicts a number by taking a weighted sum of features. If you turn this weighted-sum score into a value between 0 and 1, it can express a probability for binary classification. Logistic regression does that with a transformation called the sigmoid.

```math
z = \mathbf{w}^{\top}\mathbf{x} + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}
```

**In plain words:** Multiply each input feature by its importance, add them up, then add a base value b to get a score z. Put that score through a transformation called the sigmoid to turn it into a value between 0 and 1. Use that result as p, the "predicted probability of being positive."

### Reading the symbols

- **wᵀx** ("w transpose x"): Here it is the dot product: multiply matching slots and add them up. The T on top is not a power. It is the transpose mark, which swaps the direction of rows and columns.
- **b and z** ("bias and score"): b is the base value added to the weighted sum. z is the score before the transformation, and it can be negative or larger than 1.
- **σ(z)** ("sigma of z"): Here the small σ is the name of the sigmoid function. It is different from the capital Σ, which means a sum.
- **e⁻ᶻ** ("e to the minus z"): e is a fixed number, about 2.718. A negative exponent means a reciprocal, so e⁻ᶻ = 1/eᶻ. This value is always positive.
- **1/(1+e⁻ᶻ)** ("1 divided by the whole parenthesis"): The denominator is always greater than 1, so the result lands between 0 and 1. What you divide by is the whole 1+e⁻ᶻ, not just e⁻ᶻ.

### Step by step

1. First compute z. If z = 0, the next steps are e⁰ = 1, denominator 1+1 = 2, final probability 1/2.
2. As z gets larger, −z gets smaller and e⁻ᶻ gets smaller too. The denominator approaches 1, so the probability climbs toward 1.
3. When z gets smaller, the opposite happens. What this formula decides is the probability. The cutoff above which you call something positive is decided separately.

### Following the numbers

If z = 0, then p = 1/2. If z ≈ 1.386, then p ≈ 0.8, and if z ≈ −1.386, then p ≈ 0.2. Whether 0.8 counts as positive depends on a separate cutoff value called the threshold. Computing the probability and making the final call are two separate steps.

### Where it shows up in models

It is used in basic models that predict things like whether someone clicks or whether a customer leaves. The binary classification output layer of a neural network can use the same structure. However, if several nonlinear layers come before it, the model can draw complex boundaries with respect to the original input.

### What changes if you tweak it

Keep the predicted probability at 0.8 and only raise the decision threshold from 0.5 → 0.9, and the call flips from positive → negative. The probability the model computed did not change. If you take out the sigmoid itself, you cannot call outputs like 2 or −3 probabilities.

### Watch out

Squeezing values into the 0 to 1 range does not mean they also match real-world frequencies. Checking whether about 80% of the cases predicted at 0.8 are actually positive is a question of probability calibration, which is different from simply limiting the range.

> **Takeaway:** Logistic regression is a binary classification model that turns a linear score into a probability.

**Check question:** In a model with p = 0.8, what changes if you move the decision threshold from 0.5 to 0.9?

<details>
<summary>Answer</summary>

The probability stays the same, but the final call changes from positive to negative.

</details>

---

<a id="q10"></a>
## Q10. Why take the log of the correct answer's probability to make a loss?

If you want a model that gives high probability to the correct answer, the loss should get smaller as the correct answer's probability gets larger. The negative log does this, and it also harshly penalizes confident wrong answers that give the correct answer a probability close to 0. And a goal that multiplies the probabilities of many examples together can be handled as a sum once you take the log.

```math
L = -\log p_{\text{true}}, \qquad \log(p_1 p_2) = \log p_1 + \log p_2
```

**In plain words:** Take the probability the model gave to the item that is actually correct. Take the log of that probability and flip its sign to use it as the loss. If the correct answer's probability is close to 1, the loss is small. If it is close to 0, the loss grows sharply.

### Reading the symbols

- **p_true** ("p true"): Not the highest probability the model assigned, but the probability it gave to the actual correct item. If the correct answer is cat, read the probability in the cat slot.
- **log p** ("log of p"): Here it is the natural log. It asks, "To what power must e be raised to get p?" Since e⁰ = 1, log 1 = 0.
- **−log p** ("minus log p"): The log of any p with 0 < p < 1 is negative. Putting a minus sign in front turns it into a positive penalty. As the probability approaches 0, the penalty grows without limit.
- **log(p₁p₂) = log p₁ + log p₂**: Multiplying two probabilities and then taking the log gives the same value as adding the log of each probability. The p₁p₂ inside the parentheses on the left is a multiplication.

### Step by step

1. From the predicted distribution, pick out only the probability of the actual correct answer. If that value is 0.5, then log 0.5 ≈ −0.693.
2. Add the minus sign to get a penalty of 0.693. A small log table or a calculator is fine. The key is the relationship: the larger the correct answer's probability, the smaller the penalty.
3. If you want to make the correct-answer probabilities of many examples large all together, you can think of their product. With the log, you can move this multiplication goal over to addition, which makes it easy to combine the per-example losses.

### Following the numbers

If the correct answer's probability is 0.9, the loss is about 0.105. At 0.5 it is 0.693, and at 0.1 it is 2.303. Unlike just counting whether the answer was wrong, this gives a learning signal about how appropriate the probability was, even among predictions that already land on the correct answer.

### Where it shows up in models

Cross-entropy for classification with a single correct class has this form. LLMs also compute the loss from the probability they gave to the actual next token. In general, cross-entropy averages the negative log probabilities, weighting each one by its share in the correct-answer distribution.

### What changes if you tweak it

As the correct answer's probability goes 0.1 → 0.5 → 0.9, the loss falls from about 2.303 → 0.693 → 0.105. Even when the final call is already correct, raising the probability given to that correct answer can push the loss down further.

### Watch out

What goes down is this loss with respect to the given correct answers. If the training labels contain errors, the model may learn to believe wrong content even more strongly. A low next-token loss for an LLM does not, on its own, guarantee that every sentence is true.

> **Takeaway:** Negative log probability turns how much the model believed the correct answer into a penalty it can learn from.

**Check question:** If you raise the correct answer's probability, does this loss get bigger or smaller?

<details>
<summary>Answer</summary>

It gets smaller. As the probability approaches 1, the loss approaches 0.

</details>

---

<a id="q11"></a>
## Q11. What does a tree learn, and what does boosting add?

A decision tree asks questions about input values and splits the examples into several regions. The prediction for each region is also set from the data. Boosting adds small models one after another to make up for the errors the existing model left behind. In squared-error regression, you can think of it as the next tree learning the residuals that are left over.

```math
F_{m+1}(x) = F_m(x) + \eta\, h_m(x)
```

**In plain words:** Predict with the model built so far. Take the correction from the newly built small model and shrink it by the learning rate. Add that small correction to the existing prediction to get the prediction for the next stage.

### Reading the symbols

- **Fₘ(x)** ("F sub m of x"): The prediction of the stage-m model. It is the result of putting x into the current model. The subscript m shows how many rounds of models have been stacked.
- **hₘ(x)** ("h sub m of x"): The output of the new correction model. It is the prediction of a small model added to fix what the existing model missed. Here you can picture a small decision tree.
- **ηhₘ(x)** ("eta times h sub m of x"): The correction multiplied by the learning rate. η is read "eta." It decides how much of the new model's output to apply. If η = 0.1, 10% of the correction is applied.
- **Fₘ₊₁(x)** ("F sub m plus 1 of x"): The prediction of the next-stage model. It is the existing prediction and the correction combined. m+1 means the stage number went up by one. It does not mean adding 1 to x.

### Step by step

1. Apply the existing model and the new correction model separately to the same input x. For example, the existing prediction is 6 and the correction is 3.
2. Multiply the correction by η = 0.1 to get 0.3. Add it to the existing 6, and the new prediction is 6.3.
3. In squared-error regression, the new model can be trained to follow the residual, which is correct answer − existing prediction. If the correct answer is 10, the residual is 4, but the new tree does not necessarily hit 4 exactly.

### Following the numbers

Suppose an example's correct answer is 10 and the existing prediction is 6, so the residual is 4. If the new tree proposes a correction of 3 for that example and η = 0.1, the new prediction is 6.3. Instead of fixing everything perfectly in one go, it repeats small corrections.

### Where it shows up in models

Tree boosting methods such as LightGBM and XGBoost are widely used for predicting tabular data. A tree finds feature thresholds and a split structure. It does not learn every branching choice by differentiation the way a typical neural network does.

### What changes if you tweak it

Fix the existing prediction at 6 and the correction at 3, and change only η from 0.1 → 1: the new prediction goes from 6.3 → 9. It fixes more in a single step. However, the next tree learns whatever error remains at that point, so a model retrained from scratch with a different learning rate is not related to the original by a simple multiple.

### Watch out

The more trees you add, the more the model can chase even small accidents in the training data. That is why the number of trees to add is chosen using validation performance. In classification boosting, the model sometimes adds scores from before the transformation rather than probabilities, so not every problem can be explained only as correct answer − prediction.

> **Takeaway:** Boosting stacks models that each correct the existing prediction a little.

**Check question:** If you make η smaller, what happens to the effect of the same tree correction?

<details>
<summary>Answer</summary>

It gets smaller. To get the same amount of correction, you usually need more stages.

</details>

---

<a id="q12"></a>
## Q12. Why can a highly accurate model be useless?

In data where most cases are normal, answering "normal" for everything still gives high accuracy. But it does not find a single one of the events it was supposed to catch. Depending on the goal, you have to look separately at how accurate the alarms are and how well the model avoids missing real events.

```math
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}
```

**In plain words:** Precision asks, "Of the things the model flagged as events, how many were actual events?" Recall asks, "Of the events that actually happened, how many did the model find?" Both use the same number of hits, but they differ in what they treat as the whole.

### Reading the symbols

- **TP** ("true positives"): The number of alarms that went off correctly. These are cases that really were events and that the model also flagged as events. TP appears in the numerator of both formulas.
- **FP** ("false positives"): The number of alarms that went off wrongly. These are cases that were actually normal but that the model flagged as events. TP+FP is all the alarms the model raised.
- **FN** ("false negatives"): The number of missed events. These are cases that were actual events but that the model let pass as normal. TP+FN is all the actual events.
- **Precision, Recall**: A fraction means "the share of the whole that qualifies." 0.4 means 40%. The number of cases correctly called normal does not enter either formula directly.

### Step by step

1. For precision, start from the list of alarms. If 8 of 20 alarms are real, then TP = 8, FP = 12, and 8/(8+12) = 40%.
2. For recall, start from the list of actual events. If you found 8 of 10 actual events, then TP = 8, FN = 2, and 8/(8+2) = 80%.
3. The two values being different is not a contradiction. The first formula checks whether the alarms can be trusted. The second checks how well the model avoided missing events.

### Following the numbers

If 10 out of 1,000 cases are events, calling everything normal still gives 99% accuracy. If another model raises 20 alarms and finds 8 events, its precision is 8/20 = 40% and its recall is 8/10 = 80%.

### Where it shows up in models

These are used in areas such as anomaly detection and search. Even with the same model, lowering the threshold usually produces more alarms and raises recall. Precision often drops, but it does not go down monotonically on every dataset.

### What changes if you tweak it

Say you lower the decision threshold, raise 10 more alarms, and only 1 more real event is added. Precision drops from 8/20 = 40% to 9/30 = 30%, and recall rises from 8/10 = 80% to 9/10 = 90%. This is a case where you find more, but you also get more false alarms.

### Watch out

Lowering the threshold does not always lower precision. It depends on what the added alarms are made of. If there are no alarms at all, or no actual events at all, the denominator is 0, and you need a separate rule to handle it. You also have to weigh which costs more: a missed event or a false alarm.

> **Takeaway:** A good evaluation metric connects the model's score to the problem you actually want to solve.

**Check question:** What if 20 of 100 alarms are real and there were 25 actual events?

<details>
<summary>Answer</summary>

Precision is 20% and recall is 80%. The two numbers use different denominators.

</details>

---

<a id="q13"></a>
## Q13. How can a model learn structure without any answers?

Even without correct-answer labels, you can set a criterion like "let's group similar examples together." k-means repeats two moves: assign each example to the nearest center, then move each group's center to the average position.

```math
J = \sum_{i=1}^{N} \lVert \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \rVert^2
```

**In plain words:** For each example, subtract the center of the group it is assigned to. Square the length of that difference to get a penalty for that example. Adjust the group assignments and the centers in the direction that makes J, the sum of all the examples' penalties, smaller.

### Reading the symbols

- **xᵢ** ("x sub i"): The vector for the i-th example. It is a bundle of numbers holding one example's features. N is the number of examples, and Σ adds up the penalties from i = 1 to N.
- **cᵢ** ("c sub i"): The group number of the i-th example. Keep the example number i and the group number cᵢ apart. For example, if example 7 belongs to group 2, then c₇ = 2.
- **μ with cᵢ below it** ("mu sub c sub i"): The center of that group. μ is read "mu." If example 7 has c₇ = 2, you fetch the center μ₂. You use the example number to find the group, and the group number to find the center.
- **‖example − center‖²** ("squared norm of example minus center"): The squared distance to the center. Square the difference in each slot and add them up. Since it is the squared length, you end up not needing a square root at all.
- **J**: The total distance penalty. It is not a score compared against correct-answer labels. It is the goal "members of the same group should gather close to their center" turned into a number.

### Step by step

1. Pick one example, check the group number it is assigned to, then find that group's center. You must not take the center from just any group.
2. Compute the difference between the example and the center, and square it. If the example is 1 and the center is 1.5, then (1−1.5)² = 0.25. Repeat for every example and add them up.
3. k-means alternates between reassigning examples to the nearest center and moving each group's center to the average of the examples in that group. This formula is the score that process is trying to shrink.

### Following the numbers

If you gather the numbers 1, 2, 9, 10 into two groups, the center of [1, 2] is 1.5 and the center of [9, 10] is 9.5. The sum of squared distances to the centers is 1, from adding four values of 0.25.

### Where it shows up in models

It can be used to explore customer behavior or to group similar examples in an embedding space. It is one example of unsupervised learning, which finds structure without correct-answer labels. Whether that structure is useful for the business needs separate interpretation and validation.

### What changes if you tweak it

With [1, 2, 9, 10], if you use one group, the total penalty around the center 5.5 is 65. Put the two groups' centers at 1.5 and 9.5, and the total is 1. With four groups of one point each, even 0 is possible, so you cannot keep increasing the number of groups just by looking at the penalty.

### Watch out

The number of groups k is a choice made outside this formula. Feature units and the starting centers also change the result. A low distance penalty does not guarantee a grouping that is useful for the business, or a true grouping that already existed in the world.

> **Takeaway:** Even without answers, setting a criterion such as closeness lets you find structure.

**Check question:** What problem arises if you feed in income in Korean won and visit count as a single-digit number?

<details>
<summary>Answer</summary>

Differences in income, which sits on a large scale, can dominate the distance. You need to rescale the features in a way that fits what they mean.

</details>

---
