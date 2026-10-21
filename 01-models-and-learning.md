# Part 1. Models and Learning

[← Previous part](00-reading-formulas.md) · [Contents](README.md) · [Next part →](02-math-intuition.md)

- [Q01. What actually changes when a model learns?](#q01)
- [Q02. Why turn "how wrong" into a single number?](#q02)
- [Q03. Why split data into training and test sets?](#q03)
- [Q04. What differs between training a model and using it?](#q04)

---

<a id="q01"></a>
## Q01. What actually changes when a model learns?

A model is a rule that takes an input and computes an answer. Learning is the process of changing the adjustable numbers inside that rule to fit the data. Instead of a person writing down the correct answer for every possible case, we decide which computational structure to use and then adjust the numbers inside it.

```math
\hat{y} = f_\theta(x)
```

**In plain words:** Put the input x into the computation rule f. The settings used for this computation are θ. The answer that comes out of the computation is called ŷ. In other words, the right side is the process that produces the answer, and the left side is the name given to the result.

### Reading the symbols

- **x**: The data you feed into the model. It could be a single temperature, or a bundle of numbers converted from a sentence.
- **f<sub>θ</sub>(x)** ("f-theta of x"): The parentheses do not mean multiplication; they mean "compute with this input." The θ written below shows which settings f uses.
- **θ** ("theta"): A single name for all the numbers the model changes through learning. For example, in a straight-line model it refers to both the slope w and the base value b.
- **ŷ** ("y-hat"): The hat on top marks it as a predicted value, to tell it apart from the actual correct answer y. The equals sign = means both sides are the same value.

### Step by step

1. First, separate the input from the settings. The temperature 30 is x, while the 2 that multiplies the temperature and the 10 added afterward belong to θ.
2. If f's computation rule is "multiply by 2 and add 10," compute 30×2+10. Write the result, 70, in the place of ŷ.
3. Learning does not end with doing this computation once. It also includes comparing the result with the correct answer and then fixing the 2 and the 10.

### Following the numbers

Let's predict ice cream sales with ŷ=wx+b. If x is a temperature of 30, w is 2, and b is 10, the prediction is 70. At the same temperature, changing w to 3 gives 100. We did not change the input; we changed the calculator's settings.

### Where it shows up in models

It is used in both training and inference. The weights of a neural network and the coefficients of a regression model are parameters. During training these values are changed; in typical inference, the fixed values are used to compute answers for new inputs.

### What changes if you tweak it

With the same input of 30, changing only w from 2 to 3 changes the prediction from 70 to 100. Conversely, if you fix w=2 and b=10 and change only the input from 30 to 31, the prediction goes from 70 to 72. The first is a change to the model's settings; the second is a computation on a new input.

### Watch out

This formula by itself contains no comparison with the correct answer and no correction of the settings. Don't read "computing with the model" and "training the model" as meaning the same thing. If a relationship cannot be expressed as a straight line, there is a limit to what fixing only w and b can do.

> **Takeaway:** Learning means changing the calculator's settings to fit the data.

**Check question:** If you give the same model a new question as input, do its parameters necessarily change?

<details>
<summary>Answer</summary>

No. Usually only the input and the intermediate computed values change. Changing the parameters requires a separate training process.

</details>

---

<a id="q02"></a>
## Q02. Why turn "how wrong" into a single number?

To fix a model, you need a standard for comparing which parameters are better. The loss function turns the difference between the prediction and the correct answer into a score. The score itself does not tell you which way to correct things; you also have to compute how that score changes as the parameters change before learning can happen.

```math
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
```

**In plain words:** For each example, subtract the correct answer from the prediction. Square that difference to get a penalty for that example. Add up all the penalties and divide by the number of examples, and you get the average penalty L.

### Reading the symbols

- **L**: The loss value, which shows how wrong the model is. In this formula it is 0 or greater, and it is 0 when every prediction equals the correct answer.
- **ŷᵢ, yᵢ** ("y-hat sub i", "y sub i": the i-th prediction and correct answer): The subscript i is the example number. For example, if i=2, you compare the second example's prediction with that example's correct answer.
- **Σᵢ₌₁ᴺ** ("sum from i = 1 to N"): The big sigma Σ is the summation symbol. It repeats the inner computation from example 1 through example N and adds everything up.
- **(ŷᵢ−yᵢ)²** ("the squared error"): Multiply the difference inside the parentheses by itself. Squaring −1 gives 1, so it does not cancel out against positive errors.
- **1/N** ("one over N"): The same as dividing by the number of examples N. It turns the sum into an average, which reduces the effect of the number of examples itself making the score bigger.

### Step by step

1. Start inside the parentheses. A prediction of 5 minus a correct answer of 3 gives an error of 2. Square this error to give 4 points.
2. Compute the next example the same way. With a prediction of 6 and a correct answer of 7, the error is −1, and its square is 1 point. What Σ asks you to do is 4+1.
3. Finally, divide by N=2 to get 2.5. Squaring is a choice to penalize large errors strongly, and dividing is a choice to get the average per example.

### Following the numbers

If the correct answers are 3 and 7 and the predictions are 5 and 6, the errors are 2 and −1. Simply averaging them gives 0.5, but the MSE is (4+1)/2=2.5. The fact that both examples were wrong does not get canceled out.

### Where it shows up in models

It is used to train regression neural networks. You can use it when predicting sales volumes, prices, or sensor values. In image generation, the difference between the predicted noise and the noise that was actually added is also sometimes measured this way.

### What changes if you tweak it

If the error doubles from 2 to 4, the penalty quadruples from 4 to 16. If you use the absolute value instead of the square, the penalty only doubles, from 2 to 4. So squared error reacts more sensitively to a few big mistakes.

### Watch out

A single average penalty cannot tell you which examples caused the problem. In particular, if a few examples have very large errors, those examples can dominate the whole loss. The loss is also measured in the square of the original prediction's unit.

> **Takeaway:** The loss function decides for the model what counts as wrong, and how much each kind of wrong matters.

**Check question:** If the error doubles from 2 to 4, by how many times does the squared error grow?

<details>
<summary>Answer</summary>

It goes from 4 to 16, so it grows four times. Squared error penalizes big mistakes more strongly.

</details>

---

<a id="q03"></a>
## Q03. Why split data into training and test sets?

The score you get by re-solving problems you have already studied cannot tell you whether you will do well on new problems. The same is true for models. You adjust the parameters with training data, choose the settings and when to stop training with validation data, and then check the final performance with test data.

```math
L_{\text{test}} = \frac{1}{M}\sum_{j=1}^{M}\ell\big(f_\theta(x_j),\, y_j\big)
```

**In plain words:** Put one test example into the trained model to get a prediction. Compare that prediction with the test answer to get a penalty. Repeat this for every test example; the average of the penalties is called the test loss.

### Reading the symbols

- **L<sub>test</sub>** ("L test"): The average loss computed on the test data. "test" is not a variable to multiply by; it is a label naming the kind of evaluation.
- **xⱼ, yⱼ** ("x sub j", "y sub j": the j-th input and correct answer): j is the test example number. Inputs and answers must be paired by the same number. M is the total number of test examples.
- **f<sub>θ</sub>(xⱼ)** (the prediction you get by feeding a test input into the model): θ holds the parameter values decided during training. These values stay fixed while the test is being scored.
- **ℓ(prediction, answer)** ("ell of prediction and answer": the loss with the two values plugged in): The small ℓ is a calculator for the penalty of one example. The comma separates the two inputs. For example, it could square the difference between the two values.
- **Σ and 1/M** (sum and average): Add up each example's penalty and divide by M. It is the same averaging structure as in [Q02](#q02), but the role of the data being used is different.

### Step by step

1. Start with the innermost part, f<sub>θ</sub>(xⱼ). Make a prediction from the test input.
2. Put that result into ℓ together with the correct answer yⱼ to get the loss for one example. Then take the average over the whole test set with Σ and 1/M.
3. This formula contains no instruction to change θ. It is a step for checking the score, not a step for using the test answers to train the model further.

### Following the numbers

Imagine you train on past data from January to August, choose the settings with September, and then evaluate on October. If you look at October's answers and keep picking whichever settings fit best, October effectively becomes validation data too.

### Where it shows up in models

It is used to evaluate every kind of prediction model. Time series are split so that the future does not get mixed into training on the past. When duplicate records of the same person or seller appear on both sides, consider a group split that fits the purpose of the evaluation.

### What changes if you tweak it

If you pick model settings by looking at the test score, that test data influences the choice of settings. The more often you repeat this selection, the more you may end up choosing a model that fits that data only by chance. So use validation data to choose settings, and keep the final test separate.

### Watch out

Splitting the data does not remove all leakage on its own. For example, if you first fill in missing values using the mean of the entire dataset, information from the test data also enters the preprocessing. The standard practice is to compute learned preprocessing values, such as the mean, from the training data as well.

> **Takeaway:** An evaluation only means something on problems where the model has not seen the answers in advance.

**Check question:** If you revised the settings ten times by looking at the test score, is that test still fair?

<details>
<summary>Answer</summary>

As a final evaluation, it has been contaminated. You need separate evaluation data that was not used for making choices.

</details>

---

<a id="q04"></a>
## Q04. What differs between training a model and using it?

Training involves both computing an answer and revising the computation rule. Inference usually only computes. When a conversation continues and the model refers to earlier messages, that can happen simply because the input context grows, so this alone does not mean the model has been retrained.

```math
\hat{y} = f_\theta(x), \qquad \theta_{\text{new}} = \theta - \eta\,\nabla_\theta L
```

**In plain words:** The left formula means "compute the answer with the current parameters." The right formula means "make new parameters by subtracting (gradient of the loss × learning rate) from the current parameters." Usually inference does only the first computation, while training, after comparing with the correct answer, also does the second.

### Reading the symbols

- **ŷ=f<sub>θ</sub>(x)** (predicting with the current parameters): This is the formula from [Q01](#q01). The computation that processes an input to produce an answer is also called the forward pass.
- **θ<sub>new</sub>, θ** ("theta new", "theta": the new parameters and the current parameters): "new" is a label meaning "the new one." You compute the new value from the right side of the equals sign, then use that value in the next training step.
- **∇<sub>θ</sub>L** ("nabla theta L": the gradient of the loss with respect to θ): ∇, read "nabla," is a symbol that bundles together the rates of change for many parameters. It shows how much the loss changes when each parameter is raised slightly.
- **η** ("eta": the learning rate): A positive number multiplied by the gradient to control the actual size of the update. The minus sign in front means you want to move in the direction opposite to the one in which the loss increases.

### Step by step

1. First, make a prediction. If you are training, compare it with the correct answer and compute the loss L. Computing the loss is a step needed between the two formulas above.
2. Get the gradient from the loss. If there are several parameters, you get one gradient for each of them.
3. Multiply each gradient by η and subtract it from the matching parameter. This simple update method is gradient descent. When you only run inference, this update step does not happen.

### Following the numbers

Say the predicted sales were 70 but the actual value was 80. An inference service can return 70 and be done. A training job computes the difference from 80 and updates the parameters so that the next prediction improves.

### Where it shows up in models

Generating an LLM response is usually inference. Pretraining and fine-tuning are training. RAG, which puts search results into the prompt, adds that information to the input; on its own, it does not update the parameters.

### What changes if you tweak it

If you add the sentence "The correct answer is 80" when using the same model, the input x changes, so the answer may change. On the other hand, if you compute the loss with that correct answer and update θ, the model's own parameters change. The response may change in both cases, but what changed is in a different place.

### Watch out

Saving the conversation history or an external memory is also separate from updating θ. The saved content can be fed back in as the next input, but that alone does not mean the neural network's weights have been trained.

> **Takeaway:** The answer changing and the model's parameters changing are two different things.

**Check question:** Is changing the temperature a form of training?

<details>
<summary>Answer</summary>

No. It is generally a setting that changes the selection probabilities during inference. We will check this with a formula in [Q29](07-generation-and-inference.md#q29).

</details>

---
