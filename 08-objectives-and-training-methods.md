# Part 8. Beyond the Basic Objective: How Models Are Trained

[← Previous part](07-generation-and-inference.md) · [Contents](README.md) · [References →](references.md)

- [Q33. What do pretraining and SFT share, and where do they differ?](#q33)
- [Q34. Is every "penalize the mistakes" method reinforcement learning?](#q34)
- [Q35. How can we learn from the results of actions we can't differentiate?](#q35)

---

<a id="q33"></a>
## Q33. What do pretraining and SFT share, and where do they differ?

Pretraining and supervised fine-tuning (SFT) of an autoregressive LLM can use the same form of loss: one that raises the probability of the correct next token. The key difference is which data and which positions the training targets. SFT usually adjusts the model with examples of the responses you want.

```math
L_{\text{SFT}} = -\frac{1}{|A|}\sum_{t \in A} \log p_\theta(y_t \mid x, y_{<t})
```

**In plain words:** Show the model the prompt and the reference response up to the current point. Turn the probability the model gave to the next reference-response token into a penalty. Collect this penalty only at the response positions marked as training targets, and average it.

### Reading the symbols

- **x, yₜ, y<ₜ** (x, y sub t, y before t: the request and the reference response): x is the prompt, yₜ is the response token to predict, and y<ₜ is the reference response up to just before that token.
- **p_θ(yₜ | x, y<ₜ)** (p theta of y sub t given x and y before t: the probability given the request and the earlier part of the response): The comma after | means the two pieces of information are used together as the condition. The model looks not only at the request but also at the earlier part of the response.
- **A and t∈A** (A, and t in A: the positions that get a loss): A is the set of selected token positions. ∈ shows membership. In this explanation, the positions of the assistant's reference response are selected.
- **|A| and −log** (the size of A, and minus log: a token penalty averaged by the count): |A| is the number of selected positions. Take the negative log of each correct-token probability, add them up, and divide by this number. SFT stands for supervised fine-tuning.

### Step by step

1. Put the prompt into the input, and use the reference response tokens as the correct next tokens. If the probability of a correct token is 0.2, the penalty at that position is about 1.609.
2. Add up only the response positions that belong to A. Leaving the prompt tokens out of the loss does not mean deleting the prompt itself from the input.
3. It uses the same next-token loss form as pretraining, while what data and which positions it learns from are different. This formula is how the model learns to follow example responses.

### Following the numbers

Say the prompt is "2 plus 2 is" and the reference response is "4." If one token of the correct response has a probability of 0.2, the loss at that position is about 1.609. Raising that probability to 0.8 lowers it to about 0.223.

### Where it shows up in models

Pretraining is used to learn patterns in a wide variety of text, and SFT is used to adjust response behaviors, such as following instructions or sticking to a format, through examples. Both are training that modifies the parameters. This is different from just putting examples into the prompt.

### What changes if you tweak it

If you change the reference response attached to the same prompt from a long-winded answer to a short, accurate one, the tokens whose probability the training is asked to raise change too. Also, if you include the prompt in the loss target A, part of the training's weight goes to predicting the prompt as well.

### Watch out

Even with good examples, there is no guarantee that the behavior holds in every new situation. Wrong or biased examples can become training targets just as they are. Putting the loss only on the response positions is one common setup, and not every SFT implementation uses the same range.

> **Takeaway:** The heart of SFT lies less in some different, magical loss and more in training examples that capture the behavior you want.

**Check question:** If pretraining and SFT have the same loss shape, is the training effect the same too?

<details>
<summary>Answer</summary>

No. If the data distribution, the target positions, the amount of training, and what gets updated are different, the behavior the model learns is different too.

</details>

---

<a id="q34"></a>
## Q34. Is every "penalize the mistakes" method reinforcement learning?

In supervised learning, too, the loss grows when the model is wrong. Having a penalty does not by itself make something reinforcement learning. Reinforcement learning improves the policy that chooses actions, based on the rewards that come from the results of those actions. If an action also changes later states and opportunities, the long-term results have to be considered as well.

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
```

**In plain words:** Start with the reward received now and add the rewards to come. Multiply the reward one step later by γ, and the reward two steps later by γ twice. The value you get by adding them up this way, adjusting how much weight future rewards get, is the return Gₜ.

### Reading the symbols

- **rₜ, rₜ₊₁** (r sub t, r sub t plus 1: the reward now and at the next step): r stands for reward. In this notation, read it as the reward obtained at that step after the action. The subscript is the order in time.
- **γ** (gamma: the discount rate): It is usually set between 0 and 1. If it is 0.9, the weight shrinks to 90% of the previous one with each step further away.
- **γ²rₜ₊₂** (gamma squared times r sub t plus 2: the share of the reward two steps later): γ² is γ × γ. The t+2 at the bottom is a position in time and the 2 at the top is an exponent, so they play different roles.
- **Gₜ and ⋯** (G sub t, and "and so on": the cumulative reward from now on): Gₜ is the value gathered from now onward. The three dots mean you keep adding the later terms by the same rule.

### Step by step

1. Suppose the reward now is 1, the next reward is 10, and there are no rewards after that. With γ = 0.9, 1 + 0.9 × 10 = 10.
2. A choice that gives 5 now and no rewards afterward has a return of 5. Looking only at the immediate reward, 5 is better, but once the future is included, the first choice is better.
3. This formula is how you compute the return from experience. Which action's probability to raise is decided by a policy-learning algorithm that uses this value.

### Following the numbers

With an immediate reward of 1, a next reward of 10, and γ = 0.9, the return is 1 + 9 = 10. This can be better than an action that gets a reward of 5 right away but nothing afterward. If you choose by looking only at the current score, you miss this difference.

### Where it shows up in models

It is used in robot control, games, reward-based tuning of language models, and so on. In a language model, the choices that generate a response can be treated as actions, and the result can be given a reward. Not all reinforcement learning is a long sequential problem; some cases involve only a single choice.

### What changes if you tweak it

For the same two choices, lowering γ from 0.9 to 0.1 changes the first choice's return from 10 to 2, and the choice that gets 5 immediately becomes the better one. The discount rate changes "how much we count rewards that are far away"; it does not change the rewards actually received.

### Watch out

The word "penalty" or a negative score alone does not make something reinforcement learning. Reducing the gap between a fixed, correct sales figure and a prediction is usually supervised learning. The key distinction is whether you choose actions, get rewards from the results, and improve the policy behind those actions. Reinforcement learning also includes problems that end after a single choice.

> **Takeaway:** What makes something reinforcement learning is not the name "penalty" but the structure: improving a policy through action choices and rewards.

**Check question:** If you put a minus sign in front of MSE and call it a reward, does it automatically become reinforcement learning?

<details>
<summary>Answer</summary>

No. You have to look at what choice is made, what result is observed, and how the policy is improved.

</details>

_Further reading: [S8 · OpenAI, *Spinning Up in Deep RL*](references.md)_

---

<a id="q35"></a>
## Q35. How can we learn from the results of actions we can't differentiate?

When you choose an action and get a result from an outside environment, it can be hard to differentiate that result directly. Policy gradient instead differentiates "the probability of choosing that action." If the result is better than the baseline, it produces a signal in the direction of raising that probability; if worse, in the direction of lowering it. Below is a simple per-sample loss that shows the principle.

```math
L_{\text{PG}} = -A \log \pi_\theta(a \mid s), \qquad A = R - b
```

**In plain words:** Subtract the comparison baseline from the result score of the chosen action to get A, how much better it went. Multiply the log of the probability the policy gave that action at the moment of choosing by −A. Reducing this loss creates a learning signal toward choosing actions that beat the baseline more often and actions that fell short less often.

### Reading the symbols

- **s, a, π_θ(a | s)** (s, a, and pi theta of a given s: the situation, the chosen action, and its probability): π is read "pi" and means the policy. It is "the probability of choosing action a in situation s," and | marks the condition. θ is the parameters of the policy model.
- **R and b** (the observed result and the comparison baseline): R is the reward or the cumulative reward. b is a baseline, such as the usual expected score for that situation. In simple examples, a baseline set independently of the action choice is used.
- **A = R − b** (how much better than the baseline): If it is positive, the result was better than the baseline; if negative, worse. In this formula, A is held fixed and you differentiate the probability side of the policy.
- **L_PG = −A log π** (L sub PG: the loss for policy gradient training): The coefficient −A in front of the log flips whether the probability is raised or lowered. PG is short for the name "policy gradient." The update gathers this signal from many experiences.

### Step by step

1. With a baseline b = 5 and a result R = 8, A = 3. At an action probability of 0.2, the loss is −3 log 0.2 ≈ 4.828.
2. With the same A, raising the probability to 0.3 makes the loss about 3.612. An update that tries to lower the loss pushes that action's probability up.
3. Instead of differentiating the action choice or the outside result itself, you differentiate the probability of choosing that action. Which actions were actually good is estimated from many experiences.

### Following the numbers

If the baseline score for a situation is 5 and the result of the chosen action is 8, A = 3. When this action's probability is 0.2, the loss is −3 log 0.2 ≈ 4.828. If the probability rises to 0.3, the loss drops to about 3.612. If A is negative, the opposite holds: lowering the probability is the favorable direction.

### Where it shows up in models

This formula is the basic policy gradient loss for a single sample. In practice, many actions and experiences are averaged to improve the expected reward. Language models can also use this principle to adjust the probability of their choices, using the evaluation results of the responses they generated.

### What changes if you tweak it

If the result had been R = 2, A = −3. The loss is about −4.828 at a probability of 0.2 and about −3.612 at 0.3. If you want a smaller loss, this time the way to get it is to lower the probability. If A = 0, this experience gives no update signal through this term.

### Watch out

The loss here is not an error score that is always 0 or more. It is a signed formula for making updates, so you should not compare the quality of actions across different experiences by the loss number alone. You need many experiences and a suitable baseline so that you do not put too much trust in a single lucky result, and this one formula is not the whole PPO or RLHF algorithm.

> **Takeaway:** Even if you cannot differentiate the choice of an action directly, you can differentiate the probability of choosing that action.

**Check question:** In this basic policy gradient term, what signal does a sample with A = 0 give?

<details>
<summary>Answer</summary>

The policy gradient for that term is 0. The result is treated as neither better nor worse than the baseline.

</details>

_Further reading: [S8 · OpenAI, *Spinning Up in Deep RL*](references.md)_

---
