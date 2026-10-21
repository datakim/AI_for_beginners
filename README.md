# AI for Beginners: Machine Learning and Deep Learning, Read Through Formulas

**35 core questions in 8 parts.** Read the symbols, follow the calculation, and connect each formula to real models.

## What this guide is for

What does a model compute, what does it learn, and why is it built the way it is? This guide translates the key formulas into plain sentences, works through them with small numbers, and then connects them to real models. The goal is not to memorize formulas but to understand why each part is there and what happens when you change it.

It starts from arithmetic and simple algebra. Ideas like vectors, derivatives, and logarithms are explained where they first appear. If you have already used models, you can also read it to reconnect familiar terms to how they work inside.

## How each question is laid out

Every question has two halves. The first half reads the formula in plain words and walks through the role of each symbol and the order of calculation. The second half checks the idea with small numbers and with real models, shows what happens when you change one part, and points out where the interpretation stops.

On a first read, go through Q01 to Q10 in order. After that you can jump to any topic you like; each part reuses ideas from the earlier ones. Each question ends with a check question, and the answer is folded right below it.

## Scope

This is an independent explainer focused on how the calculations work. Career advice and Python syntax are out of scope.

The numbers in the examples are made up to show the calculation. Where the details differ between models, the text says how far a statement applies. Sources for further reading are collected in [References](references.md).

## Contents

**[Before you read the formulas](00-reading-formulas.md)**

### [Part 1. Models and Learning](01-models-and-learning.md)
- [Q01. What actually changes when a model learns?](01-models-and-learning.md#q01)
- [Q02. Why turn "how wrong" into a single number?](01-models-and-learning.md#q02)
- [Q03. Why split data into training and test sets?](01-models-and-learning.md#q03)
- [Q04. What differs between training a model and using it?](01-models-and-learning.md#q04)

### [Part 2. The Math Intuition You Need](02-math-intuition.md)
- [Q05. Why use a vector instead of a single number?](02-math-intuition.md#q05)
- [Q06. Why does a dot product work as a relevance score?](02-math-intuition.md#q06)
- [Q07. What does matrix multiplication actually change?](02-math-intuition.md#q07)
- [Q08. How does a derivative tell us which way to adjust?](02-math-intuition.md#q08)

### [Part 3. Classical Machine Learning and Evaluation](03-classical-ml-and-evaluation.md)
- [Q09. How are linear regression and logistic regression connected?](03-classical-ml-and-evaluation.md#q09)
- [Q10. Why take the log of the correct answer's probability to make a loss?](03-classical-ml-and-evaluation.md#q10)
- [Q11. What does a tree learn, and what does boosting add?](03-classical-ml-and-evaluation.md#q11)
- [Q12. Why can a highly accurate model be useless?](03-classical-ml-and-evaluation.md#q12)
- [Q13. How can a model learn structure without any answers?](03-classical-ml-and-evaluation.md#q13)

### [Part 4. How Neural Networks Learn](04-how-neural-networks-learn.md)
- [Q14. Why do we need activation functions when stacking layers?](04-how-neural-networks-learn.md#q14)
- [Q15. How does backpropagation work out each parameter's influence?](04-how-neural-networks-learn.md#q15)
- [Q16. Why does picking exactly one option cut off the learning signal?](04-how-neural-networks-learn.md#q16)
- [Q17. What else does a neural network learn besides the answer?](04-how-neural-networks-learn.md#q17)

### [Part 5. Why Training Works or Breaks](05-when-training-works-or-breaks.md)
- [Q18. Why can a large learning rate make the loss go up?](05-when-training-works-or-breaks.md#q18)
- [Q19. Why learn in small batches, and why use Adam?](05-when-training-works-or-breaks.md#q19)
- [Q20. Why does the training score improve while the real-world score gets worse?](05-when-training-works-or-breaks.md#q20)
- [Q21. Why do residual connections help deep models learn?](05-when-training-works-or-breaks.md#q21)
- [Q22. What does LayerNorm keep steady?](05-when-training-works-or-breaks.md#q22)

### [Part 6. Transformers and LLMs](06-transformers-and-llms.md)
- [Q23. How does text become numbers a model can compute with?](06-transformers-and-llms.md#q23)
- [Q24. Why does attention make separate Q, K, and V?](06-transformers-and-llms.md#q24)
- [Q25. Why does attention mix several tokens instead of picking one?](06-transformers-and-llms.md#q25)
- [Q26. Why divide attention scores by the square root of the dimension?](06-transformers-and-llms.md#q26)
- [Q27. How does a model tell order apart and keep out future information?](06-transformers-and-llms.md#q27)
- [Q28. How does predicting one next token turn into generating sentences?](06-transformers-and-llms.md#q28)

### [Part 7. Generation and Inference](07-generation-and-inference.md)
- [Q29. What exactly does temperature change?](07-generation-and-inference.md#q29)
- [Q30. How is top-p different from temperature?](07-generation-and-inference.md#q30)
- [Q31. What does the KV cache remember to make things faster?](07-generation-and-inference.md#q31)
- [Q32. Why does a diffusion model predict noise instead of the image?](07-generation-and-inference.md#q32)

### [Part 8. Beyond the Basic Objective: How Models Are Trained](08-objectives-and-training-methods.md)
- [Q33. What do pretraining and SFT share, and where do they differ?](08-objectives-and-training-methods.md#q33)
- [Q34. Is every "penalize the mistakes" method reinforcement learning?](08-objectives-and-training-methods.md#q34)
- [Q35. How can we learn from the results of actions we can't differentiate?](08-objectives-and-training-methods.md#q35)

**[References](references.md)**

## A note on the math

Formulas are written in LaTeX and render on GitHub. If a formula shows up as raw text in your viewer, open the file on github.com.
