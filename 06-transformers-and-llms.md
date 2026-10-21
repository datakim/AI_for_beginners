# Part 6. Transformers and LLMs

[← Previous part](05-when-training-works-or-breaks.md) · [Contents](README.md) · [Next part →](07-generation-and-inference.md)

- [Q23. How does text become numbers a model can compute with?](#q23)
- [Q24. Why does attention make separate Q, K, and V?](#q24)
- [Q25. Why does attention mix several tokens instead of picking one?](#q25)
- [Q26. Why divide attention scores by the square root of the dimension?](#q26)
- [Q27. How does a model tell order apart and keep out future information?](#q27)
- [Q28. How does predicting one next token turn into generating sentences?](#q28)

---

<a id="q23"></a>
## Q23. How does text become numbers a model can compute with?

A language model does not multiply a string directly by a matrix. First, a tokenizer splits the text into tokens and gives each one an ID number. Then the vector that matches that ID is pulled out of an embedding table. The ID is an address, and the vector is the representation used in the computation.

```math
\mathbf{x}_t = E[\mathrm{id}_t]
```

**In plain words:** Split the sentence into tokens and look up each token's ID. Use the ID of the t-th token to pull out the matching row of the embedding table E. The vector of numbers you pulled out becomes the model input xₜ at that position.

### Reading the symbols

- **t** and **idₜ** ("id sub t": the position and the token ID): t says which position in the sentence it is, and idₜ says which token sits at that position. The position number and the vocabulary number are different things.
- **E** (the embedding table): A table that holds, row by row, a vector of numbers for each token in the vocabulary. These numbers are learnable parameters.
- **E[idₜ]** ("E at id sub t", pulling the matching row out of the table): The square brackets mean "select a row." They do not mean multiplying E by the ID, or treating the ID as an input number and comparing its size.
- $\mathbf{x}_t$ (bold x sub t, the input vector at position t): A bundle of numbers, several entries long, pulled out of the table. At the stage where only the table lookup has happened, the same token ID fetches the same vector.

### Step by step

1. Suppose "cat" has the made-up token ID 7. If it is in the first position of the sentence, then t = 1 and id₁ = 7.
2. If row 7 of E is [0.2, −0.1], pull out that vector and put it in x₁. The token ID 7 is not itself used as if it carried a meaning worth "7."
3. Once later layers compute with position and the surrounding tokens, even the same token can end up with a different representation depending on context. This formula shows only the table lookup step that comes before that.

### Following the numbers

In a made-up vocabulary, say "cat" is ID 7 and "dog" is ID 42. If E[7] = [0.2, −0.1] and E[42] = [0.3, 0.4], these vectors become the model input. The IDs being 35 apart does not mean the meanings are that far apart.

### Where it shows up in models

This is used at the input stage of an LLM. The embedding vectors change during training, and as they pass through the later Transformer layers, they turn into representations that reflect context. Even the same token can have different later representations depending on its role in the sentence.

### What changes if you tweak it

Change the table entry for token 7 from [0.2, −0.1] to [0.8, 0.3], and the input vector fetched whenever that token appears changes. On the other hand, if you consistently reorder the IDs and the table rows together, only the ID labels change, and the same vectors can still be fetched.

### Watch out

A token is not always a single word. It can be a piece of a word, a punctuation mark, and so on. Also, you should not equate this input vector directly with the embeddings used for document search. Search may add context processing, aggregation, and training aimed specifically at retrieval.

> **Takeaway:** A token ID is an address, and an embedding vector is the learned representation fetched from that address.

**Check question:** If the same token appears in two sentences, is its vector the same in every layer?

<details>
<summary>Answer</summary>

The token embedding from the initial lookup can be the same, but the later representations can differ as position information and context are taken into account.

</details>

---

<a id="q24"></a>
## Q24. Why does attention make separate Q, K, and V?

Things become more flexible if you separate three things: the criterion for what information to look for, the criterion for showing what information I hold, and the content that actually gets passed along. Q is the query, K is the key, and V is the value. These names are metaphors that describe roles; in reality, they are vectors made with learned matrices.

```math
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
```

**In plain words:** Transform the same input representation X with three different tables of mixing ratios. Q is used on the side that compares to decide where to look, K on the side being compared against, and V on the side of the content actually fetched. Even with the same input, you can make different features suited to each role.

### Reading the symbols

- **X** (the matrix that collects the token representations): Each row is the vector for one token. This formula is the basic form of self-attention, which makes three representations from the same input.
- $W_Q, W_K, W_V$ ("W sub Q", "W sub K", "W sub V": three transformation matrices): The subscripts Q, K, and V under W are names that tell the roles apart. They do not mean you multiply by the subscript letter. What gets learned is the numbers inside the matrices.
- **Q, K, V** (query, key, value): Q and K are used to make scores for comparing tokens with each other, and V is mixed using the weights derived from those scores.
- **XW** ("X times W", a matrix product): As in [Q07](02-math-intuition.md#q07), this is a computation that mixes the input features according to ratios. The three formulas are not "plug the result of the first into the second"; each one transforms the same X separately.

### Step by step

1. Say one row of the input is [1, 2]. If the ratios for Q are set as the column [1, 0], you get 1×1 + 2×0 = 1.
2. With ratios [0, 1] for K you get 2, and with ratios [1, 1] for V you get 3. Different information was pulled from the same input.
3. Compare the current token's Q with the K of the candidates it might attend to, and use the resulting weights to fetch the candidates' V. It is a structure that separates the representation used for comparison from the representation that gets passed along.

### Following the numbers

Say a token's representation is [1, 2]. With simple projections, if Q reads only the first entry it gets 1, if K reads only the second entry it gets 2, and if V reads the sum of both entries it gets 3. Even from the same input, different features can be pulled out depending on the purpose.

### Where it shows up in models

In self-attention, all three come from the same input X. In cross-attention, the input used to make Q can differ from the input used to make K and V. For example, you can set things up so that the representation being generated looks up information from an image or from another sentence.

### What changes if you tweak it

In a single attention computation, if you hold Q and K fixed and change only V, the weights stay the same, but the content that gets mixed in changes. If you hold V fixed and change Q or K, the scores and weights can change. "Where to look, and how much" and "what to fetch" can be adjusted separately.

### Watch out

Q does not mean an actual question sentence, and K does not mean keywords attached by a person. They are all learned numerical representations. You could tie the three transformations together as one and the computation would still work, but you would lose the freedom to learn them differently.

> **Takeaway:** Q and K decide how much to attend, and V holds what gets fetched.

**Check question:** If the lookup scores stay the same but only V changes, does the attention output stay the same too?

<details>
<summary>Answer</summary>

No. The weights are the same, but the content actually being mixed is different, so the output changes.

</details>

_Further reading: [S5 · Vaswani et al., *Attention Is All You Need*](references.md)_

---

<a id="q25"></a>
## Q25. Why does attention mix several tokens instead of picking one?

The information you need may be split across several positions. A weighted sum brings that information in together, and because the weights are soft, it can pass gradients all the way back to the selection scores. This is the differentiable selection we saw in [Q16](04-how-neural-networks-learn.md#q16), now put to work for actually passing information along.

```math
\alpha_i = \mathrm{softmax}(s)_i, \qquad \mathbf{o} = \sum_i \alpha_i \mathbf{v}_i
```

**In plain words:** Turn the scores the current token gave each candidate into attention weights with softmax. Multiply each candidate's value vector by its weight, then add matching entries together. That gathers the information from several candidates into a single vector o.

### Reading the symbols

- **s** and **softmax(s)ᵢ** (the list of scores and the i-th weight): s is the full set of candidate scores. softmax is the "exponentiate, then divide by the total" computation from [Q16](04-how-neural-networks-learn.md#q16), and the trailing i means entry i of the result.
- **αᵢ** (alpha i, the attention weight): The weight allotted to each allowed candidate. The weights add up to 1. If the current token changes, the weight it gives each candidate can change too.
- $\mathbf{v}_i$ (bold v sub i, the value vector of the i-th candidate): The information, several entries long, that one candidate passes along. αᵢvᵢ multiplies every entry by the same weight αᵢ.
- $\mathbf{o} = \sum_i \alpha_i \mathbf{v}_i$ ("o equals the sum over i of alpha i times v i", the output made by adding up each candidate's share): This does not merge different entries into one number. First entries are added to first entries, second entries to second entries, producing a vector.

### Step by step

1. Multiply the weights [0.6, 0.3, 0.1] by the value vectors [10, 0], [0, 10], and [0, 0], respectively. You get [6, 0], [0, 3], and [0, 0].
2. Adding matching entries gives [6, 3]. The first candidate's first feature and the second candidate's second feature both came in.
3. If the scores change during training, α changes and the result can change too, which creates a path for gradients to reach the parameters that make the scores. It does two jobs at once: combining information and making a learnable selection.

### Following the numbers

With weights [0.6, 0.3, 0.1] and value vectors [10, 0], [0, 10], and [0, 0], the output is [6, 3]. Picking only the first candidate gives [10, 0], so the information held by the second candidate is not brought in.

### Where it shows up in models

This is used in Transformers when each token takes in information from other tokens to update its own representation. During training, gradients flow from the final loss through this weighted sum to both the Q·K score computation and the V content computation.

### What changes if you tweak it

Keep the values the same and change the weights from [0.6, 0.3, 0.1] to [0.8, 0.1, 0.1], and the output goes from [6, 3] to [8, 1]. The first candidate's share grew and the second candidate's share shrank. Picking only the top one gives [10, 0], and the second feature disappears.

### Watch out

Averaging can make some of the original information indistinguishable. Different combinations of candidates can also produce the same output. Multiple heads leave room to handle different combinations, but a single high weight alone cannot fully explain what caused the final answer.

> **Takeaway:** Attention's weighted sum brings in information from several places while also letting the attention weights be learned.

**Check question:** If every candidate has exactly the same V, does changing only the weights change the output?

<details>
<summary>Answer</summary>

No. Since it is a weighted sum whose weights add up to 1, the output stays the same. Being soft does not always mean a useful learning signal reaches the scores.

</details>

_Further reading: [S5 · Vaswani et al., *Attention Is All You Need*](references.md)_

---

<a id="q26"></a>
## Q26. Why divide attention scores by the square root of the dimension?

A dot product adds up the products of many entries. As the number of entries grows, the spread of the scores can grow too, and softmax can pile almost everything onto one candidate and become insensitive to changes. Dividing by the square root of the dimension is a design choice to soften this growth in scale.

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
```

**In plain words:** First, compare every query with every key to build a table of scores. Divide each score by the square root of the number of entries in a key to adjust its size. For each query's row of scores, make weights with softmax, then mix the value vectors using those weights.

### Reading the symbols

- **QKᵀ** (Q times K transpose): Kᵀ is K with its rows and columns swapped. One entry of this product is the dot product of one query and one key. The result is a table of scores between tokens.
- **dₖ** and **√dₖ** ("d sub k" and "root d sub k": the key dimension and its square root): dₖ is the number of entries in a key vector. If dₖ = 64, then √dₖ = 8. The subscript k is a label showing that this is the dimension on the key side.
- **softmax(score table)** (making weights row by row): It normalizes among the scores that one query gave to its candidates. It does not add up the whole score table at once and divide by that.
- **The final multiplication by V** (combining content using the weights): Multiplying one query's row of weights by V gives the weighted sum from [Q25](#q25). Attention( ) on the left side of the formula is the name for this whole computation.

### Step by step

1. Start from the inside by computing QKᵀ. Say one query's scores are [8, 0].
2. If dₖ = 64, divide by 8 to get [1, 0]. The softmax weights are about [0.731, 0.269].
3. Mix V using these weights. If you skip the division, the weights for [8, 0] are about [0.9997, 0.0003], heavily lopsided toward one side from the start.

### Following the numbers

With dₖ = 64 and scores [8, 0], applying softmax directly gives a first weight of about 0.9997. With [1, 0], after dividing by 8, it is about 0.731. This is an example of adjusting the size; it does not mean the scores always come out to 8 in every model.

### Where it shows up in models

This is used in scaled dot-product attention. Multiple heads run this computation in parallel with different projections, then combine and transform the results. People do not assign each head a fixed role such as grammar or meaning.

### What changes if you tweak it

With the same score table, a larger divisor shrinks the gaps between scores and flattens the weights. A smaller divisor makes them sharper. √dₖ is a reference value meant to soften the way the typical size of a dot product grows as the dimension grows; it is not a device for arbitrarily making the probabilities uniform.

### Watch out

Under the simple assumption that the components have mean 0 and variance 1 and are reasonably independent, the standard deviation of the dot product is proportional to √dₖ. Values in real training do not follow this assumption exactly. Multi-head attention runs this computation on several sets of representations, and dₖ is usually the key dimension of a single head.

> **Takeaway:** Dividing by √dₖ is a device for adjusting the scale of dot-product scores as the dimension grows.

**Check question:** If the key dimension grows from 64 to 256, by what factor does the divisor grow?

<details>
<summary>Answer</summary>

It goes from 8 to 16, so it doubles. The dimension grows fourfold, but its square root only doubles.

</details>

_Further reading: [S5 · Vaswani et al., *Attention Is All You Need*](references.md)_

---

<a id="q27"></a>
## Q27. How does a model tell order apart and keep out future information?

Attention, which compares only content, needs a separate position signal. Also, when training to predict the next token, the model must be kept from seeing future tokens, since they contain the answer. Position information and the causal mask solve different problems.

```math
\mathbf{x}_t = \mathbf{e}_t + \mathbf{p}_t, \qquad \alpha_{t,j} = \frac{e^{s_{t,j} + M_{t,j}}}{\sum_l e^{s_{t,l} + M_{t,l}}}
```

**In plain words:** The first formula says: add a position vector to the vector that represents the token's meaning. The second says: add a mask to each attention score, then turn the scores into weights. Allowed positions are kept, and future positions are made to get a weight of 0.

### Reading the symbols

- $\mathbf{e}_t, \mathbf{p}_t, \mathbf{x}_t$ (bold e, p, and x sub t: content, position, and the combined input): The subscript t is the position in the sentence. The bold e in the first formula is the token vector, a different symbol from e ≈ 2.718, the base of the exponent in the second formula.
- **sₜ,ⱼ** ("s sub t, j", the score position t gives to position j): t is the position currently gathering information, and j is the candidate position it may attend to. The comma separates the two indices.
- **Mₜ,ⱼ** ("M sub t, j", the mask value for that pair): Allowed spots get 0, and spots to block, such as the future, get −∞. −∞ (negative infinity) is mathematical notation for heading endlessly toward smaller values.
- $e^{s+M}$ ("e to the s plus M", the exponential of the score with the mask added): If M = 0, the score is kept as is. If M = −∞, read it as the limit in which the exponential becomes 0.
- **Σₗ** and **αₜ,ⱼ** ("sum over l" and "alpha sub t, j": the sum over candidates and the final weight): l is a temporary index that sweeps over all the candidate positions. Dividing one candidate's value in the numerator by the total in the denominator gives that candidate's weight.

### Step by step

1. The first formula is one way of putting in position information. Even the same token gets a different pₜ when t differs, so it can produce a different input representation.
2. In the second formula, compute sₜ,ⱼ + Mₜ,ⱼ first. For the second position, the first and second positions are usually allowed, and the third onward are blocked.
3. After exponentiating, split the weight among the allowed candidates. A blocked candidate is left with 0 in the numerator, so its weight is 0. There must be at least one position available to compare against.

### Following the numbers

If the input is [I, ate, apples], the representation at the second position looks at the first and second tokens and predicts the third. Letting it see the third token at this point would amount to reading the answer in advance. Position numbers also help tell where the same word appears.

### Where it shows up in models

This is used in causal attention for training and generation in GPT-style models. Adding position vectors is a representative example; there are also methods like RoPE that apply a rotation to Q and K to reflect relative position. Not every Transformer uses the same way of handling position.

### What changes if you tweak it

At the second position, say all three candidate scores are 0. Put a mask of −∞ on the third (the future), and the weights are [1/2, 1/2, 0]. Remove the mask, and they become [1/3, 1/3, 1/3], which lets the model see the next token's correct answer in advance during training.

### Watch out

Position representations deal with "where is it?", and the causal mask deals with "where is it allowed to look?" They are not the same mechanism. Adding position vectors as above is one design, and there are other approaches such as RoPE. The mask itself also provides some order structure, so you cannot flatly say that without position vectors the model knows nothing at all about order.

> **Takeaway:** Position tells the model where each token is, and the causal mask limits how far it may look.

**Check question:** If you have added position embeddings, does that make the mask that hides the future unnecessary?

<details>
<summary>Answer</summary>

No. Telling the model positions and preventing the answer from leaking are separate jobs.

</details>

_Further reading: [S5 · Vaswani et al., *Attention Is All You Need*](references.md)_

---

<a id="q28"></a>
## Q28. How does predicting one next token turn into generating sentences?

Look at what came before, predict the probabilities of the next piece, attach the chosen piece to what came before, and predict again: that is how a sentence keeps going. In training, the model practices next-token prediction at many positions in real sentences. One easy form of computation is applied over and over across many contexts.

```math
L = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{\lt t})
```

**In plain words:** At each position, show the model the earlier tokens and get the probability of the actual next token. Put a minus sign in front of the log of that probability to make a penalty. Add up the penalties across positions and divide by the number of positions to get the average loss.

### Reading the symbols

- **xₜ** and $x_{\lt t}$ ("x sub t" and "x before t": the next correct token and the context before it): xₜ is the actual token to predict now, and $x_{\lt t}$ is the tokens before it. The < marks the range "the earlier indices."
- $p_\theta(x_t \mid x_{\lt t})$ ("p theta of x t given x before t", the probability given the preceding context): The vertical bar | does not mean division; it means "under this condition." θ is the parameters of the model that computes the probability.
- **−log** ("minus log", turning the correct answer's probability into a penalty): This is the negative log from [Q10](03-classical-ml-and-evaluation.md#q10). The lower the probability given to the correct token, the bigger the penalty.
- **Σₜ₌₁ᵀ** and **1/T** ("sum from t = 1 to T" and "one over T": the sum over positions and the average): Compute with t set to each position being evaluated, then divide by the total number of positions, T. For the starting position in training, a start token or a given context can be used.

### Step by step

1. If the probability of "ate," which actually came after "I," is 0.5, the penalty at that position is about 0.693.
2. If the probability of "apples" after "I ate" is 0.25, the penalty is about 1.386. The average of the two is about 1.040.
3. This formula is the training objective. When generating, there is no correct answer, so you pick one token from the model's distribution, attach it to the context, and repeat the same next-token computation.

### Following the numbers

If the model gave "ate" a probability of 0.5 after the context "I," and gave "apples" a probability of 0.25 after the context "I ate," the average loss over the two positions is (0.693 + 1.386)/2 ≈ 1.040. In generation, the next probabilities are computed from a new context with the chosen token attached.

### Where it shows up in models

This is the basic pretraining objective of autoregressive language models. During training, the correct context already exists, so many positions can be computed at once under a causal mask. During generation, the next token does not exist yet, so it usually proceeds one step at a time, in order.

### What changes if you tweak it

Even with the model held fixed, changing the preceding context can change the next token's probabilities. If one token is chosen badly during generation, that token goes into the next context, so it affects the outputs that follow. This is the difference between training, which is given the correct context, and generation, which continues from the model's own output.

### Watch out

"Guess the next token" is a simple form of objective; it does not mean the model necessarily learns only simple abilities. By dealing with diverse contexts, it can learn many rules. However, this objective alone does not guarantee that facts are verified against the outside world or that the user's goals are achieved.

> **Takeaway:** Generating a sentence is the process of repeating next-token prediction, continuing from the model's own output.

**Check question:** If many positions are computed at the same time during training, doesn't the model get to see future answers?

<details>
<summary>Answer</summary>

If the causal mask is applied correctly, each position sees only the earlier part it is allowed to see. Computing in parallel is not the same as accessing future information.

</details>

---
