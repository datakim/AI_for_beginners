# Before You Read the Formulas

[Contents](README.md) · [Part 1 →](01-models-and-learning.md)

A formula is a set of calculation instructions written in compressed form. You don't need to memorize unfamiliar symbols all at once. First look for three things: what goes in, what gets calculated, and what comes out.

## Calculate the right side, then name it on the left

When the result sits on the left, as in ŷ = f<sub>θ</sub>(x), it is easiest to follow the calculation on the right first. The equals sign means both sides have the same value. A formula is not literally the order in which code runs, but you can read it as the procedure that produces the result.

## From the inside of the parentheses outward

(ŷ − y)² means: find the difference first, then square it. 1/(1 + e<sup>−z</sup>) means: compute the whole denominator first, then divide 1 by it. The parentheses in f(x) mean "feed the input into the function". They do not mean f times x.

## Small marks above, below, and beside a symbol play different roles

**x₂ versus x².** x₂ is the value in the second slot, or of the second example. x² is x multiplied by itself. Numbers below a symbol usually pick out *which* thing; numbers above usually tell you to *do* an operation.

**Σ and averages.** Σ adds up terms you have calculated many times. A 1/N in front divides that total by N and turns it into an average. Check the starting index below and the ending index above to see what is being repeated.

**Fractions.** The numerator on top is the amount being divided; the denominator below is what you divide by. "Part ÷ whole" is a share, and "change ÷ amount moved" is a rate of change. Two fractions can look the same and still mean different things, depending on what the text around them says.

## The same symbol can mean different things in different places

T can mean a transpose, a number of tokens, or a temperature. ε can be a tiny constant added for numerical stability, or it can be noise. This guide redefines the symbols in every question. Don't carry a meaning over from an earlier question just because the symbol looks the same.

On your first read you don't need to compute derivatives or logarithms by hand. If you can say which way the result moves when a value gets bigger, you are ready to move on.
