# oco_project
We want to classify handwritten digits using a Linear
Support Vector Machine (SVM), dealing with two categories: $\{-1, 1\}$
where 1 represents the digit 0 and -1 all the other digits.

Whe have 28x28 pixels images that we represent as vectors of
$\mathbb{R}^{784}$. We note $a_i \in \mathbb{R}^{785}$ an image and
it's intercept, $b_i$ its category.

Mathematically, we want to find
$x \in \mathbb{R}^{785}$ that minimizes the following soft margin
problem:

Given $n$ images and labels $(a_i, b_i)_{1\leq i \leq n}$

$$ \underset{x\in \mathbb{R}^{785}}{\min}f(x) :=\{  \frac 1n \underset{1\leq i \leq n}{\sum}l_{a_i, b_i}(x) +\frac \lambda 2 ||x||^2 \} $$

where
$l_{a,b}(x) = \text{hinge}(b \cdot x^T a) = \max ( 0, 1-b\cdot x^Ta )$.

We compared several onlinve convex optimization algorithms such as Online Gradient Descent, Stochastic Mirror Descent and Online Newton step for this problem. 

The report can be find [here](https://github.com/Ferdinand-Genans/oco_project/blob/main/Report_OCO_project.pdf).
