---
comments: true
layout: post
title: Why does \(L_1\) regularization produce sparse solution?
subtitle: wut.

---

  Supervised machine learning problems are typically of the form "minimize your
error while regularizing your parameters." The idea is that while many choices
of parameters may make your training error low, the goal isn't low training
error -- it's low test-time error. Thus, parameters should be minimize training
error while remaining "simple," where the definition of "simple" is left up to
the regularization function. Typically, supervised learning can be phrased as
minimizing the following objective function,

$$
  w^{*} = \arg\min_{w} \sum_{i} L(y_i, f(x_i; w)) + \lambda \Omega(w)
$$

  where $L(y_i, f(x_i; w))$ is the loss for predicting $f(x_i; w)$ when the
true label is for sample $i$ is $y_i$ and $\Omega(w)$ is a regularization
function.

Sparsifying Regularizers
========================

  There are many choices for $\Omega(w)$, but the ones I'm going to talk about
today are so called "sparsifying regularizers" such as $||w||_1$. These norms
are most often employed because they produce "sparse" $w^{*}$ -- that is,
$w^{*}$ with many zeros. This is in stark contrast to other regularizers such
as $\frac{1}{2}||w||_2^2$ which leads to lots of small but nonzero entries in
$w^{*}$.

Why Sparse Solutions?
=====================

  **Feature Selection** One of the key reasons people turn to sparsifying
regularizers is that they lead to automatic feature selection. Quite often,
many of the entries of $x_i$ are irrelevant or uninformative to predicting
the output $y_i$. Minimizing the objective function using these extra
features will lead to lower training error, but when the learned $w^{*}$ is
employed at test-time it will depend on these features to be more informative
than they are. By employing a sparsifying regularizer, the hope is that these
features will automatically be eliminated.

  **Interpretability** A second reason for favoring sparse solutions is that
the model is easier to interpret. For example, a simple sentiment classifier
might use a binary vector where an entry is 1 if a word is present and 0
otherwise. If the resulting learned weights $w^{*}$ has only a few non-zero
entries, we might believe that those are the most indicative words in deciding
sentiment.

How does it work?
=================

  We now come to the 100 million question: why do regularizers like the 1-norm
lead to sparse solutions? At some point someone probably told you "they're our
best convex approximation to $\ell_0$ norm," but there's a better reason than
that.  In fact, I claim that any regularizer that is non-differentiable at $w_i
= 0$ can lead to sparse solutions.

  **Intuition** The intuition lies in the idea of subgradients. Recall that the
subgradient of a (convex) function $\Omega$ at $x$ is any vector $v$ such that,

$$
  \Omega(y) \ge \Omega(x) + v^T (y-x)
$$

  The set of all subgradients for $\Omega$ at $x$ is called the subdifferential
and is denoted $\partial \Omega(x)$. If $\Omega$ is differentiable at $x$,
then $\partial \Omega(x) = \{ \nabla \Omega(x) \}$ -- in other words,
$\partial \Omega(x)$ contains 1 vector, the gradient. Where the
subdifferential begins to matter is when $\Omega$ *isn't* differentiable at
$x$. Then, it becomes something more interesting.

  Suppose we want to minimize an unconstrained objective like the following,

$$
  \min_{x} f(x) + \lambda \Omega(x)
$$

  By the [KKT conditions][kkt_conditions], 0 must be in the subdifferential at
the minimizer $x^{*}$,

$$
\begin{align*}
  0 & \in \nabla f(x^{*}) + \partial \lambda \Omega(x^{*}) \\
  - \frac{1}{\lambda} \nabla f(x^{*}) & \in \partial \Omega(x^{*}) \\
\end{align*}
$$

  Looking forward, we're particularly interested in when the previous
inequality holds when $x^{*} = 0$. What conditions are necessary for this to be
true?

  **Dual Norms** Since we're primarily concerned with $\Omega(x) = ||x||_1$,
let's plug that in. In the following, it'll actually be easier to prove things
about any norm, so we'll drop the 1 from here on out.

  Recal the definition of a dual norm. In particular, the dual norm of a norm
$||\cdot||$ is defined as,

$$
  ||y||_{*} = \sup_{||x|| \le 1} x^{T} y
$$

  A cool fact is that the dual of a dual norm is the original norm. In other words,

$$
  ||x|| = \sup_{||y||_{*} \le 1} y^{T} x
$$

  Let's take the gradient of the previous expression on both sides. A nice fact
to keep in mind is that if we take the gradient of an expression of the form
$\sup_{y} g(y, x)$, then its gradient with respect to x is $\nabla_x g(y^{*},
x)$ where $y^{*}$ is any $y$ that achieves the $\sup$. Since $g(y, x) = y^{T}
x$, that means $\nabla_x g(y, x) = y^{*}$

$$
  \partial ||x|| = \{ y^{*} :  y^{*} = \arg\max_{||y||_{*} \le 1} y^{T} x \}
$$

  Now let $x = 0$. Then $y^{T} x = 0$ for all $y$, so any $y$ with $||y||_{*}
\le 1$ is in $\partial ||x||$ for $x = 0$.

  Back to our original goal, recall that

$$
  -\frac{1}{\lambda} \nabla f(x) \in \partial ||x||
$$

  If $||-\frac{1}{\lambda} \nabla f(x)||_{*} \le 1$, then we've already
established that $-\frac{1}{\lambda} \nabla f(x)$ is in $\partial ||x||$ for $x
= 0$. In other words, $x^{*} = 0$ solves the original problem!

Conclusion
==========

  In the previous section, we showed that in order to solve the problem
$\min_{x} f(x) + \lambda \Omega(x)$, it is necessary that $-\frac{1}{\lambda}
\nabla f(x^{*}) \in \partial \Omega(x^{*})$. If $\Omega(x^{*})$ is
differentiable at $x^{*}$, then there can be only 1 possible choice for
$x^{*}$, but in all other cases there are a multitude of potential solutions.
When we are particularly concerned with sparse solutions and when $\Omega(x)$
isn't differentiable at $x = 0$, there is a set of values which
$-\frac{1}{\lambda} \nabla f(x^{*})$ can take on such that $x^{*} = 0$ is still
an optimal solution. This is why $\Omega(x) = ||x||_1$ leads to sparsification.

References
==========

  Everything written here was explained to me by the ever-knowledgable
MetaOptimize king, [Alexandre Passos][atpassos].

[kkt_conditions]: http://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions
[atpassos]: https://twitter.com/atpassos
