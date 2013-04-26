---
comments: true
layout: post
title: Why does \(L_1\) regularization produce sparse solutions?
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

Non-smooth Regularizers and their Solutions
===========================================

  We now come to the \$ 100 million question: why do regularizers like the 1-norm
lead to sparse solutions? At some point someone probably told you "they're our
best convex approximation to $\ell_0$ norm," but there's a better reason than
that.  In fact, I claim that any regularizer that is non-differentiable at $w_i
= 0$ and can be decomposed into a sum can lead to sparse solutions.

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
about any norm, so we'll drop the 1 for the rest of this section.

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
x$, that means,

$$
  \nabla_x \sup_{y} g(y, x) = \nabla_x \left( (y^{*})^T x \right) = y^{*} = \arg\max_{||y||_{*} \le 1} y^{T} x
$$

$$
  \partial ||x|| = \{ y^{*} :  y^{*} = \arg\max_{||y||_{*} \le 1} y^{T} x \}
$$

  Now let $x = 0$. Then $y^{T} x = 0$ for all $y$, so any $y$ with $||y||_{*}
\le 1$ is in $\partial ||x||$ for $x = 0$.

  Back to our original goal, recall that

$$
  -\frac{1}{\lambda} \nabla f(x) \in \partial ||x||
$$

  If $||-\frac{1}{\lambda} \nabla f(x)||_{*} \le 1$ when $x=0, then we've already
established that $-\frac{1}{\lambda} \nabla f(0)$ is in $\partial ||0||$. In
other words, $x^{*} = 0$ solves the original problem!

Onto Coordinate-wise Sparsity
=============================

  We've just established that $||\frac{1}{\lambda} \nabla f(0)||_{*} \le 1$
implies $x^{*} = 0$, but we don't want all of $x^{*}$ to be 0, we want *some
coordinates* of $x^{*}$ to be 0. How can we take what we just concluded and
apply it only a subvector of $x^{*}$?

  Rather than a general norm, let's return once again to the $L_1$ norm. The
$L_1$ norm has a very special property that will be of use here:
separability. In words, this means that the $L_1$ norm can be expressed as a
sum of functions over $x$'s individual coordinates, each independent of every
other. In particular, $||x||_1 = \sum_{i} |x_{i}|$.  It's easy to see that the
function $\Omega_i(x) = |x_i|$ is independent of the rest of $x$'s elements.

  Let's take another look at our objective function,

$$
\begin{align*}
  \min_{x} f(x) + \lambda ||x||_1
  & = \min_{x_i} \left( \min_{x_{-i}} f(x_i, x_{-i}) + \lambda \sum_{j} |x_j| \right) \\
  & = \min_{x_i} g(x_i) + \lambda |x_i|
\end{align*}
$$

  where $x_{-i}$ is all coordinates of $x$ except $x_i$ and $g(x_i) =
\min_{x_{-i}} f(x_i, x_{-i}) + \lambda \sum_{j \ne i} |x_j|$. Taking the
derivative of $g(x_i)$ with respect to $x_i$, we again require that,

$$
\begin{align*}
  0 &\in \nabla_{x_i} g(x_i) + \lambda \partial |x_i| \\
  -\frac{1}{\lambda} \nabla_{x_i} g(x_i) & \in \partial |x_i| \\
  -\frac{1}{\lambda} \nabla_{x_i} f(x_i, x_{-i}^{*}) & \in \partial |x_i|
\end{align*}
$$

  Hmm, that looks familiar. And isn't $|x_i| = ||x_i||_1$? That means that if

$$
  \left| \left| \frac{1}{\lambda} \nabla_{x_i} f(x_i, x_{-i}^{*}) \right| \right|_{\infty}
  = \left| \frac{1}{\lambda} \nabla_{x_i} f(x_i, x_{-i}^{*}) \right| \le 1
$$

  when $x_i = 0$, then $x_i^{*} = 0$. In other words, given the optimal values
for all coordinates other than $i$, we can evaluate the derivative of
$\frac{1}{\lambda} f$ with respect to $x_i$ and check if the absolute value
of that is less than 1. If it is, then $x_i = 0$ is optimal!

Conclusion
==========

  In the first section, we showed that in order to solve the problem
$\min_{x} f(x) + \lambda \Omega(x)$, it is necessary that $-\frac{1}{\lambda}
\nabla f(x^{*}) \in \partial \Omega(x^{*})$. If $\Omega(x^{*})$ is
differentiable at $x^{*}$, then there can be only 1 possible choice for
$x^{*}$, but in all other cases there are a multitude of potential solutions.
When $\Omega(x)$ isn't differentiable at $x = 0$, there is a non-singleton set
of values which $-\frac{1}{\lambda} \nabla f(x^{*})$ can be in such that $x^{*}
= 0$ is an optimal solution. If $\Omega(x) = ||x||$, then a sufficient
condition for $x^{*} = 0$ to be optimal is $||\frac{1}{\lambda} \nabla
f(x)||_{*} \le 1$ at $x = 0$.

  In the next section, we showed that in the special case of the $L_1$ norm, we
can express the norm as the sum of $L_1$ norms applied to $x$'s individual
coordinates. Because of this, we can rewrite the original optimization problem
as $\min_{x_i} g(x_i) + \lambda ||x_i||_1$ where $g(x_i) = \min_{x_{-i}} f(x_i,
x_{-i}) + \lambda ||x_{-i}||_1$. Using the same results from the previous
section, we showed that as long as $|\frac{1}{\lambda} \nabla_{x_i} f(x_i,
x_{-i}^{*})| \le 1$ when $x_i = 0$, then $x_i^{*} = 0$ is an optimal choice. In
other words, we established conditions upon which a coordinate will be 0. This
is why the $L_1$ norm causes sparsity.

References
==========

  Everything written here was explained to me by the ever-knowledgable
MetaOptimize king, [Alexandre Passos][atpassos].

[kkt_conditions]: http://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions
[atpassos]: https://twitter.com/atpassos
