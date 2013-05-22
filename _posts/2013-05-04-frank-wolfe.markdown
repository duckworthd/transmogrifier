---
comments: true
layout: post
title: Frank-Wolfe Algorithm
subtitle: wut.

---

  In this post, we'll take a look at the [Frank-Wolfe Algorithm][frank_wolfe]
also known as the Conditional Gradient Method, an algorithm particularly suited
for solving problems with compact domains. Like the [Proximal
Gradient][proximal_gradient] and [Accelerated Proximal
Gradient][accelerated_proximal_gradient] algorithms, Frank-Wolfe requires we
exploit problem structure to quickly solve a mini-optimization problem. Our
reward for doing so is a converge rate of $O(1/\epsilon)$ and the potential for
*extremely sparse solutions*.

  Returning to my [valley-finding metaphor][gradient_descent], Frank-Wolfe is a
bit like this,

<div class="pseudocode">
  1. Look around you and see which way points the most downwards
  2. Walk as far as possible in that direction until you hit a wall
  3. Go back in the direction you started, stop part way along the path, then
     repeat.
</div>


How does it work?
=================

  Frank-Wolfe is designed to solve problems of the form,

$$
  \min_{x \in D} f(x)
$$

  where $D$ is compact and $f$ is differentiable. For example, in $R^n$ any
closed and bounded set is compact. The algorithm for Frank-Wolfe is then,

<div class="pseudocode">
  **Input**: Initial iterate $x^{(0)}$

  1. For $t = 0, 1, 2, \ldots$
    2. Let $s^{(t+1)} = \arg\min_{s \in D} \langle \nabla f(x^{(t)}), s \rangle$
    3. If $g(x) = \langle \nabla f(x^{(t)}), x - s^{(t+1)} \rangle \le \epsilon$, break
    4. Let $x^{(t+1)} = (1 - \alpha^{(t)}) x^{(t)} + \alpha^{(t)} s^{(t+1)}$
</div>

  The proof relies on $\alpha^{(t)} = 2 / (t+2)$, but line search works as
well.  The intuition for the algorithm is that at each iteration, we minimize
a linear approximation to $f$,

$$
  s^{(t+1)} = \arg\min_{s \in D} f(x^{(t)}) + \nabla f(x^{(t)})^T (s - x^{(t)})
$$

  then take a step in that direction. We can immediately see that if $D$
weren't compact, $s^{(t)}$ would go off to infinity.

  <a id="upper_bound"></a>
  **Upper Bound** One nice property of Frank-Wolfe is that it comes with its
own upper bound on $f(x^{(t)}) - f(x^{*})$ calculated during the course of
the algorithm. Recall the linear upper bound on $f$ due to convexity,

$$
\begin{align*}
  f(x^{*})
  & \ge f(x) + \nabla f(x)^T (x^{*} - x) \\
  f(x) - f(x^{*})
  & \le \nabla f(x)^T (x - x^{*}) \\
\end{align*}
$$

  Since,

$$
  s^{(t+1)}
  = \arg\min_{s} \nabla f(x^{(t)})^T s
  = \arg\max_{s} \nabla f(x^{(t)})^T (x^{(t)} - s)
$$
  we know that $\nabla f(x^{(t)})^T (x^{(t)} - x^{*}) \le \nabla f(x^{(t)})^T
(x^{(t)} - s^{(t+1)})$ and thus,

$$
  f(x) - f(x^{*}) \le \nabla f(x^{(t)})^T (x^{(t)} - s^{(t+1)})
$$

<a id="example"></a>

A Small Example
===============

  For this example, we'll minimize a simple univariate quadratic function
constrained to lie in an interval,

$$
  \min_{x \in [-1,2]} (x-0.5)^2 + 2x
$$

  Its derivative is given by $2(x-0.5) + 2$, and since we are dealing with real
numbers, the minimizers of the linear approximation must be either $-1$ or
$2$ if the gradient is positive or negative, respectively. We'll use a stepsize
of $\alpha^{(t)} = 2 / (t+2)$ as prescribed by the convergence proof in the
next section.


<div class="img-center">
  <img src="/assets/img/frank_wolfe/animation.gif"></img>
  <span class="caption">
    Frank-Wolfe in action. The red circle is the current value for
    $f(x^{(t)})$, and the green diamond is $f(x^{(t+1)})$. The dotted line is
    the linear approximation to $f$ at $x^{(t)}$. Notice that at each step,
    Frank-Wolfe stays closer and closer to $x^{(t)}$ when moving in the
    direction of $s^{(t+1)}$.
  </span>
</div>

<div class="img-center">
  <img src="/assets/img/frank_wolfe/convergence.png"></img>
  <span class="caption">
    This plot shows how quickly the objective function decreases as the
    number of iterations increases. Notice that it does not monotonically
    decrease, as with Gradient Descent.
  </span>
</div>

<div class="img-center">
  <img src="/assets/img/frank_wolfe/iterates.png"></img>
  <span class="caption">
    This plot shows the actual iterates and the objective function evaluated at
    those points. More red indicates a higher iteration number. Since
    Frank-Wolfe uses linear combinations of $s^{(t+1)}$ and $x^{(t)}$, it
    tends to "bounce around" a lot, especially in earlier iterations.
  </span>
</div>

<a id="proof"></a>

Why does it work?
=================

  We begin by making the two assumptions given earlier,

1. $f$ is convex, differentiable, and finite for all $x \in D$
2. $D$ is compact

  **Assumptions** First, notice that we never needed to assume that a solution
$x^{*}$ exists. This is because $D$ is compact and $f$ is finite, meaning $x$
cannot get bigger and bigger to make $f(x)$ arbitrarily small.

  Secondly, we never made a Lipschitz assumption on $f$ or its gradient. Since
$D$ is compact, we don't have to -- instead, we get the following for free.
Define $C_f$ as,

$$
  C_f = \max_{\substack{
                x,s \in D \\
                \alpha \in [0,1] \\
                y = x + \alpha (s-x)
              }}
          \frac{2}{\alpha^2} \left(
            f(y) - f(x) - \langle \nabla f(x), y - x \rangle
          \right)
$$

  This immediate implies the following upper bound on $f$ for all $x, y \in
D$ and $\alpha \in [0,1]$,

$$
  f(y) \le f(x) + \langle \nabla f(x), y-x \rangle + \frac{\alpha^2}{2} C_f
$$

  **Proof Outline** The proof for Frank-Wolfe is surprisingly simple. The idea
is to first upper bound $f(x^{(t+1)})$ in terms of $f(x^{(t)})$, $g(x^{(t)})$,
and $C_f$. We then transform this per-iteration bound into a bound on
$f(x^{(t)}) - f(x^{*})$ depending on $t$ using induction. That's it!

  **Step 1** Upper bound $f(x^{(t+1)})$. As usual, we'll denote $x^{+} \triangleq
x^{(t+1)}$, $x \triangleq x^{(t)}$, $s^{+} \triangleq s^{(t+1)}$, and $\alpha
\triangleq \alpha^{(t)}$. We begin by using the upper bound we just obtained for
$f$ in terms of $C_f$, substituting $x^{+} = (1 - \alpha) x + \alpha s^{+}$ and
then $g(x) = \nabla f(x)^T (x - s^{+})$,

$$
\begin{align*}
  f(x^{+}) 
  & \le f(x) + \nabla f(x)^T (x^{+} - x) + \frac{\alpha^2}{2} C_f \\
  & = f(x) + \nabla f(x)^T ( (1-\alpha) x + \alpha s^{+} - x ) + \frac{\alpha^2}{2} C_f \\
  & = f(x) + \nabla f(x)^T ( \alpha s^{+} - \alpha x ) + \frac{\alpha^2}{2} C_f \\
  & = f(x) - \alpha \nabla f(x)^T ( x - s^{+} ) + \frac{\alpha^2}{2} C_f \\
  & = f(x) - \alpha g(x) + \frac{\alpha^2}{2} C_f \\
\end{align*}
$$

  **Step 2** Use induction on $t$. First, recall the upper bound on $f(x) -
f(x^{*}) \le g(x)$ [we derived above](#upper_bound). Let's add $-f(x^{*})$ into
what we got from Step 1, then use the upper bound on $f(x) - f(x^{*})$ to get,

$$
\begin{align*}
  f(x^{+}) - f(x^{*})
  & = f(x) - f(x^{*}) - \alpha g(x) + \frac{\alpha^2}{2} C_f \\
  & \le f(x) - f(x^{*}) - \alpha ( f(x) - f(x^{*}) ) + \frac{\alpha^2}{2} C_f \\
  & = (1 - \alpha) (f(x) - f(x^{*})) + \frac{\alpha^2}{2} C_f \\
\end{align*}
$$

  Now, we employ induction on $t$ to show that,

$$
  f(x^{(t)}) - f(x^{*}) \le \frac{4 C_f / 2}{t+2}
$$

  We'll assume that the step size is $\alpha^{(t)} = \frac{2}{t+2}$, giving us
$\alpha^{(0)} = 2 / (0+2) = 1$ and the base case,

$$
\begin{align*}
  f(x^{(1)} - f(x^{*})
  & \le (1 - \alpha^{(0)}) ( f(x^{(0)}) - f(x^{*}) ) + \frac{\alpha^2}{2} C_f \\
  & = (1 - 1) ( f(x^{(0)}) - f(x^{*}) ) + \frac{1}{2} C_f \\
  & \le \frac{4 C_f / 2}{(0 + 1) + 2}
\end{align*}
$$

  Next, for the recursive case, we use the inductive assumption on $f(x) - f(x^{*})$, the definition of $\alpha^{(t)}$, and some algebra,

$$
\begin{align*}
  f(x^{+}) - f(x^{*})
  & \le (1 - \alpha) ( f(x) - f(x^{*}) ) + \frac{ \alpha^2}{2} C_f \\
  & \le \left(1 - \frac{2}{t+2} \right) \frac{4 C_f / 2}{t + 2} + \left( \frac{2}{t+2} \right)^2 C_f / 2 \\
  & \le \frac{4 C_f / 2}{t + 2} \left( 1 - \frac{2}{t+2} + \frac{1}{t+2} \right) \\
  & = \frac{4 C_f / 2}{t + 2} \left( \frac{t+1}{t+2} \right) \\
  & \le \frac{4 C_f / 2}{t + 2} \left( \frac{t+2}{t+3} \right) \\
  & = \frac{4 C_f / 2}{(t + 1) + 2} \\
\end{align*}
$$

  Thus, if we want an error tolerance of $\epsilon$, we need
$O(\frac{1}{\epsilon})$ iterations to find it. This matches the convergence
rate of Gradient Descent an Proximal Gradient Descent, but falls short of their
accelerated brethren.


When should I use it?
=====================

  Like Proximal Gradient, efficient use of Frank-Wolfe requires solving a
mini-optimization problem at each iteration. Unlike Proximal Gradient, however,
this mini-problem will lead to unbounded iterates if the input space is not
compact -- in other words, Frank-Wolfe cannot directly be applied when your
domain is all of $R^{n}$. However, there is a very special case wherein
Frank-Wolfe shines.

  <a id="sparsity"></a>
  **Sparsity** The primary reason machine learning researchers have recently
taken an interest in Frank-Wolfe is because in certain problems the iterates
$x^{(t)}$ will be extremely sparse.  Suppose that $D$ is a polyhedron defined
by a set of linear constraints. Then $s^{(t)}$ is a solution to a Linear
Program, meaning that each $s^{(t)}$ lies on one of the vertices of the
polyhedron. If these vertices have only a few non-zero entries, then $x^{(t)}$
will too, as $x^{(t)}$ is a linear combination of $s^{(1)} \ldots s^{(t)}$.
This is in direct contrast to gradient and proximal based methods, wherein
$x^{(t)}$ is the linear combination of a set of non-sparse *gradients*.

  **Atomic Norms** One particular case where Frank-Wolfe shines is when
minimizing $f(x)$ subject to $||x|| \le c$ where $|| \cdot ||$ is an "atomic
norm". We say that $||\cdot||$ is an atomic norm if $||x||$ is the smallest $t$
such that $x/t$ is in the convex hull of a finite set of points $\mathcal{A}$,
that is,

$$
  ||x|| = \inf \{ t : x \in t \, \text{Conv}(\mathcal{A}) \}
$$

  For example, $||x||_1$ is an atomic norm with $\mathcal{A}$ being the set of
all vectors with only one $+1$ or one $-1$ entry. In these cases, finding
$\arg\min_{||s|| \le c} \langle \nabla f(x), s \rangle$ is tantamount to
finding which element of $\mathcal{A}$ minimizes $\langle \nabla f(x), s
\rangle$ (since $\text{Conv}(\mathcal{A})$ defines a polyhedron). For a whole
lot more on Atomic Norms, see [this tome][chandrasekaranm2010] by
Chandrasekaranm et al.

Extensions
==========

  **Step Size** The proof above relied on a step size of $\alpha^{(t)} =
\frac{2}{t+2}$, but as usual [Line Search][line_search] can be applied to
accelerate convergence.

  **Approximate Linear Solutions** Though not stated in the proof above,
another cool point about Frank-Wolfe is that you don't actually need to solve
the linear mini-problem exactly, but you will still converge to the optimal
solution (albet at a slightly slower rate). In particular, assume that each
mini-problem can be solved approximately with additive error $\frac{\delta
C_f}{t+2}$ at iteration $t$,

$$
  \langle s^{(t+1)}, \nabla f(x^{(t)}) \rangle
  \le \min_{s} \langle s, \nabla f(x^{(t)}) \rangle + \frac{\delta C_f}{t+2}
$$

  then Frank-Wolfe's rate of convergence is

$$
  f(x^{(t)}) - f(x^{*}) \le \frac{2 C_f}{t+2} (1 + \delta)
$$

  The proof for this can be found in the supplement to [Jaggi's][jaggi2013]
excellent survey on Frank-Wolfe for machine learning.

Linear Invariance
=================

  Another cool fact about Frank-Wolfe is that it's *linearly invariant* -- that
is, if you rotate and scale the space, nothing changes about the convergence
rate. This is in direct contrast to many other methods which depend on the
[condition number][condition_number] of a function (for functions with
Hessians, this is the ratio between the largest and smallest eigenvalues,
$\sigma_{\max} / \sigma_{\min})$.

  Suppose we transform our input space with a surjective (that is, onto) linear
transformation $M: \hat{D} \rightarrow D$. Let's now try to solve the problem,

$$
  \min_{\hat{x} \in \hat{D}} \hat{f}(\hat{x}) = f(M \hat{x}) = f(x)
$$

  Let's look at the solution to the per-iteration mini-problem we need to solve
for Frank-Wolfe,

$$
\begin{align*}
  \min_{\hat{s} \in \hat{D}} \langle \nabla \hat{f}(\hat{x}), \hat{s} \rangle
  = \min_{\hat{s} \in \hat{D}} \langle M^T \nabla f( M \hat{x}), \hat{s} \rangle
  = \min_{\hat{s} \in \hat{D}} \langle \nabla f( x ), M \hat{s} \rangle
  = \min_{s \in D} \langle \nabla f( x ), s \rangle
\end{align*}
$$

  In other words, we will find the same $s$ if we solve in the original space,
or if we find $\hat{s}$ and then map it back to $s$. No matter how $M$ warps
the space, Frank-Wolfe will do the same thing. This also means that if there's
a linear transformation you can do to make the points of your polyhedron
sparse, you can do it with no penalty!

References
==========

  **Proof of Convergence, Linear Invariance** Pretty much everything in this
article comes from [Jaggi's][jaggi2013] fantastic article on Frank-Wolfe for
machine learning.

[frank_wolfe]: http://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
[proximal_gradient]: /blog/proximal-gradient.html
[accelerated_proximal_gradient]: /blog/accelerated-proximal-gradient.html
[gradient_descent]:/blog/gradient-descent.html
[line_search]: /blog/gradient-descent.html#line_search
[condition_number]: http://en.wikipedia.org/wiki/Condition_number
[chandrasekaranm2010]: http://pages.cs.wisc.edu/~brecht/papers/2010-crpw_inverse_problems.pdf
[jaggi2013]: http://jmlr.csail.mit.edu/proceedings/papers/v28/jaggi13-supp.pdf

Reference Implementation
========================

{% highlight python %}

def frank_wolfe(minisolver, gradient, alpha, x0, epsilon=1e-2):
  """Frank-Wolfe Algorithm

  Parameters
  ----------
  minisolver : function
      minisolver(x) = argmin_{s \in D} <x, s>
  gradient : function
      gradient(x) = gradient[f](x)
  alpha : function
      learning rate
  x0 : array
      initial value for x
  epsilon : float
      desired accuracy
  """
  xs = [x0]
  iteration = 0
  while True:
    x = xs[-1]
    g = gradient(x)
    s_next = minisolver(g)
    if g * (x - s_next) <= epsilon:
      break
    a = alpha(iteration=iteration, x=x, direction=s_next)
    x_next = (1 - a) * x + a * s_next
    xs.append(x_next)
    iteration += 1
  return xs


def default_learning_rate(iteration, **kwargs):
  return 2.0 / (iteration + 2.0)


if __name__ == '__main__':
  import os

  import numpy as np
  import pylab as pl
  import yannopt.plotting as plotting

  ### FRANK WOLFE ALGORITHM ###

  # problem definition
  function    = lambda x: (x - 0.5) ** 2 + 2 * x
  gradient    = lambda x: 2 * (x - 0.5) + 2
  minisolver  = lambda y: -1 if y > 0 else 2 # D = [-1, 2]
  x0 = 1.0

  # run gradient descent
  iterates = frank_wolfe(minisolver, gradient, default_learning_rate, x0)

  ### PLOTTING ###

  plotting.plot_iterates_vs_function(iterates, function,
                                     path='figures/iterates.png', y_star=0.0)
  plotting.plot_iteration_vs_function(iterates, function,
                                      path='figures/convergence.png', y_star=0.0)

  # make animation
  iterates = np.asarray(iterates)
  try:
    os.makedirs('figures/animation')
  except OSError:
    pass

  for t in range(len(iterates)-1):
    x = iterates[t]
    x_plus = iterates[t+1]
    s_plus = minisolver(gradient(x))

    f = function
    g = gradient
    f_hat = lambda y: f(x) + g(x) * (y - x)

    xmin, xmax = plotting.limits([-1, 2])
    ymin, ymax = -4, 8

    pl.plot(np.linspace(xmin ,xmax), function(np.linspace(xmin, xmax)), alpha=0.2)
    pl.xlim([xmin, xmax])
    pl.ylim([ymin, ymax])
    pl.xlabel('x')
    pl.ylabel('f(x)')

    pl.plot([xmin, xmax], [f_hat(xmin), f_hat(xmax)], '--', alpha=0.2)
    pl.vlines([-1, 2], ymin, ymax, color=np.ones((2,3)) * 0.2, linestyle='solid')
    pl.scatter(x, f(x), c=[0.8, 0.0, 0.0], marker='o', alpha=0.8)
    pl.scatter(x_plus, f(x_plus), c=[0.0, 0.8, 0.0], marker='D', alpha=0.8)
    pl.vlines(x_plus, f_hat(x_plus), f(x_plus), color=[0.0,0.8,0.0], linestyle='dotted')
    pl.scatter(s_plus, f_hat(s_plus), c=0.35, marker='x', alpha=0.8)

    pl.savefig('figures/animation/%02d.png' % t)
    pl.close()


{% endhighlight %}
