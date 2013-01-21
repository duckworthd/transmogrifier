---
comments: true
layout: post
title: Beginnings
subtitle: Where everything starts over...

---

This is the start of it all.

Some example $\LaTeX$,

$$
\begin{align}
  f(x) &= x f(x-1)    \\
  g(x) &= x^2 g \left( \frac{x}{2} \right)
\end{align}
$$

And now some example code,

{% highlight python linenos %}
def f(x):
  if x == 0:
    return x * f(x-1)
  else:
    return 1

def g(x):
  if x <= 0:
    return 1
  else:
    return (x ** 2) * g(x/2)
{% endhighlight %}

# Title

## Secondary Title

### Trinary Title

#### Links

Isn't [Google](http://www.google.com) great?

#### Lists

* item 1
* item 2
* item 3

#### Images

![glyphicons](/assets/img/glyphicons-halflings.png)
