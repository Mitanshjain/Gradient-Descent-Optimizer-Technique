# Gradient Descent Optimizer 

> A step-by-step introduction to Gradient Descent: intuition,
> mathematics, derivatives, learning rate, parameter updates,
> convergence, common mistakes, variants, and Python implementation.

------------------------------------------------------------------------

## Table of Contents

1.  [What is Gradient Descent?](#1-what-is-gradient-descent)
2.  [Why Do We Need Gradient
    Descent?](#2-why-do-we-need-gradient-descent)
3.  [Important Terms](#3-important-terms)
4.  [Starting with the Equation of a Straight
    Line](#4-starting-with-the-equation-of-a-straight-line)
5.  [Actual Value vs Predicted
    Value](#5-actual-value-vs-predicted-value)
6.  [Loss Function and Mean Squared
    Error](#6-loss-function-and-mean-squared-error)
7.  [Substituting the Prediction Equation into the
    Loss](#7-substituting-the-prediction-equation-into-the-loss)
8.  [Why Do We Differentiate the
    Loss?](#8-why-do-we-differentiate-the-loss)
9.  [Derivative with Respect to the Intercept
    b](#9-derivative-with-respect-to-the-intercept-b)
10. [Derivative with Respect to the Weight
    m](#10-derivative-with-respect-to-the-weight-m)
11. [Gradient Descent Update Rule](#11-gradient-descent-update-rule)
12. [What is the Learning Rate?](#12-what-is-the-learning-rate)
13. [Worked Numerical Example](#13-worked-numerical-example)
14. [Gradient Descent for Multiple Data
    Points](#14-gradient-descent-for-multiple-data-points)
15. [How Gradient Descent Actually
    Learns](#15-how-gradient-descent-actually-learns)
16. [Convergence](#16-convergence)
17. [Learning Rate Problems](#17-learning-rate-problems)
18. [Batch, Stochastic, and Mini-Batch Gradient
    Descent](#18-batch-stochastic-and-mini-batch-gradient-descent)
19. [Gradient Descent Algorithm](#19-gradient-descent-algorithm)
20. [Python Implementation from
    Scratch](#20-python-implementation-from-scratch)
21. [Visual Intuition](#21-visual-intuition)
22. [Common Beginner Mistakes](#22-common-beginner-mistakes)
23. [Gradient Descent vs
    Backpropagation](#23-gradient-descent-vs-backpropagation)
24. [Important Equations Cheat
    Sheet](#24-important-equations-cheat-sheet)
25. [Interview Questions](#25-interview-questions)
26. [Final Summary](#26-final-summary)

------------------------------------------------------------------------

# 1. What is Gradient Descent?

**Gradient Descent is an optimization algorithm used to minimize a
loss/cost function by repeatedly updating model parameters in the
direction that reduces the loss.**

In simple words:

> The model makes a prediction → we calculate how wrong it is → Gradient
> Descent determines how the parameters should change → the parameters
> are updated → the model predicts again.

This process repeats until the loss becomes sufficiently small or
training reaches a stopping condition.

Gradient Descent is one of the most important optimization concepts in
Machine Learning and Deep Learning.

It is used in models such as:

-   Linear Regression
-   Logistic Regression
-   Neural Networks
-   Deep Learning models
-   Many other optimization-based ML algorithms

------------------------------------------------------------------------

# 2. Why Do We Need Gradient Descent?

Suppose we have a Linear Regression model:

``` text
y = mx + b
```

Different values of `m` and `b` create different lines.

Some lines fit the training data badly.

Some fit it better.

Our objective is to find values of `m` and `b` that produce the
**smallest possible error**.

Gradient Descent helps us search for those values.

``` text
Initial m, b
    ↓
Make predictions
    ↓
Calculate loss
    ↓
Calculate gradients
    ↓
Update m, b
    ↓
Make better predictions
    ↓
Repeat
```

------------------------------------------------------------------------

# 3. Important Terms

Before learning the mathematics, understand these terms.

  Term              Meaning
  ----------------- --------------------------------------------------
  `x`               Input feature
  `y`               Actual/target value
  `ŷ` or `y_pred`   Predicted value
  `m`               Weight/slope
  `b`               Bias/intercept
  Loss              Measurement of prediction error
  Gradient          Slope/derivative of the loss
  Learning Rate     Size of each parameter update
  Epoch             One complete pass through the training dataset
  Optimization      Process of finding parameters that minimize loss

### Important correction

In the equation:

``` text
y = mx + b
```

-   `m` is the **slope/weight**.
-   `b` is the **y-intercept/bias**.

`b` is **not the minimum point**. It tells us where the line crosses the
y-axis when `x = 0`.

------------------------------------------------------------------------

# 4. Starting with the Equation of a Straight Line

The basic equation is:

$$
y = mx + b
$$

For a machine-learning prediction, we usually write:

$$
\hat{y} = mx + b
$$

where:

-   $x$ = input
-   $m$ = weight/slope
-   $b$ = intercept/bias
-   $\hat{y}$ = predicted value

For example:

``` text
m = 2
x = 5
b = 3
```

Then:

$$
\hat{y} = 2(5) + 3 = 13
$$

So the model predicts `13`.

------------------------------------------------------------------------

# 5. Actual Value vs Predicted Value

Suppose:

``` text
Actual value    = 15
Predicted value = 13
```

The prediction error can be represented as:

$$
y-\hat{y}
$$

Therefore:

$$
15-13=2
$$

But simply adding raw errors is problematic because positive and
negative errors can cancel each other.

For this reason, regression commonly uses squared errors.

------------------------------------------------------------------------

# 6. Loss Function and Mean Squared Error

A common loss function for regression is **Mean Squared Error (MSE)**.

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

where:

-   $n$ = **number of observations/data points**
-   $y_i$ = actual value of observation $i$
-   $\hat{y}_i$ = predicted value
-   $(y_i-\hat{y}_i)^2$ = squared error

> **Important:** In MSE, `n` represents the number of
> samples/observations, not the number of features.

For one training example:

$$
L=(y-\hat{y})^2
$$

This is the squared-error loss for that example.

### Why square the error?

Squaring:

1.  makes negative and positive errors both positive,
2.  penalizes larger errors more heavily,
3.  produces a differentiable function that works conveniently with
    Gradient Descent.

------------------------------------------------------------------------

# 7. Substituting the Prediction Equation into the Loss

We know:

$$
\hat{y}=mx+b
$$

and for one sample:

$$
L=(y-\hat{y})^2
$$

Substitute the prediction:

$$
L=(y-(mx+b))^2
$$

Therefore:

$$
\boxed{L=(y-mx-b)^2}
$$

This equation tells us that the loss depends on the parameters `m` and
`b`.

Our goal is:

$$
\boxed{\text{Find }m\text{ and }b\text{ that minimize }L}
$$

------------------------------------------------------------------------

# 8. Why Do We Differentiate the Loss?

Imagine the loss function as a hill or bowl.

Gradient Descent wants to move toward the lowest point.

The derivative tells us:

-   the **direction** in which the loss changes,
-   and how **steeply** it changes.

For a parameter $\theta$:

$$
\frac{\partial L}{\partial \theta}
$$

is the gradient of the loss with respect to that parameter.

If the gradient is positive, decreasing the parameter can reduce the
loss.

If the gradient is negative, increasing the parameter can reduce the
loss.

This is why Gradient Descent updates parameters in the **opposite
direction of the gradient**.

------------------------------------------------------------------------

# 9. Derivative with Respect to the Intercept b

For one sample:

$$
L=(y-mx-b)^2
$$

We want:

$$
\frac{\partial L}{\partial b}
$$

Using the chain rule:

$$
\frac{\partial L}{\partial b}
=
2(y-mx-b)
\frac{\partial(y-mx-b)}{\partial b}
$$

Now:

$$
\frac{\partial y}{\partial b}=0
$$

$$
\frac{\partial(mx)}{\partial b}=0
$$

$$
\frac{\partial b}{\partial b}=1
$$

Therefore:

$$
\frac{\partial(y-mx-b)}{\partial b}=-1
$$

So:

$$
\boxed{
\frac{\partial L}{\partial b}
=
-2(y-mx-b)
}
$$

Since:

$$
\hat{y}=mx+b
$$

we can also write:

$$
\boxed{
\frac{\partial L}{\partial b}
=
-2(y-\hat{y})
}
$$

------------------------------------------------------------------------

# 10. Derivative with Respect to the Weight m

Again:

$$
L=(y-mx-b)^2
$$

Differentiate with respect to `m`:

$$
\frac{\partial L}{\partial m}
=
2(y-mx-b)
\frac{\partial(y-mx-b)}{\partial m}
$$

Now:

$$
\frac{\partial y}{\partial m}=0
$$

$$
\frac{\partial(mx)}{\partial m}=x
$$

$$
\frac{\partial b}{\partial m}=0
$$

Therefore:

$$
\frac{\partial(y-mx-b)}{\partial m}=-x
$$

Hence:

$$
\boxed{
\frac{\partial L}{\partial m}
=
-2x(y-mx-b)
}
$$

or:

$$
\boxed{
\frac{\partial L}{\partial m}
=
-2x(y-\hat{y})
}
$$

These derivatives tell Gradient Descent how `m` and `b` should change.

------------------------------------------------------------------------

# 11. Gradient Descent Update Rule

The general Gradient Descent formula is:

$$
\boxed{
\theta_{new}
=
\theta_{old}
-
\alpha\frac{\partial L}{\partial\theta}
}
$$

where:

-   $\theta$ = parameter being updated
-   $\alpha$ = learning rate
-   $\frac{\partial L}{\partial\theta}$ = gradient

For the intercept:

$$
\boxed{
b_{new}
=
b_{old}
-
\alpha\frac{\partial L}{\partial b}
}
$$

For the weight:

$$
\boxed{
m_{new}
=
m_{old}
-
\alpha\frac{\partial L}{\partial m}
}
$$

The minus sign is extremely important.

We subtract the gradient because the gradient points toward the
direction of **increasing** loss. Gradient Descent wants to move toward
**decreasing** loss.

------------------------------------------------------------------------

# 12. What is the Learning Rate?

The **learning rate** controls how large each Gradient Descent step is.

It is commonly represented by:

$$
\alpha
$$

Example:

``` text
learning_rate = 0.001
```

Suppose:

``` text
b_old = 50
dL/db = 100
learning_rate = 0.001
```

Without a learning rate:

$$
b_{new}=50-100=-50
$$

That is an extremely large jump.

Using a learning rate:

$$
b_{new}
=
50-(0.001)(100)
$$

$$
b_{new}=50-0.1
$$

$$
\boxed{b_{new}=49.9}
$$

This produces a controlled update.

### Is `0.001` always the learning rate?

No.

`0.001` is only a common example. Depending on the problem, values such
as these may be tried:

``` text
0.1
0.01
0.001
0.0001
```

The best value depends on the model, data, optimizer, feature scale, and
training setup.

------------------------------------------------------------------------

# 13. Worked Numerical Example

Consider one training example:

``` text
x = 2
y = 10

Initial:
m = 3
b = 1

Learning rate = 0.01
```

### Step 1 --- Prediction

$$
\hat{y}=mx+b
$$

$$
\hat{y}=3(2)+1=7
$$

So:

``` text
Actual    = 10
Prediction = 7
```

### Step 2 --- Calculate loss

$$
L=(y-\hat{y})^2
$$

$$
L=(10-7)^2
$$

$$
\boxed{L=9}
$$

### Step 3 --- Gradient with respect to b

$$
\frac{\partial L}{\partial b}
=
-2(y-\hat{y})
$$

$$
=-2(10-7)
$$

$$
\boxed{\frac{\partial L}{\partial b}=-6}
$$

### Step 4 --- Gradient with respect to m

$$
\frac{\partial L}{\partial m}
=
-2x(y-\hat{y})
$$

$$
=-2(2)(10-7)
$$

$$
\boxed{\frac{\partial L}{\partial m}=-12}
$$

### Step 5 --- Update b

$$
b_{new}
=
b_{old}
-
\alpha\frac{\partial L}{\partial b}
$$

$$
b_{new}
=
1-(0.01)(-6)
$$

$$
\boxed{b_{new}=1.06}
$$

### Step 6 --- Update m

$$
m_{new}
=
m_{old}
-
\alpha\frac{\partial L}{\partial m}
$$

$$
m_{new}
=
3-(0.01)(-12)
$$

$$
\boxed{m_{new}=3.12}
$$

### Step 7 --- New line

The old model was:

$$
\hat{y}=3x+1
$$

The updated model becomes:

$$
\boxed{\hat{y}=3.12x+1.06}
$$

For `x = 2`:

$$
\hat{y}=3.12(2)+1.06=7.30
$$

The prediction moved:

``` text
7.00 → 7.30
```

toward the actual value:

``` text
10
```

That is Gradient Descent performing one optimization step.

------------------------------------------------------------------------

# 14. Gradient Descent for Multiple Data Points

Real datasets normally contain many observations.

For Linear Regression:

$$
\hat{y}_i=mx_i+b
$$

The MSE is:

$$
\boxed{
J(m,b)
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

Substituting the model:

$$
\boxed{
J(m,b)
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-(mx_i+b))^2
}
$$

The gradients become:

### Gradient for m

$$
\boxed{
\frac{\partial J}{\partial m}
=
-\frac{2}{n}
\sum_{i=1}^{n}
x_i(y_i-\hat{y}_i)
}
$$

### Gradient for b

$$
\boxed{
\frac{\partial J}{\partial b}
=
-\frac{2}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)
}
$$

Then update:

$$
m
\leftarrow
m-\alpha\frac{\partial J}{\partial m}
$$

$$
b
\leftarrow
b-\alpha\frac{\partial J}{\partial b}
$$

These steps repeat many times.

------------------------------------------------------------------------

# 15. How Gradient Descent Actually Learns

A simplified training loop is:

``` text
Choose initial m and b
        ↓
Calculate predictions
        ↓
Calculate loss
        ↓
Calculate dL/dm and dL/db
        ↓
Update m and b
        ↓
Calculate predictions again
        ↓
Calculate new loss
        ↓
Repeat
```

Ideally:

``` text
Epoch 1  → Loss = 100
Epoch 2  → Loss = 70
Epoch 3  → Loss = 45
Epoch 4  → Loss = 25
Epoch 5  → Loss = 12
...
```

The loss gradually approaches a minimum.

------------------------------------------------------------------------

# 16. Convergence

**Convergence** means Gradient Descent has reached a point where
additional updates produce little or no meaningful improvement in the
loss.

For example:

``` text
Loss:
10.0000
4.2000
1.5000
0.4000
0.1000
0.0910
0.0902
0.0901
```

The optimization is becoming stable around the minimum.

Possible stopping conditions include:

-   fixed number of epochs reached,
-   loss stops improving,
-   gradient becomes very small,
-   parameter updates become extremely small.

------------------------------------------------------------------------

# 17. Learning Rate Problems

The learning rate is one of the most important hyperparameters.

## Learning rate too large

If $\alpha$ is too large:

``` text
Minimum
   ↓
---\      /---
    \    /
     \  /
      \/
```

The optimizer can repeatedly jump across the minimum.

Possible effects:

-   oscillation,
-   unstable training,
-   loss increases,
-   divergence,
-   in extreme cases, very large gradients/values.

This is often described as **overshooting**.

## Learning rate too small

If $\alpha$ is too small:

-   updates are tiny,
-   training becomes very slow,
-   many iterations may be required.

### Important distinction

A very small learning rate is **not the same thing as the
vanishing-gradient problem**.

Vanishing gradients usually occur in deep neural networks when gradients
become extremely small as they are propagated backward through many
layers.

Likewise, a large learning rate can cause instability or divergence, but
**exploding gradients** specifically refers to gradients becoming
excessively large, commonly during backpropagation in deep/recurrent
networks.

So a safer beginner summary is:

``` text
Learning rate too large → overshooting / instability / possible divergence
Learning rate too small → very slow convergence
Good learning rate      → stable and reasonably fast convergence
```

------------------------------------------------------------------------

# 18. Batch, Stochastic, and Mini-Batch Gradient Descent

There are three commonly discussed forms.

## 1. Batch Gradient Descent

Uses the **entire training dataset** to calculate one gradient update.

``` text
All training samples
       ↓
Calculate gradient
       ↓
One update
```

### Advantages

-   stable gradient,
-   deterministic for a fixed dataset/model state.

### Disadvantages

-   expensive for very large datasets,
-   each update can be slow.

------------------------------------------------------------------------

## 2. Stochastic Gradient Descent (SGD)

Uses **one training example at a time**.

``` text
One sample
    ↓
Gradient
    ↓
Update
```

### Advantages

-   frequent updates,
-   can work well with very large/streaming datasets.

### Disadvantages

-   noisy updates,
-   loss may fluctuate rather than decrease smoothly.

------------------------------------------------------------------------

## 3. Mini-Batch Gradient Descent

Uses a **small batch of samples** per update.

Example:

``` text
Dataset = 10,000 samples
Batch size = 32
```

The model processes 32 samples, calculates gradients, updates the
parameters, and continues with the next batch.

This is the standard approach for training most modern neural networks
because it provides a useful balance between computational efficiency
and gradient stability.

------------------------------------------------------------------------

# 19. Gradient Descent Algorithm

A simple algorithm for Linear Regression:

``` text
1. Initialize m and b.
2. Choose learning rate α.
3. Choose number of epochs.
4. Repeat:
      a. Calculate predictions:
            y_pred = mx + b

      b. Calculate loss:
            MSE = mean((y - y_pred)^2)

      c. Calculate gradients:
            dm = (-2/n) Σ[x(y - y_pred)]
            db = (-2/n) Σ[y - y_pred]

      d. Update:
            m = m - α × dm
            b = b - α × db

5. Return the optimized m and b.
```

------------------------------------------------------------------------

# 20. Python Implementation from Scratch

``` python
import numpy as np

# Sample dataset
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Initial parameters
m = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
epochs = 1000
n = len(x)

for epoch in range(epochs):

    # Step 1: Prediction
    y_pred = m * x + b

    # Step 2: Calculate MSE
    loss = np.mean((y - y_pred) ** 2)

    # Step 3: Calculate gradients
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    # Step 4: Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

    # Print progress occasionally
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Loss: {loss:.6f}, "
            f"m: {m:.4f}, "
            f"b: {b:.4f}"
        )

print("\nFinal Parameters")
print("m =", m)
print("b =", b)
```

The dataset follows approximately:

$$
y=2x+1
$$

Therefore, after enough training, Gradient Descent should learn values
close to:

``` text
m ≈ 2
b ≈ 1
```

------------------------------------------------------------------------

# 21. Visual Intuition

Imagine standing on a mountain in fog.

You want to reach the lowest point, but you cannot see the whole
mountain.

You inspect the slope under your feet.

-   If the ground slopes upward to your right, move left.
-   If it slopes upward to your left, move right.
-   Repeat until the ground becomes nearly flat.

In Gradient Descent:

``` text
Mountain height = Loss
Your position   = Current parameter values
Slope           = Gradient
Step size       = Learning rate
Lowest point    = Minimum loss
```

Mathematically:

$$
\text{new position}
=
\text{old position}
-
\text{learning rate}\times\text{slope}
$$

That is the core idea behind Gradient Descent.

------------------------------------------------------------------------

# 22. Common Beginner Mistakes

### Mistake 1 --- Saying `n` in MSE is the number of features

Incorrect:

``` text
n = number of features
```

Correct:

``` text
n = number of observations/samples included in the average
```

------------------------------------------------------------------------

### Mistake 2 --- Saying `b` is the minimum point

Incorrect:

``` text
b = minimum point
```

Correct:

``` text
b = intercept/bias
```

The minimum belongs to the **loss function**, not to `b` itself.

------------------------------------------------------------------------

### Mistake 3 --- Confusing error and loss

For one prediction:

$$
Error=y-\hat{y}
$$

Squared error:

$$
(y-\hat{y})^2
$$

Dataset MSE:

$$
MSE=\frac{1}{n}\sum(y_i-\hat{y}_i)^2
$$

------------------------------------------------------------------------

### Mistake 4 --- Forgetting the learning rate

Incorrect:

$$
m_{new}=m_{old}-\frac{\partial L}{\partial m}
$$

Standard Gradient Descent:

$$
\boxed{
m_{new}
=
m_{old}
-
\alpha\frac{\partial L}{\partial m}
}
$$

------------------------------------------------------------------------

### Mistake 5 --- Thinking Gradient Descent directly changes predictions

Gradient Descent updates **model parameters**.

``` text
Gradient Descent
      ↓
changes m and b
      ↓
changes the model
      ↓
changes predictions
      ↓
reduces loss
```

------------------------------------------------------------------------

### Mistake 6 --- Calling every large-learning-rate problem "exploding gradient"

A learning rate that is too large usually causes **overshooting,
oscillation, or divergence**.

Exploding gradients are a related but separate deep-learning phenomenon
in which gradient values themselves become extremely large.

------------------------------------------------------------------------

### Mistake 7 --- Calling a small learning rate "vanishing gradient"

A small learning rate causes **small parameter updates and slow
learning**.

Vanishing gradient means the calculated gradients themselves become
extremely small, especially in deep neural networks.

------------------------------------------------------------------------

# 23. Gradient Descent vs Backpropagation

These terms are related but not identical.

### Backpropagation

Backpropagation calculates the gradients of the loss with respect to
neural-network parameters.

It answers:

> How much did each parameter contribute to the loss?

### Gradient Descent

Gradient Descent uses those gradients to update the parameters.

It answers:

> Given these gradients, how should the parameters be changed?

Simplified:

``` text
Forward Pass
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Optimizer / Gradient Descent
     ↓
Updated weights
```

------------------------------------------------------------------------

# 24. Important Equations Cheat Sheet

## Linear model

$$
\boxed{\hat{y}=mx+b}
$$

## Prediction error

$$
\boxed{e=y-\hat{y}}
$$

## Squared error for one observation

$$
\boxed{L=(y-\hat{y})^2}
$$

## Mean Squared Error

$$
\boxed{
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

## Linear Regression MSE

$$
\boxed{
J(m,b)=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-(mx_i+b))^2
}
$$

## Gradient with respect to m

$$
\boxed{
\frac{\partial J}{\partial m}
=
-\frac{2}{n}
\sum_{i=1}^{n}
x_i(y_i-\hat{y}_i)
}
$$

## Gradient with respect to b

$$
\boxed{
\frac{\partial J}{\partial b}
=
-\frac{2}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)
}
$$

## Weight update

$$
\boxed{
m_{new}
=
m_{old}
-
\alpha
\frac{\partial J}{\partial m}
}
$$

## Bias update

$$
\boxed{
b_{new}
=
b_{old}
-
\alpha
\frac{\partial J}{\partial b}
}
$$

## General Gradient Descent equation

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\alpha\nabla_\theta J(\theta_t)
}
$$

where:

-   $\theta$ represents the model parameters,
-   $\nabla_\theta J$ represents the gradient,
-   $\alpha$ is the learning rate.

------------------------------------------------------------------------

# 25. Interview Questions

### Q1. What is Gradient Descent?

Gradient Descent is an optimization algorithm that minimizes a loss
function by iteratively updating model parameters in the direction
opposite to the gradient.

### Q2. Why do we subtract the gradient?

The gradient points toward the direction of steepest increase in the
loss. Subtracting it moves the parameters toward decreasing loss.

### Q3. What is the learning rate?

The learning rate is a hyperparameter that controls the size of each
parameter update.

### Q4. What happens if the learning rate is too high?

The optimizer may overshoot the minimum, oscillate, or diverge.

### Q5. What happens if the learning rate is too low?

Training becomes slow because parameter updates are very small.

### Q6. What is a gradient?

A gradient contains derivatives that describe how the loss changes with
respect to model parameters.

### Q7. What is convergence?

Convergence occurs when further optimization steps cause little or no
meaningful reduction in loss.

### Q8. What are the main types of Gradient Descent?

-   Batch Gradient Descent
-   Stochastic Gradient Descent
-   Mini-Batch Gradient Descent

### Q9. What parameters are optimized in simple Linear Regression?

The slope/weight `m` and intercept/bias `b`.

### Q10. Is Gradient Descent itself a model?

No. Gradient Descent is an **optimization algorithm** used to train
models by minimizing an objective/loss function.

------------------------------------------------------------------------

# 26. Final Summary

The complete idea can be remembered as:

``` text
Data
 ↓
Model
 ↓
Prediction
 ↓
Compare prediction with actual value
 ↓
Calculate loss
 ↓
Calculate gradients
 ↓
Use learning rate
 ↓
Update parameters
 ↓
Repeat
 ↓
Loss decreases
 ↓
Model improves
```

For simple Linear Regression:

$$
\hat{y}=mx+b
$$

Calculate the loss:

$$
MSE=
\frac{1}{n}
\sum(y_i-\hat{y}_i)^2
$$

Calculate gradients:

$$
\frac{\partial J}{\partial m}
=
-\frac{2}{n}
\sum x_i(y_i-\hat{y}_i)
$$

$$
\frac{\partial J}{\partial b}
=
-\frac{2}{n}
\sum(y_i-\hat{y}_i)
$$

Update parameters:

$$
m_{new}
=
m_{old}
-
\alpha\frac{\partial J}{\partial m}
$$

$$
b_{new}
=
b_{old}
-
\alpha\frac{\partial J}{\partial b}
$$

Then repeat until the loss is minimized sufficiently.

------------------------------------------------------------------------

## One-Line Definition to Remember

> **Gradient Descent is an optimization algorithm that repeatedly
> adjusts model parameters in the opposite direction of the loss
> gradient to minimize the loss function.**

------------------------------------------------------------------------

## Learning Note

Do not try to memorize only the formulas. Remember the logic:

**Predict → Measure Error → Find Slope → Take a Controlled Step Downhill
→ Repeat.**

Once this idea is clear, optimizers used in Deep Learning---such as SGD
with Momentum, AdaGrad, RMSProp, and Adam---become much easier to
understand.
