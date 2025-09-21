+++
title = "A note on 1st and 2nd order optimization"
date = "2024-09-21"

[taxonomies]
tags=["optimization","optimizer","firstorder","secondorder","sgd"]

[extra]
comment = true
+++

Optimization lies at the heart of training deep learning models, serving as the driving force behind the training of complex neural networks. The goal of optimization is to minimize a cost function $J(\theta)$.
\[
J(\theta)=\mathbb{E}_{(x,y)\sim \hat{p}_{data}} L(f(x;\theta),y),
\]
where $L$ is the per-example loss function, $f(x;\theta)$ is the predicted output when the input is $x$, and $\hat{p}_{data}$ is the empirical distribution. $J(\theta)$ is defined with respect to the training set. The expectation is taken across the data-generating distribution $p_{data}$ rather than just over the finite training set is given by:
$$J^*(\theta) = \mathbb{E}_{x,y}\sim p_{data}L(f(x;\theta),y).
$$
This quantity is known as the **risk**. If we knew the true distribution $p_{data}(x,y)$, risk minimization would be an optimization problem solvable by an optimization algorithm. When we do not know $p_{data}(x,y)$ but only have a set of training samples $\hat{p}_{data}(x,y)$, we have a machine learning problem. So we minimize the **empirical risk**
$$\mathbb{E}_{x,y\sim\hat{p}_{data}(x,y)}[L(f(x;\theta))]=\frac{1}{m}\sum_{i=1}^mL(f(x^{(i)};\theta), y^{(i)})$$
The training process based on minimizing this average training error is known as empirical risk minimization. We now minimize this risk with optimization methods. One of the most popular or dominant method have been stochastic gradient descent (SGD) (Robbins & Monro, 1951). SGD and its variant have been the cornerstone of deep learning optimization due to their simplicity and effectiveness in high-dimensional spaces (Bottou, 2010). Later various adaptive methods that scales the gradient by square roots of some form of the average of the squared values of past gradients have been proposed. Adam (Kingma & Ba, 2015), RMSprop (Tieleman & Hinton, 2012), and Adagrad (Duchi et al., 2011), have gained popularity due to their ability to automatically tune learning rates based on the history of gradients. These methods have demonstrated significant advantages over non-adaptive approaches, particularly in handling sparse gradients, non-stationary objectives, and ill-conditioned optimization landscapes. Since the non-convex and high-dimensional nature of deep learning loss landscapes poses significant challenges, such as saddle points, poor conditioning, and vanishing/exploding gradients (Goodfellow et al., 2016).

Optimization methods in deep learning are broadly categorized into first-order and second-order methods, distinguished by their reliance on gradient and hessian information, respectively.

## First-Order methods

First-order optimization methods rely solely on the first derivative of the loss function $L(\theta)$ with respect to the parameters $\theta$. Given an optimization problem:
$$\min\limits_{\theta\in\mathbb{R}^d}L(\theta)$$
where $L:\mathbb{R}^d\rightarrow\mathbb{R}$ is a continuously differentiable function, first-order methods rely o the gradient $\nabla L(\theta)$ to perform iterative updates of the form:
$$\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$$
where $\eta >0$ is the step size (learning rate).

## Second-Order methods

Second-order optimization methods leverage the second derivatives of the loss function $L(\theta)$ to account for the curvature information. Given the optimization problem:
$$\min\limits_{\theta\in\mathbb{R}^d}L(\theta)$$
where $L:\mathbb{R}^d\rightarrow\mathbb{R}$ is twice differentiable, these methods use the Hessian matrix $H=\nabla^2L(\theta)$ to guide the update direction. A general update rule is:
$$\theta_{t+1}=\theta_t-\eta\mathbf{H}^{-1}\nabla L(\theta_t)$$
where $\mathbf{H}^{-1}\nabla L(\theta_t)$ accounts for local curvature, allowing for more informed updates compared to first-order methods. Newton's method, Quasi-Newton methods like BFGS and L-BFGS are some popular second-order methods.
Second-order methods offers significantly faster convergence compared to first-order methods. Under strong convexity and smoothness, Newton's method converges quadratically with a rate of $O((\frac{1}{2})^{2^t})$ near the minimum (Nocedal & Wright, 2006). While SGD converges sub-linearly with a rate of $O(\frac{1}{\sqrt{T}})$ for non-convex functions (Bottou, 2010). First-order methods struggle with regions where the gradient norm is small, $\nabla L(\theta) \approx 0$, such as saddle points or plateaus. Second-order methods can leverage Hessian eigenvalues to escapes the plateaus.
Despite the theoretical advantages, second-order methods are not widely used in deep learning. The main computational challenge is the Hessian inversion $\mathbf{H}^{-1}$. The Hessian matrix has $O(d^2)$ elements, where $d$ is the number of parameters, computing and storing $\mathbf{H}$ requires $O(d^2)$ memory, and inverting it requires $O(d^3)$ operations. For deep learning models with millions of parameters, storing and inverting the Hessian is infeasible. Even approximate methods like L-BFGS or K-FAC requires $O(d^2)$, a significant computational overhead compared to 1st order methods $O(d)$.
The non-convex and high-dimensional nature of deep learning loss landscapes poses significant challenges, such as saddle points, poor conditioning, and vanishing/exploding gradients (Goodfellow et al., 2016). The Hessian $\mathbf{H}$ may not be positive definite, making its inversion unstable or meaningless. Also, in practice, the stochastic gradients gradients computed from mini-batches introduces noise.
$$\theta_{k+1}=\theta_k-\alpha_k\Bigg(\frac{1}{|B_k|}\sum_{i\in B_k}\nabla L_i(\theta_k)\Bigg),$$
where $B_k\subset {1,2,\cdots,M}$ is the batch sampled from the data set and $\alpha_k$ is the step size at iteration $k$. While Hessian computed from noisy gradients is unreliable and can lead to poor updates, SGD and its variants under $|B_k|\ll M$ are employed, typically $|B_k| \ll {32, 64, \cdots , 512}$ have been successfully used in practice for a large number of applications [Keskar et al., 2017] (Simonyan & Zisserman, 2014; Graves et al., 2013; Mnih et al., 2013).
Second-order methods leverage curvature information for faster convergence but this benefits are outweighed by their computational and practical limitations.


Modern neural networks are typically trained with first-order gradient methods, which can be broadly categorized into two branches: the accelerated stochastic gradient descent (SGD) family, such as Nesterov accelerated gradient (NAG), SGD with momentum and heavy-ball method (HB); and the adaptive learning rate methods, such as Adagrad, AdaDelta, RMSProp and Adam.