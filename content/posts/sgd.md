+++
title = "A note on SGD"
date = "2024-09-14"

[taxonomies]
tags=["sgd","momentum","optimization","nag"]

[extra]
comment = true
+++

## Stochastic Gradient Descent (SGD)

It is a first-order method that minimizes the loss function $L(\theta)$ by iteratively updating the parameters $\theta$ using the gradient of the loss computed on a mini-batch of data.
$$\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t;\mathcal{B}_t),$$
where, $\eta$ is the learning rate and $\nabla L(\theta_t;\mathcal{B}_t)$ is the gradient of the loss function computed on a mini-batch $\mathcal{B}_t$ at iteration $t$.
For non-convex loss function but not strongly convex function, SGD converges to a stationary point with a rate of $O(1/\sqrt{T})$ (Ghadimi & Lan, 2013). And for a convex loss functions, SGD achieves a convergence rate of $O(1/\sqrt{T})$ (Bottou, 2010).
The stochastic nature of SGD introduces noise in the updates, which helps to escape the saddle points. This property is crucial for optimizing non-convex loss landscapes, which are riddled with saddle points (Ge et al., 2015). But on ill-conditioned loss landscape, SGD converges slowly because the gradient direction may not align with the direction of minimum, since the direction is perpendicular to the level curve. Also SGD can oscillate significantly, especially in high-curvature region, because it does not account for the history of gradients. This issue can be mitigated by introducing momentum.

### Momentum
Momentum (Polyak, 1964) introduces a velocity term $v_t$ to accumulate past gradients, leading to smoother and faster updates:
$$v_{t+1}=\beta v_t+(1-\beta)\nabla L(\theta_t;\mathcal{B}_t),$$
$$\theta_{t+1}=\theta_t-\eta v_{t+1},$$
where, $\beta \in [0,1)$ is the momentum coefficient and $v_t$ is the velocity at iteration $t$.
Momentum dampens the noise or smooths the gradient updates, reducing oscillations and accelerating convergence in directions of persistent gradient consistency by accumulating past gradients. The convergence rate is improved to $O(1/T)$ for convex function (Polyak, 1964). Also, the accumulated velocity helps the optimizer escape shallow local minima and saddle points.

### Nesterov Accelerated Gradient (NAG)
The idea is to evaluate the gradient at a look-ahead position:
$$v_{t+1}=\beta v_t-\eta\nabla L(\theta_t+\beta v_t;\mathcal{B}_t),$$
$$\theta_{t+1}=\theta_t+v_{t+1},$$
Here $\theta_t+\beta v_t$ is lookahead point where the gradient is computed instead of the current parameter $\theta_t$.
NAG achieves a faster convergence rate than standard momentum, particularly for convex functions. The convergence rate is $O(1/T^2)$ under strong convexity (Nesterov, 1983). Ilya Sutskever (2012) in his thesis “Training Recurrent Neural Networks”, investigated the impact of NAG in deep learning, particularly in the context of training deep networks. He empirically demonstrated that NAG consistently outperformed classical momentum-based SGD in training deep neural networks.


SGD and its variants uses a global learning rate for all parameters, which may lead to suboptimal convergence in high-dimensional spaces. In a shallower layers, that are closer to the input might have larger gradients, therefore uniform learning rates may lead to under training deeper layers or over shooting in shallower ones (Hinton Course). Since, global learning rate treats every weight equally even where some weights update too slowly (vanishing gradients) or too aggressively (exploding gradients), destabilizing the training.
To mitigate these issues we need per parameter adaptive learning rates.
