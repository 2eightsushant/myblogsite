+++
title = "Amazing Linear Units"
date = "2024-09-01"

[taxonomies]
tags=["relu","swish","swiglu","gelu","activation"]

[extra]
comment = true
+++

### ReLU
Rectifying Linear Unit (ReLU) [Hahnloser et al., 2000; Jarrett et al., 2009; Nair & Hinton, 2010] is a cornerstone activation function in deep learning, valued for its simplicity and ability to facilitate effective gradient propagation. It is defined as:
$$g(z)=\text{max}(0,z)$$
 ReLU behaves similar to linear units for positive inputs but outputs zero for negative inputs, ensuring that derivatives remain large and consistent when the unit is active ($z>0$).
#### Vanishing gradients
The vanishing gradient problem arises when gradients diminish as they are backpropagated through deep layers, resulting in negligible weight updates in early layers. This severely slows or prevents learning in deep neural networks.

Consider a neural network composed of $L$ layers, where the output of the $l^{th}$ layer is given by,
$$h^{(l)}=g^{(l)}(\mathbf{z}^{(l)}), \quad \mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)}$$
where, $g^{(l)}$ is a differentiable activation function.

Let the loss function be $\mathcal{L}$. Then, the gradient of the loss with respect to the weights at layer $l$ is,
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}=\delta^{(l)} \cdot (\mathbf{h}^{(l-1)})^T$$
where the backpropagated error is,
$$\delta^{(l)}=\left(\prod_{k=l+1}^L{\mathbf{J}^{(k)}}\right) \cdot \nabla_{\mathbf{h}^{(L)}}\mathcal{L}$$
where $\mathbf{J}^{(k)}$ is the Jacobian matrix of the layer $k$ with respect to its input, such that
$$\mathbf{J}^{(k)}=\frac{\partial \mathbf{h}^{(k)}}{\partial \mathbf{h}^{(k-1)}}=\text{diag}(g^\prime(\mathbf{z}^{(k)})) \cdot \mathbf{W}^{(k)}$$
A vanishing gradient problem occurs when, for early layers $l \ll L$, the norm of the gradient approaches zero,
$$\Bigg\|\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \Bigg\| \rightarrow 0 \quad \text{as} \quad L \rightarrow \infty$$
This happens if the product of the Jacobians contains singular values less than 1, causing exponential decay of the gradient magnitude with network depth.

For the sigmoid, we have
$$\sigma(z)=\frac{1}{1+e^{-z}}, \quad \sigma^\prime(z)=\sigma(z)(1-\sigma(z))$$
Since $\sigma^\prime(z)$ lies in the interval $(0,0.25]$, the maximum derivative is $0.25$ at $z=0$, but for large or small $z$, $\sigma^\prime(z) \to 0$. Consequently, Jacobian norm is small and bounded as,
$$\| \mathbf{J}^{(k)} \| \le \| \text{diag}(\sigma’(\mathbf{z}^{(k)})) \| \cdot \| \mathbf{W}^{(k)} \| \ll 1$$
The product of Jacobians decays exponentially,
$$\left\| \prod_{k=l+1}^{L} \mathbf{J}^{(k)} \right\| \to 0 \quad \text{as } \quad L \to \infty$$
This exponential decay causes vanishing gradients, making sigmoid unsuitable for deep networks.

Similarly, for hyperbolic tangent
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh’(z) = 1 - \tanh^2(z) \in (0, 1]$$While the max derivative is 1 at $z=0$, but $\tanh^\prime(z) \to 0$ as $|z|$ increases. In practice most neurons operate in saturated regions where $\tanh’(z) < 1$, leading to $\| \mathbf{J}^{(k)} \| < 1$. Similar to sigmoid, the product of Jacobians decays exponentially, causing vanishing gradients. However, tanh is zero-centered $\tanh(0) = 0$, unlike sigmoid $\sigma(0) = 0.5$, which slightly improves gradient flow during optimization.

In case of ReLU, the derivative is defined as:
$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$
For $(z > 0)$ active region, the derivative is $1$, so the Jacobian norm is:
$$\| \mathbf{J}^{(k)} \| \approx \| \mathbf{W}^{(k)} \|$$
This preserves gradient magnitude across layers, avoiding exponential decay. For inactive region $(z \leq 0)$, derivative is $0$, which can lead to dead neurons that never activate or update. However, in practice, a sufficient fraction of neurons typically have $(z > 0)$, ensuring effective gradient flow.

Hence, whether gradients propagate effectively or diminish in ReLU network depends on the fraction of active neurons.
Moreover, the ReLU's effectiveness depends on the scale of weights in $\mathbf{W}^{(k)}$. If weights are not standard normal or improperly initialized, activations may be predominantly negative, causing widespread inactivation (dead neurons), or too large causing instability. This issue was observed by [Glorot & Bengio (2010)], who emphasized the need for careful initialization.

He-initialization (Kaiming) [He et al., 2015] scales the variance of weights as,
$$\text{Var}[W_{ij}] = \frac{2}{\text{fan-in}}$$
This accounts for ReLU’s non-zero output for roughly half the inputs, preserving both forward activation variance and backward gradient magnitude. If weights are properly initialized with He-initialization then $\| \mathbf{W}^{(k)} \| \approx 1$, and
$$\left\| \prod_{k=l+1}^{L} \mathbf{J}^{(k)} \right\| \not\to 0 \Rightarrow \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \right\| \not\to 0$$
This prevents exponential decay in gradients as seen in sigmoid and tanh networks.

#### Differentiability and Dying ReLU
ReLU has two notable characteristics: its non-differentiability at ($z = 0$) and the potential for the dying ReLU problem.

ReLU is not differentiable at $z=0$. This might appear problematic for gradient-based learning algorithms. However, in practice, training typically does not require exact gradients at every point, since neural network training rarely converges to a sharp local minimum of the loss function. Moreover, due to finite numerical precision on digital computers, exact zeros are rare, and automatic differentiation frameworks usually define $g’(0)$ as either $0, 1$, or use subgradient methods. This practical treatment enables ReLU to function effectively despite the non-differentiability at a single point.

Another issue with ReLU is that it does not learn from examples when the activation is zero. Specifically, for $z \leq 0$, the output is zero, and during backpropagation, such neuron does not contribute no gradient to the preceding layer. If $\mathcal{L}$ is the loss function and the output of the neuron as $h=g(z)$, then:
$$\frac{\partial \mathcal{L}}{\partial z}=\frac{\partial \mathcal{L}}{\partial h} \cdot g^\prime(z)$$
Such that no weight updates will occur for $z \leq 0$ . If this persists, it can lead to a phenomenon known as dying ReLU problem, where neurons can become permanently inactive (dead) and stop learning altogether.

Dead neurons reduce the network’s capacity to learn complex patterns. In extreme cases, a significant fraction of dead neurons can lead to slower convergence or degraded performance.

To mitigate this, several generalizations of ReLU have been proposed that allow non-zero gradients when $z \leq 0$. A common generalized form is,
$$h(z)=\text{max}(0,z)+\alpha \ \text{min}(0,z)$$
Absolute value rectification fixes $\alpha=-1$ resulting in $g(z)=|z|$ [Jarrett et al, 2009]. Leaky ReLU uses small fixed $\alpha \ll 1$ [Mass et al, 2013] and PReLU (Parametric relu) uses $\alpha$ as learnable parameter[He et al, 2013].

While these variants address the dying ReLU issue, they often provide only marginal improvements in performance over standard ReLU. Moreover, they introduce additional parameters or hyperparameters, which may require tuning and slightly increase computational overhead.

### Swish and SwiGLU
The Swish activation function[Ramachandran, 2017] was discovered through an automated search for scalar nonlinearities that balance expressiveness and computational simplicity. It is defined as:
$$\text{Swish}(z)=z.\sigma(\beta z)$$
where , $\sigma=\frac{1}{1+e^{-z}}$ is the sigmoid function, and $\beta$ is fixed or trainable scalar parameter.

The Swish activation function offers adaptable behavior controlled by its scaling parameter $\beta$.
- When $\beta = 1$, Swish becomes equivalent to the Sigmoid-weighted Linear Unit (SiL), originally introduced by Elfwing et al. (2017) in the context of reinforcement learning.
- At $\beta = 0$, the sigmoid term becomes constant, reducing Swish to a simple linear mapping with half the input magnitude, i.e., $f(z) = \frac{z}{2}$.
- As $\beta \to \infty$, the sigmoid function approaches a binary step, and Swish asymptotically mimics the ReLU function.

This behavior shows that Swish acts as a smooth, continuous function that bridges the gap between purely linear activations and hard-threshold nonlinearities like ReLU. When $\beta$ is treated as a learnable parameter, the network gains the ability to dynamically adjust the nonlinearity during training, adapting the activation shape based on the optimization landscape.

Unlike the piecewise linear ReLU, Swish is continuously differentiable across the entire domain. Its derivative is: 
$$\text{Swish}^\prime(z) = \sigma(\beta z) + \beta z \cdot \sigma(\beta z)(1 - \sigma(\beta z))$$
The derivative is non-zero for all $z \in \mathbb{R}$, ensuring continuous gradient flow during backpropagation, even for negative inputs. Additionally, Swish exhibits a non-monotonic "bump" in the region approximately $-5 \leq z \leq 0$, for $\beta=1$ allowing it to produce small negative outputs and thus mitigate the dying ReLU problem by keeping neurons active.

While Swish performs well as a standalone activation, its global scalar gating scales neurons modulates all neurons outputs uniformly. This can  potentially lead to over-smoothing gradients, particularly in deep networks where diverse feature modulation is often needed. In transformer architectures, which rely on dynamic gating, a more expressive mechanism is needed.

The Gated Linear Unit (GLU), introduced by [Dauphin et al. (2016)], addresses this by employing a sigmoid-based gated projection to control feature flow.
$$\text{GLU}(x) = \sigma(xW + b) \odot (xV + c)$$

SwiGLU, proposed by [Shazeer (2020)], bridges the strengths of Swish and GLU. By replacing GLU’s sigmoid gate with Swish, it leverages the smooth, non-monotonic nonlinearity of Swish with GLU’s adaptive gating. It is defined as
$$\text{SwiGLU}(x) = \text{Swish}(xW + b) \odot (xV + c)$$
where,
$x \in \mathbb{R}^d$ is the input vector,
$W, V \in \mathbb{R}^{d \times d}$ are weight matrices, and $b, c \in \mathbb{R}^d$ are biases that are omitted by default,
$\text{Swish}(z) = z \cdot \sigma(\beta z)$, with $\beta = 1$ by default and
$\odot$ denotes element-wise multiplication.

The element-wise product with $( xV + c )$ selectively scale features based on input context, enhancing expressiveness compared to Swish’s uniform modulation. SwiGLU thus inherits Swish’s gradient-preserving, smooth behavior and combines it with GLU-style adaptive gating, making it highly effective in transformer-based architectures.

Swish demonstrated measurable improvements in performance over ReLU, achieving a +0.9% increase in top-1 accuracy on Mobile NASNet-A and +0.6% on Inception-ResNet-v2 when evaluated on the ImageNet classification benchmark[Ramachandran, 2017]. Similarly, SwiGLU outperformed both ReLU and Swish, achieving lower perplexity in transformer models, particularly on the segment-filling task [Shazeer, 2020]. It has been adopted in several state-of-the-art LLMs, like LLaMA, PaLM, and GPT-NeoX.

However, the sigmoid component in Swish and the combination of sigmoid gating and dual linear projections in SwiGLU, introduces additional computational overhead. This increased resource demand, compared to the simple and light ReLU limits their widespread adoption, particularly in environments with strict latency or memory constraints.

### GELU
The Gaussian Error Linear Unit (GELU), introduced by Hendrycks and Gimpel in 2016 [Gaussian Error Linear Units, Hendrycks], It unifies the
deterministic activation of ReLU with stochastic regularization techniques like dropout and zoneout, offering a smooth, adaptive mechanism that weights inputs based on their magnitude under a Gaussian distribution. This is achieved by defining a probability of retention that increases with the input magnitude. Specifically, for an input $x$, the mask is drawn from a Bernoulli distribution with success probability given by the standard Gaussian cumulative distribution function (CDF).

GELU is defined as:
$$\text{GELU}(x)=x \cdot \Phi(x)$$
where $\Phi(x)=P(X \leq x), X \sim \mathcal{N}(0,1)$, is the standard Gaussian cumulative distribution function (CDF). 
It can be interpreted as scaling the input proportionally to how likely it is to exceed other values under a standard normal distribution. In other words, inputs are modulated based on their relative magnitude within the input space.
Thus, GELU multiplies the input by a binary mask that is random, yet input-dependent. As a result, smaller values of $x$ are more likely to be suppressed, while larger values are more likely to be retained. This input-sensitive behavior introduces a smooth form of gating, similar in spirit to adaptive dropout [Ba & Frey, 2013], although GELU relies on the Gaussian CDF rather than the logistic function and is not paired with a separate nonlinearity.

Using the error function $(\text{erf})$, a closed-form approximate of CDF, GELU can be expressed as:
$$\text{GELU}(x)=x \cdot \frac{1}{2} \left[1+\text{erf} \left(\frac{x}{\sqrt{2}} \right) \right]$$
For computational efficiency, GELU is approximated as, a $\text{tanh}$ based approximation:
$$\text{GELU}(x) \approx 0.5x \left(1+ \tanh \left[\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right]\right)$$
and sigmoid based approximation:
$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$
where $\sigma$ is the sigmoid function. These approximations, while slightly less accurate, are faster to compute and suitable for high-performance scenarios.

ReLU outputs only non-negative values, making it monotonic (non-decreasing) and piecewise linear. It lacks curvature in the positive region, which can limit its expressiveness for modeling complex functions. GELU, conversely, can produce both positive and negative outputs.
For $x>0, \text{GELU}(x)=x.\Phi(x)>0$, since $\Phi(x)>0.5$
For $x<0, \text{GELU}(x)=x.\Phi(x)<0$, since $x<0$ and $\Phi(x)>0$

Consider the derivative of GELU:
$$\text{GELU}^\prime(x)=\Phi(x) + x \cdot \phi(x)$$
where $\phi(x)$ is the standard Gaussian probability density function (PDF).
For $x>0$, both terms are positive, indicating an increasing function.
For $x<0$, $\Phi(x)>0$ but $x \cdot \phi(x)<0$, suggests GELU decreases initially as $x$ increases from $-\infty$, reaches minimum and then increases.

Thus, GELU is non-monotonic, exhibiting a minimum for some $x<0$, and non-convex, with smooth curvature across its domain. This contrasts with ReLU's piecewise linear behavior, potentially enhancing GELU's capacity to approximate complex or highly nonlinear functions [Gaussian Error Linear Units, Hendrycks]. A further distinction lies in how each function modulates its input. ReLU acts as a binary gate, passing through inputs only when they are positive. GELU, on the other hand, weights inputs probabilistically based on their magnitude relative to the standard normal distribution—effectively applying a smooth, input-dependent scaling rather than a hard threshold. Importantly, this probabilistic mechanism gives GELU a statistical interpretation: it represents the expected output of a stochastic process that conditionally retains the input.

Empirical results [Gaussian Error Linear Units, Hendrycks] support GELU’s benefits in practice. In part-of-speech tagging tasks on noisy Twitter datasets (e.g., [Gimpel et al., 2011, and Owoputi et al., 2013]), GELU achieved a median test error of 12.57%, compared to ReLU’s 12.67%, indicating greater robustness. Similar trends are observed on image classification benchmarks: on CIFAR-10, GELU achieves a median error of 7.89% versus ReLU’s 8.16%; and on CIFAR-100, GELU outperforms ReLU with 20.74% median error compared to 21.77%.

GELU represents a significant advancement in activation function design, integrating ReLU’s deterministic gating with the stochastic properties like of adaptive dropout. Its non-monotonicity, smooth curvature, and probabilistic weighting enhance expressiveness and robustness, as evidenced by superior performance in tasks like part-of-speech tagging and image classification. GELU has been the standard choice of activation function for transformer models making it a cornerstone component of modern neural network architectures.

### ReLU$^2$

The standard ReLU activation function, defined as $f(x) = \max(0, x)$, is a cornerstone of deep learning due to its simplicity and effectiveness in mitigating the vanishing gradient problem. However, its linear output for positive inputs can limit expressivity in complex tasks. ReLU$^2$ addresses this by squaring the output for positive inputs, resulting in $f(x) = (\max(0, x))^2$. This quadratic transformation enhances the network's ability to model nonlinear relationships while maintaining the sparsity inherent to ReLU, where non-positive inputs yield zero outputs. The adoption of ReLU$^2$ in transformer models, underscores its relevance in optimizing large-scale language models for both performance and efficiency.

Unlike the linear output of standard ReLU for positive inputs, ReLU$^2$ introduces a quadratic growth, enabling the network to capture more complex, nonlinear patterns with fewer layers. ReLU$^2$ exhibits quadratic growth for large positive inputs, differing from the linear growth of ReLU or the smooth asymptotics of GELU and Swish. This unique behavior can influence the network's learning dynamics, potentially leading to faster convergence in certain tasks.

Neural networks employing ReLU powers, such as ReLU$^2$ and ReLU$^3$, have demonstrated superior performance over standard ReLU in tasks requiring complex function approximation. In $\textit{Neural networks with ReLU powers need less depth}$, Cabanilla et al. utilized the approximation spaces framework to show that ReLU power networks require fewer layers to approximate smooth, nonlinear functions that cannot be represented by low-degree polynomials. Experimental results on the Rastrigin and Ackley functions, which are characterized by high nonlinearity and multiple local minima, revealed that ReLU$^2$ and ReLU$^3$ networks with two hidden layers consistently outperformed standard ReLU networks with equivalent architectures. This suggests that ReLU powers can achieve higher approximation accuracy with reduced depth, leading to more efficient models with lower computational costs.

Sparse attention is a technique designed to enhance the efficiency of transformer models by limiting the attention mechanism to a subset of input tokens, rather than computing pairwise interactions across all tokens. This approach is particularly valuable for processing long sequences, where full attention mechanisms scale quadratically with sequence length. Sparse attention patterns, such as local or strided attention, reduce computational complexity while maintaining model performance, making them essential for scaling large language models (LLMs). The effectiveness of sparse attention is closely tied to the activation patterns within the feedforward network (FFN) layers, where activation functions like ReLU$^2$ play a critical role in promoting sparsity.

Recent research has highlighted the emergence of sparse activation patterns in transformer models, where only a small fraction of neurons are activated for a given input. In $\textit{The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers}$, Li et al. observed that in the T5 model, $90\%$ of inputs activate less than $5\%$ of neurons in FFN layers. This sparsity is not due to "dead" neurons but rather a dynamic property, with nearly all neurons activating at least occasionally. The study found that sparsity is ubiquitous across various transformer configurations, datasets, and tasks, including both vision and natural language processing. Notably, deeper and wider models exhibit higher sparsity, with the percentage of activated neurons decreasing as model size increases. For instance, in a fine-tuned T5-Large model, the percentage of nonzero entries in activation maps dropped from approximately 50\% at initialization to an average of $2.7\%$ after training, with some layers as low as $1.1\%$.

This sparsity is attributed to training dynamics, where gradient descent tends to favor sparse activation maps. The biological analogy to sparse neural firing in brains suggests that this property enhances efficiency, as only a small fraction of neurons contribute to computation at any given time, reducing energy costs. By leveraging sparsity, transformer models can significantly reduce floating-point operations (FLOPs) during inference, as a large proportion of computations involve multiplying by zero.

The choice of activation function significantly influences the trade-off between model performance and sparsity. In $\textit{ReLU Strikes Back: On the Generalization of Deep ReLU Networks}$, Mirzadeh et al. trained the OPT 1.3B model on 100 billion tokens from the RefinedWeb dataset, comparing ReLU, SwiGLU, and GELU. The results indicated that while SwiGLU and GELU may offer marginal performance improvements, they result in lower sparsity compared to ReLU. Specifically, ReLU achieved a $32\%$ reduction in inference FLOPs (from 6.6G to 4.5G per token) due to its high sparsity, although it required slightly longer training to match the performance of non-ReLU activations. The study also demonstrated that inserting additional ReLU layers after normalization layers could further reduce inference FLOPs by up to threefold, highlighting ReLU's efficiency in sparse computation.

However, non-ReLU activations like SwiGLU often achieve slightly higher performance at the cost of reduced sparsity, making them less efficient for resource-constrained environments. This trade-off underscores the importance of selecting an activation function that aligns with the specific requirements of the task, balancing accuracy with computational efficiency.

ReLU$^2$ has emerged as an optimal activation function for balancing performance and sparsity in large language models. In $ReLU^2 \textit{Wins: Discovering Efficient Activation Functions for Sparse LLMs}$, Zhang et al. proposed a novel definition of neuron activation based on output magnitudes exceeding a threshold, rather than strictly zero/non-zero values. This approach broadens the scope of sparse activation, allowing ReLU$^2$ to achieve higher sparsity ratios while maintaining competitive performance. The study compared ReLU, SwiGLU, ReGLU, and ReLU$^2$, finding that ReLU$^2$ offered the highest sparsity ratio for a given performance level and the highest performance for a given sparsity ratio. Although SwiGLU achieved the best overall performance, ReLU$^2$ provided the best trade-off, making it ideal for sparse LLMs.

Furthermore, in $\textit{Accelerating Transformer Inference and Training with 2:4 Activation Sparsity}$, Haziza et al. demonstrated that replacing SwiGLU with ReLU$^2$ in transformer models maintained model accuracy while significantly increasing sparsity in FFN activations during both training and inference. This sparsity was leveraged with 2:4 sparsity patterns on GPUs, achieving up to 1.3x faster FFN computations. The ability of ReLU$^2$ to preserve accuracy while enhancing sparsity makes it particularly suitable for large-scale transformer models, where computational efficiency is critical.

The sparsity induced by ReLU$^2$ can be further optimized using hardware accelerators, such as GPUs with sparse tensor cores. The successful integration of ReLU$^2$ in models like Primer and BitNet demonstrates its practical feasibility in large-scale architectures.
