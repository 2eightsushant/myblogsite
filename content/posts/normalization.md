+++
title = "A Note on Normalization"
date = "2024-09-14"

[taxonomies]
tags=["normalization","batchnorm","layernorm","rmsnorm"]

[extra]
comment = true
+++

Training state-of-the-art, deep neural networks is computationally expensive. One way to accelerate and stabilize training is to normalize the activities of the neurons. Training deep neural networks poses significant challenges because the input to each layer depends on the parameters of all preceding layers. Small updates to these parameters can lead to substantial changes in the activation distributions of deeper layers, a problem termed internal covariate shift[Shimodaira, 2000]. This phenomenon complicates optimization by slowing convergence, requiring smaller learning rates, and increasing sensitivity to initial parameter values, particularly in networks with saturating activation functions. There have been several researches on the normalization. One of the pioneer work is Batch Normalization [Ioffe]. Following the work, Batch renormalization [Ioffe], Instance normalization [Hoffer],  Weight normalization [Salimans] and so on. But we cover three major normalization here.

### Batch Normalization
Batch Normalization (BN)[BatchNormalization, Ioffe] mitigates internal covariate shift by stabilizing the distribution of layer inputs throughout training. By maintaining consistent input distributions, BN enables faster and more robust training. This concept builds on earlier findings [LeCun1998, Wiesler2011] suggesting that neural networks benefit from input normalization, such as whitening, which transforms inputs to have zero mean, unit variance, and, ideally, no correlation between features.

However, whitening all features jointly is computationally intensive and often non-differentiable. To address this, Batch Normalization employs two practical simplifications:
1. Per-Feature Normalization
	Rather than whitening the entire feature vector, BN normalizes each feature independently. For an input vector $\mathbf{x} = (x^1, x^2, \dots, x^d)$, the normalized value for feature $k$ is computed as:
	$$\hat{x}^k = \frac{x^k - \mu^k}{\sqrt{(\sigma^k)^2 + \epsilon}}$$
	where $\mu^k$ and $(\sigma^k)^2$ are the mean and variance of feature $k$ over a mini-batch, and $\epsilon$ is a small constant for numerical stability. To maintain the network's expressive power, BN introduces learnable parameters $\gamma^k$ and $\beta^k$ which scale and shift the normalized value, applying an affine transformation:
	$$y^k = \gamma^k \hat{x}^k + \beta^k$$
	These parameters, optimized during training, allow the network to recover the original feature distribution if necessary.
2. Mini-Batch Statistics  
    BN estimates the mean and variance using mini-batches. For a mini-batch  
    $\mathcal{B} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m\}$,  
    the normalization for each input $\mathbf{x}_i$ is:  

    $$
    \text{BN}_{\gamma, \beta}(\mathbf{x}_i) = \gamma \cdot \hat{\mathbf{x}}_i + \beta
    $$  

    where  

	$$
	\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
	$$

    $$
    \mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m \mathbf{x}_i
    $$  

    $$
	\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m \left( \mathbf{x}_i - \mu_{\mathcal{B}} \right)^2
	$$
	
	These statistics are differentiable, enabling integration with backpropagation.

By stabilizing activation distributions, BN reduces the risk of unstable gradients or activations caused by parameter updates, thereby improving training efficiency.

Despite its benefits, BN is largely unsuitable for Transformer architectures due to several incompatibilities:
- Reliance on Batch Statistics: BN computes normalization parameters based on the entire mini-batch, creating dependencies between examples. This approach is problematic in Transformers, where batch sizes may be small (e.g., during inference with a batch size of 1) or variable, leading to unreliable statistics.
- Transformers process sequences of different lengths, often requiring padding or masking. Incorporating these variations into BN's batch-level computations adds complexity and reduces efficiency.
- BN's dependence on batch statistics necessitates synchronization across devices in distributed training, increasing communication overhead and limiting parallelism.
- In autoregressive Transformer models, causal masking ensures that each token only attends to prior tokens. BN's use of batch-wide statistics risks violating this constraint by inadvertently incorporating information from future tokens, leading to causality issues.

### Layer Normalization
Layer Normalization (LN) [Layer Normalization, Ba, 2016] normalizes the summed inputs to the neurons within a single training example, unlike BN which computes statistics across the batch dimension. As a result, LN does not introduce dependencies between training examples and can be applied in online and recurrent settings.
To reduce internal covariate shift, LN normalizes the summed inputs within each layer by fixing the mean and variance. Specifically, It computes statistics over feature dimension, rather than over the batch dimension. For a hidden layer $l$, the normalization is computed as:
$$\mu^l=\frac{1}{H}\sum_{i=1}^H{x_i^l} \quad (\sigma^l)^2=\frac{1}{H}\sum_{i=1}^H{(x_i^l-\mu^l)^2}$$
where $H$ is the number of hidden units in the layer, and $x_i^l$ is the $i^\text{th}$ hidden unit in layer $l$. The normalized and scaled output is:
$$\hat{x}_i^l = \frac{x_i^l - \mu^l}{\sqrt{(\sigma^l)^2 + \epsilon}}$$
and,
$$y_i^l = \gamma^l \hat{x}_i^l + \beta^l$$

In contrast, BN normalizes across the batch dimension (over examples) for each feature whereas LN normalizes across the feature dimension (within a single example) for each batch element. This difference in normalization axes leads to their distinct scaling invariance properties.

#### Key properties of LN
1. Weight re-scaling and re-centering invariance
	In BN, re-scaling the incoming weights of a single neuron doesn't effect the normalized output, since normalization is performed across the batch dimension. Whereas, LN behaves differently; LN is not invariant to re-scaling of individual weight vectors, i.e., rows of the weight matrix. But it is invariant to uniform re-scaling or shifting of the entire weight matrix.
	
	Let $W \in \mathbb{R}^{H \times D}$ be the weight matrix at layer $l$, and $\mathbf{x}_i \in \mathbb{R}^D$ be the input to the layer for example $i$. Then the pre-activation for neuron $j$ is
	$$
	z_{ij} = \mathbf{w}_j^\top \mathbf{x}_i,
	$$
	where $\mathbf{w}_j$ is the $j$-th row of $W$.

	Scaling $\mathbf{w}_j \mapsto \delta \mathbf{w}_j$ implies that the pre-activation scales as
	$$
	z_{ij} \mapsto \delta z_{ij}.
	$$
	
	BN computes statistics over the batch $B$ for each neuron:
	 $$\mu_j = \frac{1}{B} \sum_{i=1}^B z_{i,j}, \quad \sigma_j^2 = \frac{1}{B} \sum_{i=1}^B (z_{i,j} - \mu_j)^2$$
	$$
	\hat{z}_{i,j} = \frac{z_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
	$$

	After scaling, the normalized output is
	$$
	\hat{z}_{i,j}^\prime 
	= \frac{\delta z_{i,j} - \delta \mu_j}{\sqrt{\delta^2 \sigma_j^2 + \epsilon}}
	\;\approx\; \frac{z_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
	= \hat{z}_{i,j}.
	$$

	The normalized outputs $\hat{z}_i$ remain unchanged, thus BN is invariant to scaling of a single weight vector.
	
	Whereas LN normalizes across features $H$ for each example $i$, with statistics:
	$$\mu^{(i)} = \frac{1}{H} \sum_{k = 1}^H z_{i,k}, \quad (\sigma^{(i)})^2 =\frac{1}{H}\sum_{k=1}^H (z_{i,k} - \mu^{(i)})^2$$
	After scaling only one $z_{\{i,j\}} \rightarrow \delta\,z_{\{i,j\}}$ changes both the mean and variance:
	$$\mu’^{(i)} = \frac{1}{H} \left( \sum_{k \neq j} z_{i,k} + \delta z_{i,j} \right), \quad (\sigma'^{(i)})^2 \neq (\sigma^{i})^2$$
	This implies LN is not invariant to individual weight scaling vectors.
	
	However, for entire weight matrix scaling $W \mapsto \delta W$, we have 
	$\mathbf{z}_i \mapsto \delta \mathbf{z}_i$ and
	$$
	\mu^{\prime (i)} = \delta \mu^{(i)}, 
	\quad 
	(\sigma^{\prime (i)})^2 = \delta^2 (\sigma^{(i)})^2.
	$$

	The normalized output remains unchanged,
	$$
	\hat{z}_{i,j}^\prime \;\approx\; \hat{z}_{i,j}.
	$$

	Thus, LN is invariant to uniform scaling of the full weight matrix.  

	For pre-activation shifts, adding a constant vector 
	$\mathbf{b} \in \mathbb{R}^H$ (identical for all neurons) to the pre-activations $\mathbf{z}$ leaves LN invariant for uniform shifts, since the mean and variance adjust accordingly:
	$$
	\mathbf{z}_i^\prime = \mathbf{z}_i + \mathbf{b}.
	$$

	If $b_j = c$ for all $j$, then
	$$
	\mu^{\prime (i)} = \mu^{(i)} + c, 
	\quad 
	(\sigma^{\prime (i)})^2 = (\sigma^{(i)})^2 
	\;\Rightarrow\; 
	\hat{z}_{i,j}^\prime = \hat{z}_{i,j}.
	$$

	Hence, LN is also invariant to uniform additive shifts in pre-activation.
	
	But for non uniform shift, $b_j \neq b_k$, then:
	$$\mu'^{(i)} = \mu^{(i)} + \frac{1}{H} \sum_{j = 1}^H b_j, \quad (\sigma'^{(i)})^2 =\frac{1}{H}\sum_{j=1}^H (z_{i,j} + b_j - \mu'^{(i)})^2$$
	The normalized outputs will change, thus LN is not invariant to arbitrary additive shifts.


	For weight shifts, LN is invariant only if inputs are zero-mean:
	$$
	\mathbf{z}_i^\prime 
	= W \mathbf{x}_i + \mathbf{b}\,(\mathbf{1}^\top \mathbf{x}_i) 
	= W \mathbf{x}_i 
	\;\Rightarrow\; 
	\hat{z}_{i,j}^\prime = \hat{z}_{i,j}.
	$$
	 
	 Thus LN is invariant to full weight re-scaling and pre-activation weight re-centering, but they are not invariant to weight re-centering unless inputs are zero-mean. 

2. Data re-scaling and re-centering invariance
	LN is invariant to affine transformation of the input data $\mathbf{z}'=a\mathbf{z}+b$. For scalars $a \ne 0$ and bias $b \in \mathbb{R}$. In this case,
	$$\mu’ = a \mu + b, \quad (\sigma’)^2=a^2\sigma^2$$
	The normalized output becomes:
	$$\hat{z}_i’ = \frac{a(z_i - \mu)}{\sqrt{a^2 \sigma^2 + \epsilon}}  \approx \hat{z}_i$$
	
	Thus, LN preserves invariance under uniform scaling/shifting of input.

[Layer Normalization, Ba, 2016] found that LN introduces an implicit “early stopping” effect on the weight vectors and contributes to training stability. Its invariance properties and independence from batch statistics make it particularly suitable for Transformer-based architectures, where batch sizes can vary and sequences are processed independently. LayerNorm boosts performance in deep networks chiefly by re-centering and re-scaling gradients during backpropagation through the derivatives of the mean and variance ensuring stable gradient flow [Xu et al., 2019].
### RMS Normalization
Layer Normalization (LN) has become a cornerstone of modern neural architectures, including Transformers, due to its stability-enhancing re-centering and re-scaling properties. However, as models grow deeper and larger, LN’s computational overhead particularly the cost of per-step normalization offsets its training efficiency gains. To address this,[Zhang & Sennrich, 2019] proposed RMS Normalization (RMSNorm), a streamlined variant that eliminates the re-centering step while preserving re-scaling invariance.
RMSNorm simplifies LN by normalizing inputs using only the root-mean-square (RMS) statistic:
$$\text{RMS}(\mathbf{x}^l)=\sqrt{\frac{1}{H}\sum_{i=1}^H{(x_i^l)^2}}$$
and,
$$\hat{x}_i^l = \frac{x_i^l}{\text{RMS}(\mathbf{x}^l)+\epsilon}$$
where $\mathbf{x}^l$ is the input to layer $l$, $H$ is the number of hidden units and $\epsilon$ is a small constant for numerical stability.

It maintains the re-scaling invariance property to both weight and input but not the re-centering invariance property. Specifically, similar to LN it is invariant to full matrix weight scaling but not the single vector weight scaling.
Suppose for the full weight matrix scaling $W \to \delta W$ , where $\delta$ is scale factor then,
$$\text{RMS}(\delta x_i)=\delta\text{RMS}(x_i)$$
and,
$$\hat{x}_i' = \frac{\delta x_i}{\text{RMS}(\delta x)}=\hat{x}_i$$

[Zhang & Sennrich, 2019] demonstrated that RMSNorm matches LN in convergence speed and model accuracy across tasks (e.g., machine translation, language modeling) concluding re-scaling invariance not re-centering is the primary driver of LN’s success, justifying RMSNorm as a drop-in replacement.
By omitting mean subtraction and variance calculation, RMSNorm reduces the compute cost per step by  $7\%\sim64\%$ compared to LN [Zhang & Sennrich, 2019].


### Normalization in Attention
As Transformer architectures continue to scale in depth and width, stabilizing the attention mechanism has become increasingly important. One persistent challenge lies in the raw dot-product attention formulation: when query ($Q$) or key ($K$) vectors exhibit large magnitudes, their inner products ($QK^\top$) can produce extremely large values. This leads to softmax saturation and unstable gradients, necessitating normalization strategies.

#### QK Normalization
The original Transformer model [Vaswani et al., 2017] addressed above issue by scaling attention scores using a fixed factor $\frac{1}{\sqrt{d}}$, where $d$ is the dimensionality of the query and key vectors. However, this static scaling is not always sufficient in deeper or highly overparameterized networks.

QK Normalization (QKNorm) [Henrey et al., 2020] proposes a more adaptive solution by normalizing each row of the query and key matrices using the $\ell2$ norm before computing attention scores. Given $Q, K \in \mathbb{R}^{n \times d}$, QKNorm computes:
$$\tilde{Q}_i = \frac{Q_i}{\|Q_i\|_2}, \quad \tilde{K}_j = \frac{K_j}{\|K_j\|_2}$$

  This ensures all queries and keys lie on the unit hypersphere, constraining their dot product to the interval $[-1, 1]$. The attention scores are then computed as:
  $$S_{i,j} = \frac{\tilde{Q}_i \cdot \tilde{K}_j}{\tau}$$
  where $\tau$ is a learnable temperature parameter initialized to 1. This effectively replaces raw dot-product similarity with a scaled cosine similarity, reducing the risk of softmax saturation. The result more diffuse attention patterns stabilizing the training.

#### Placement of Normalization: Pre-Norm vs. Post-Norm
The placement of normalization layers, particularly LN, has a critical impact on Transformer training. Two main approaches have emerged:

##### Post-Norm
In the original Transformer design [Vaswani et al., 2017], normalization is applied after the residual connection. That is, the output of a sublayer (e.g., self-attention or feedforward) is added to its input, and the result is passed through LayerNorm:
$$\text{Output} = \text{LayerNorm}(X + \text{Sublayer}(X))$$
While this configuration can yield strong performance in relatively shallow models, it becomes unstable in deeper networks due to disrupted gradient flow during backpropagation necessitating learning rate warm-up to prevent divergence.

##### Pre-Norm
To improve training stability, Pre-Norm configuration was introduced [Xiong et al., 2020], where LN is applied before each sublayer:
$$\text{Output} = X + \text{Sublayer}(\text{LayerNorm}(X))$$
This design strengthens the identity path through residual connection, improving gradient propagation and convergence. Pre-Norm has since become the default choice in many modern large-scale Transformer architectures, such as GPT-3, PaLM, and LLaMA, due to its effectiveness in enabling deeper and more stable training.

However, while Pre-Norm improves stability, it may lead to suboptimal generalization. Compared to Post-Norm, which acts as a stronger form of regularization by normalizing after the residual addition, Pre-Norm tends to preserve signal structure at the cost of reduced model expressivity. Recent studies [Xie et al., 2023; Wang et al., 2024] show that while Pre-Norm trains faster and more stably, Post-Norm often yields better final accuracy in very deep settings provided training instabilities can be managed.

##### QKV-Norm and Circuit Collapse
A recent line of work investigates a subtle but important issue in Pre-Norm Transformers. [Menary et al. (2024)] identify what they term circuit collapse, where semantic subspaces within the hidden representations such as syntactic, positional, or world knowledge components interfere destructively due to shared normalization. Because Pre-Norm applies a single LayerNorm to the combined input of the attention module, it disrupts the independence of query, key, and value (QKV) subspaces, harming the selective focus that attention relies on.

To mitigate this, the authors propose QKV-Norm, where normalization is applied independently to the query, key, and value vectors after their respective linear projections.
Formally, QKV-Norm modifies attention as follows:
$$Q, K, V = \text{norm}(Q), \text{norm}(K), \text{norm}(V)$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$
This preserves the distinctiveness of semantic subspaces and improves internal circuit stability. QKV-Norm matches Pre-Norm performance on in-distribution tasks like language modeling, while significantly reducing interference in attention heads.
However, the authors note that this comes with a small trade-off in out-of-distribution generalization, reflecting a tension between stability and expressivity.

In a complementary study, [Rybakov et al. (2024)] further refine this idea by applying LayerNorm not only to the QK projections but also to the attention output (`Proj`) and feedforward layer output (`FC2`). Additionally, they combine this with softmax capping, limiting the influence of extremely high attention scores. These changes allow for training at 1.5× higher learning rates and improve perplexity.

#### HybridNorm
To balance the training stability of Pre-Norm with the regularization benefits of Post-Norm, recent efforts have explored hybrid normalization strategies such as Mix-LN [Li, Mix-LN, 2025] which applies Post-Norm to the earlier layers and Pre-Norm to the deeper layers. A more unified approach is HybridNorm [Zhuo, 2025], which integrates Pre-Norm and Post-Norm within each transformer block, specifically QKV-Norm in the Multi-Head Attention (MHA) module with Post-Norm in the Feedforward Network (FFN)
$$Y^l = \text{MHA}(X^l) + X^l, \quad X^{l+1} = \text{FFN}(\text{norm}(Y^l)) + \text{norm}(Y^l)$$

Further extending this idea, a variant, HybridNorm* a technique inspired from [DeepSeek-AI (2024)], where LN is applied before the first MHA block but Post-Norm is used afterward. The modified formulation is:
$$Y^0 = \text{MHA}(\text{norm}(X^0)) + X^0, \quad X^1 = \text{FFN}(\text{norm}(Y^0)) + \text{norm}(Y^0)$$

Empirical results show that both HybridNorm and HybridNorm* yield consistently lower training loss and validation perplexity compared to Pre-Norm alone. In particular, HybridNorm* demonstrates improved generalization across diverse tasks, with gains in BasicArithmetic (+3.11), HellaSwag (+1.71), and COPA (+3.78).