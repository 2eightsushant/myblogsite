+++
title = "A Note on Positional Encoding"
date = "2024-09-08"

[taxonomies]
tags=["rope","encoding","positional","attention"]

[extra]
comment = true
+++

## Permutation equivariance

A function $f(x_1, x_2, …, x_n)$ is permutation equivariant if permuting the inputs causes the outputs to be permuted in the same way:
Let $\pi$ be permutation function then,
$$f(\pi(x_1, x_2, …, x_n)) = \pi(f(x_1, x_2, …, x_n))$$
In other words, the function doesn’t care about the order of inputs internally, and if we shuffle inputs, the outputs get shuffled the same way.

Consider $X \in \mathbb{R}^{n \times d}$  be an input tokens, then we have attention function
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
where,
$Q = XW^Q \in \mathbb{R}^{3 \times d_k}$
$K = XW^K \in \mathbb{R}^{3 \times d_k}$
$V = XW^V \in \mathbb{R}^{3 \times d_v}$

Now let $P \in \mathbb{R}^{n \times n}$ be permutation matrix, then
$Q{\prime} = (PX)W^Q = P(Q),$ 
$K{\prime} = P(K),$ 
$V{\prime} = P(V),$ 

$$\text{Attn}(PX) = \text{softmax}\left( \frac{Q{\prime} K{\prime}^\top}{\sqrt{d_k}} \right)V{\prime} = \text{softmax}\left( \frac{(PQ)(PK)^\top}{\sqrt{d_k}} \right)PV$$
$$\text{Attn}(PX) = \text{softmax}\left( \frac{PQK^\top P^\top}{\sqrt{d_k}} \right)PV$$
$$\text{Attn}(PX) = \text{softmax}(PAP^\top)PV$$
where,
$A=\frac{QK^\top}{\sqrt{d_k}}$ 

Since softmax is applied row-wise, permuting both the rows and columns of $A$ and then applying softmax gives the same result as,
$$\text{Attn}(PX) = P.\text{softmax}(A)P^\top PV$$
then,
$$\text{Attn}(PX) = P.\text{softmax}(A).V$$
$$\text{Attn}(PX) = P.\text{Attn}(X)$$
This shows that attention is permutation equivariant that means if we permute the input rows (i.e. the tokens), the result of the dot products and softmax will also be permuted in the same way since matrix multiplications are row-wise independent and dot product attention is symmetric.

To break the permutation equivariance and make the attention sensitive to order we add positional encodings.

## Absolute positional encoding

Introduced in the original Transformer paper [Vaswani et al., 2017], sinusoidal positional encoding assigns a unique, deterministic vector to each token position in a sequence. These encodings are added to the input token embeddings to provide information about the token’s position.

To encode token position $p \in \mathbb{N}$ with embedding dimension $d \in 2\mathbb{N}$, the sinusoidal positional encoding $\text{PE}(p) \in \mathbb{R}^d$ is defined as:

$$
\mathrm{PE}_{p,2i} = \sin\!\left(p \cdot \omega_i\right), \quad
\mathrm{PE}_{p,2i+1} = \cos\!\left(p \cdot \omega_i\right), \quad
\text{for } 0 \leq i < \tfrac{d}{2}
$$

where the frequencies $\omega_i \in \mathbb{R}$ are logarithmically scaled:
$$\omega_i = 10000^{-2i/d}$$
This yields a position-dependent encoding vector where each pair of dimensions corresponds to a sinusoid at a distinct wavelength.
Figure>

Let $\text{PE}(p), \text{PE}(q) \in \mathbb{R}^d$ be the sinusoidal positional encodings for positions $p, q \in \mathbb{N}$, then the dot product of two such vectors is:
$$\langle \text{PE}(p), \text{PE}(q) \rangle = \sum_{i=0}^{d/2 - 1} \left[\sin(\omega_i p)\sin(\omega_i q) + \cos(\omega_i p)\cos(\omega_i q)\right]$$
Using the trigonometric identity we obtain:
$$\langle \text{PE}(p), \text{PE}(q) \rangle = \sum_{i=0}^{d/2 - 1} \cos\left(\omega_i (p - q)\right)$$
The dot product depends only on the relative offset $\Delta = p - q$, not on $p$ and $q$ individually even though the encoding is absolute, the attention mechanism can implicitly recover relative positions via dot products. However, this relation is only exact in the dot product between two positional encodings it does not hold between two embedded tokens with added position vectors because the addition of token embeddings breaks this structure.

Let $\Delta \in \mathbb{Z}$ be a relative offset. Using trigonometric addition identities:
$$\begin{aligned} \sin((p + \Delta)\omega_i) &= \sin(p\omega_i)\cos(\Delta\omega_i) + \cos(p\omega_i)\sin(\Delta\omega_i) \\ \cos((p + \Delta)\omega_i) &= \cos(p\omega_i)\cos(\Delta\omega_i) - \sin(p\omega_i)\sin(\Delta\omega_i) \end{aligned}$$
this gives us,
$$
\begin{bmatrix}
\sin(\omega_i(p+\Delta)) \\
\cos(\omega_i(p+\Delta))
\end{bmatrix}=
\underbrace{
\begin{bmatrix}
\cos(\omega_i \Delta) & \sin(\omega_i \Delta) \\
-\sin(\omega_i \Delta) & \cos(\omega_i \Delta)
\end{bmatrix}
}_{\text{Rotation matrix } R(\omega_i \Delta)}
\cdot
\begin{bmatrix}
\sin(\omega_i p) \\
\cos(\omega_i p)
\end{bmatrix}
$$
So,
$$\text{PE}(p + \Delta) = R(\Delta) \cdot \text{PE}(p)$$
where $R(\Delta) \in \mathbb{R}^{d \times d}$ is a block-diagonal rotation matrix, with each $2 \times 2$ block rotating by $\omega_i \Delta$.

[https://kazemnejad.com/blog/transformer_architecture_positional_encoding/]
[https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/]

Shifting position by $\Delta$ corresponds to rotating the embedding vector in each frequency subspace. These rotations are linear, allowing MLPs or attention to model shifts in position via linear operations on encodings. This property is central to RoPE as we see later.

Another variant of absolute positional embedding(APE) [Vaswani](https://proceedings.mlr.press/v119/liu20n/liu20n.pdf) is the learnable APE, in which each position $p$ in the input sequence is assigned a trainable vector $\mathbf{p}_p \in \mathbb{R}^{d}$, where $d$ is the embedding dimension. These positional embeddings are learned during training and added element wise to the token embeddings,
$$\mathbf{x}_p = \mathbf{e}_p + \mathbf{p}_p$$
where $\mathbf{e}_p$ is the token embedding at position $p$, and $\mathbf{x}_p$ is the resulting position aware input. However, because each $\mathbf{p}_p$ is tied to a specific position, learnable APEs do not generalize well to sequences longer than those seen during training, limiting extrapolation capability at inference. Moreover, it also introduces full $d$-dimensional paramater vector per position, increasing the model's memory footprint. 
[Vaswani] experimented with both learned and SPE and observed nearly identical results. They opted for the sinusoidal version due to its ability to generalize to sequence lengths longer than those encountered during training. Similarly, [Wang & Chen (2020)] found that learnable positional embeddings do not consistently outperform SPE. Nevertheless, prominent models like BERT and GPT have adopted learnable PE.

## Relative positional encoding

Sinusoidal encoding implicitly encodes the relative distance via dot product i.e.,
$$\langle \text{PE}(p), \text{PE}(q) \rangle = \sum_{i=0}^{d/2-1} \cos\left((p - q)\omega_i\right)$$
While in attention, the dot product is computed between query and key:
$$\text{Attention score} = \langle q_p, k_q \rangle$$
where,
$q_p = W_Q(x_p + \text{PE}(p))$,
$k_q = W_K(x_q + \text{PE}(q))$

The dot product is,

$\langle q_p, k_q \rangle = \langle W_Q(x_p + \text{PE}(p)),\, W_K(x_q + \text{PE}(q)) \rangle$
$= \langle W_Q x_p, W_K x_q \rangle . \langle W_Q x_p, W_K \text{PE}(q) \rangle . \langle W_Q \text{PE}(p), W_K x_q \rangle . \langle W_Q \text{PE}(p), W_K \text{PE}(q) \rangle$

We can see other than the last term depends on absolute positions and learned embeddings rather than relative position. The additive mix with token embeddings and projection layers breaks the relative structure and looses the relative information.

Instead of encoding absolute position $i$ into each token, define attention to depend explicitly on the relative offset $r = j - i$. In simple term, the idea is to incorporate relative position directly into the attention mechanism rather than adding it to the input embeddings [Shaw][Raffel et al 2020].

Let, $X = [x_1, x_2, \dots, x_L]^\top \in \mathbb{R}^{L \times d}$ be input sequence of token embeddings, $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_z}$ be learnable projection matrix for query, key and value respectively and $R^K \in \mathbb{R}^{(2L - 1) \times d_z}$ be learned relative positional embedding table for keys, indexed by relative distance $r = j - i \in [-L+1, L-1]$
Such that,
$Q = X W^Q \in \mathbb{R}^{L \times d_z}, K = X W^K \in \mathbb{R}^{L \times d_z}, V = X W^V \in \mathbb{R}^{L \times d_z}$

For query position $i$ and key position $j$, we can define the attention compatibility score $e_{ij} \in \mathbb{R}$ as:
$$e_{ij} = \frac{1}{\sqrt{d_z}} \left( Q_i^\top K_j + Q_i^\top R^K_{j - i} \right)$$
where,
$Q_i^\top K_j$ is standard content-based attention, 
$Q_i^\top R^K_{j - i}$ is relative position bias, and
$R^K_{j - i} \in \mathbb{R}^{d_z}$ is learnable relative key embedding for offset $j - i$

Now let $\delta \in \mathbb{Z}$, and define shifted positions $i{\prime} = i + \delta$, $j{\prime} = j + \delta$, so
$$e_{i + \delta, j + \delta} = \frac{1}{\sqrt{d_z}} \left( Q_{i + \delta}^\top K_{j + \delta} + Q_{i + \delta}^\top R_{j - i} \right)$$
Since the relative bias term $Q_i^\top R_{j - i}$ depends only on the relative offset $r = j - i$, but 
$$j{\prime} - i{\prime} = (j + \delta) - (i + \delta) = j - i$$
so, $Q_{i{\prime}}^\top R_{j{\prime} - i{\prime}} = Q_{i+\delta}^\top R_{j - i}$

Which means $e_{ij}$ is shift-invariant, preserving the relative structure.

In both sinusoidal encoding(SPE)  and relative encoding(RPE) we see decaying of attention score with the distance but the difference is in SPE it happens non-monotonic manner via dot product while in RPE it happens in monotonic fashion via learned bias per relative distance.

$$\text{SPE}(i)^\top \text{SPE}(j) = \sum_k \cos(\omega_k (i - j))$$

As $|i - j|$ grows, the sum of cosines averages out, leading to lower dot product.

In RPE,
$$e_{ij} = Q_i^\top K_j + Q_i^\top R_{j - i}$$
model often learns to assign lower values for $\| j - i \| \gg 0$, i.e., fewer attention weights to distant tokens, unless the task rewards long-range dependencies.
[Shaw] have reported RPE achieved consistent BLEU score improvements over SPE on WMT 2014 English to German and English to French translation tasks on both big and base model.

We know attention score for RPE relies on fixed learned window of relative position,
$$e_{ij} = Q_i^\top K_j + Q_i^\top R_{j - i}$$
Where $R_{j-i}$ is bounded on $j-i \in [-k,k]$ and typically $- k \ll L$

At inference, we generally have $L{\prime} > L_{\text{train}}$ so, for $|j-i|>k$, relative position is undefined so we either clip or mask zero which introduces spurious attention between distant unrelated tokens. This is why RPE fails to extrapolate on longer sequences.

Another issue is that RPE is structurally unsuitable for KV caching. During autoregressive inference, we generate per token at a time. To avoid the redundant computation of all previous keys and values at each step $t$ we cache it and update at every step, suppose

$Q_t = x_t W^Q \in \mathbb{R}^{1 \times d}$
$K_t = x_t W^K \in \mathbb{R}^{1 \times d}$
$V_t = x_t W^V \in \mathbb{R}^{1 \times d}$

Such that we cache the keys and values over previous steps in:

$$
\text{Key cache:}\quad 
\mathcal{K}_{1:t-1} =
\begin{bmatrix}
K_1 \\ K_2 \\ \cdots \\ K_{t-1}
\end{bmatrix}
\in \mathbb{R}^{(t-1) \times d}
$$

$$
\text{Value cache:}\quad 
\mathcal{V}_{1:t-1} =
\begin{bmatrix}
V_1 \\ V_2 \\ \cdots \\ V_{t-1}
\end{bmatrix}
\in \mathbb{R}^{(t-1) \times d}
$$

At decoding step $t$, we attend to cached $\{\mathcal{K}, \mathcal{V}\}_{j=1}^{t-1}$ and compute:
$$e_{tj} = Q_t^\top \mathcal{K}_j$$
$$\alpha_{tj} = \text{softmax}(e_{tj})$$
$$y_t = \sum_{j=1}^{t-1} \alpha_{tj} \mathcal{V}_j$$
However in RPE, we add position-dependent bias $R_{j - t}$, the attention score becomes:
$$e_{tj}^{\text{RPE}} = \frac{Q_t K_j^\top + Q_t R_{j - t}^\top}{\sqrt{d}}$$
The term $Q_t R_{j - t}^\top$ depends on the relative position between query $t$ and each key $j<t$ ,Since $t$ increases at every decoding step, the relative position $j - t$ changes making caching impossible, so it must be recomputed for each pair $(j, t)$ at every step violating the principle of KV caching. In simple, RPE introduces position-dependent bias based on relative offset $j - t$, but offset changes at every time step, so cached keys can’t be reused without recomputing the position bias.

Although RPE addresses the relative positional factor, it suffers from extrapolation and KV caching issues.

## Rotary positional encoding

Rotary Positional Embedding (RoPE) was introduced to overcome fundamental limitations of both sinusoidal and relative positional encodings. We saw SPE provides a fixed, non-learnable representation of absolute positions and encodes relative distance implicitly via dot products. However, this relative information is not directly accessible or linear. RPE on the other hand, such as bias-based or embedding-based schemes, makes relative positions explicit and improves modeling of local dependencies, but often lacks rotational symmetry, does not preserve relative geometry in the embedding space, and disrupts autoregressive inference by requiring re-computation of all query–key relative distances at each time step. Additionally, many RPE variants require full attention matrices, making them incompatible with efficient attention mechanisms such as Performer or Linformer. Rotary Positional Embedding (RoPE) addresses the key shortcomings of both sinusoidal and relative positional encodings by introducing position-dependent rotations in each frequency subspace of the query and key vectors. These rotations preserve relative positional relationships directly in the inner product, enabling linear operations such as attention to remain sensitive to relative distance without explicit pairwise bias terms. As proposed in RoFormer [Su et al., 2021], RoPE exhibits three important properties: 
(1) it supports flexible sequence lengths by allowing extrapolation beyond training positions through its continuous, bounded rotation mechanism; 
(2) it induces a natural decay in inter-token dependency as the relative distance increases, aligning with linguistic priors; and 
(3) it enables linear-time attention variants to incorporate relative position information efficiently, without requiring full attention matrices or breaking compatibility with key–value caching during autoregressive inference.

In SPE positional information are added to input token embeddings, in RPE it is added to attention scores or via learned biases whereas RoPE injects it directly to queries and keys via position dependent rotations.

Let $q \in \mathbb{R}^d$ and $k \in \mathbb{R}^d$ be the query and key embeddings of tokens at positions $m,n \in \mathbb{N}$ in the sequence, respectively. To incorporate the relative position $m - n$ into the query and key vectors by applying rotations lets define function $f(x, \ell)$ that applies a rotation to the token embedding $x$ at position $\ell$.
$$f(x, \ell) = x \cdot e^{i \ell \theta}$$
where, $x$ is the token embedding, $\ell$ is the position of the token in the sequence, $\theta$ is a fixed rotation angle for each position. The inner product between $q$ and $k$ gives us the attention score,
$$\text{Attention Score}(q, k) = \langle f(q, m), f(k, n) \rangle$$
$$\langle f(q, m), f(k, n) \rangle = \langle q \cdot e^{i m \theta}, k \cdot e^{i n \theta} \rangle$$
$$\langle f(q, m), f(k, n) \rangle = \langle q, k \rangle \cdot e^{i (m - n) \theta}$$
Thus, we see that the attention score is influenced by the relative position $m - n$ between the tokens. The term $e^{i (m - n) \theta}$ encodes the relative positional information, which is implicitly introduced by the rotation. The real valued attention score can be stated as,
$$\text{Re} \left[ \langle q, k \rangle \cdot e^{i(m - n)\theta} \right] = \langle q, k \rangle \cdot \cos((m - n)\theta)$$
When $m=n$ , we get the full attention strength and conversely as $|m-n|$ increases $cos$ oscillates, and if we sum over multiple such terms suppose dimension of $d/2$, then the total sum is
$$\sum_{j=1}^{d/2} \langle q_j, k_j \rangle \cos((m - n)\theta_j)$$
Due to the orthogonality of frequency components it naturally decays as with increasing $|m - n|$. In other word, the net dot product approaches zero, 
$$\text{Attention score} \approx \sum_i \langle q_j, k_j \rangle \cos((m - n)\theta_j) \to 0 \quad \text{as } |m - n| \to \infty$$ 
Hence RoPE implicitly imposes a decaying correlation between positions m and n in the attention score showing the second property of RoPE. It believes words that are closer together tend to be more syntactically or semantically related than distant ones.

The terms $e^{i m \theta}$ or $e^{i n \theta}$ represents a complex exponential that can be interpreted as a rotation in the complex plane. This is an element of a unit circle in the complex plane, and it can be written using Euler’s formula:
$$e^{i m \theta} = \cos(m \theta) + i \sin(m \theta)$$
$$e^{i n \theta} = \cos(n \theta) + i \sin(n \theta)$$

It essentially represents a 2D rotation. We can represent it in 2D rotation matrix that rotates a vector by an angle $m\theta$ is given by:
$$R(m\theta) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}$$
Figure rot mat>

RoPE does not rotate the entire vector at once but cleverly splits the d-dimensional vector space into $d/2$ pairs or sub-spaces and combine them to avoid the complexities of high dimension.
Consider embedding vector in $\mathbb{R}^d$ (where d is even) with token's absolute position as $m \in \mathbb{N}$, then the embedding vector space can be divided into $d/2$ pairs of 2D subspaces. Such that each pair of dimensions is independently rotated by a corresponding angle $\theta_i$. Specifically, for each pair $(x_{2i-1}, x_{2i})$, the rotation is given by a 2D rotation matrix $R_{\theta_i}$:
$$
R_{\theta_i} \cdot x = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}
$$
$$ = \begin{pmatrix} x_{2i-1} \cos(m\theta_i) - x_{2i} \sin(m\theta_i) \\ x_{2i-1} \sin(m\theta_i) + x_{2i} \cos(m\theta_i) \end{pmatrix}$$
where $\theta_i = 10000−2(i−1)/d, \quad i∈[1,2,...,d/2]$ 

To represent the entire rotation in $\mathbb{R}^d$ for token position $m$, we construct a block-diagonal matrix, where each block corresponds to the 2D rotation matrix for a pair of dimensions. For d-dimensional space, the rotation matrix $R_{\theta,i}$ can be written as:
$$R_{\theta,i} = \begin{pmatrix} R_{\theta_1} & 0 & 0 & \cdots & 0 \\ 0 & R_{\theta_2} & 0 & \cdots & 0 \\ 0 & 0 & R_{\theta_3} & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_{\theta_{d/2}} \end{pmatrix}$$
where each $R_{\theta_i}$ is a 2D rotation matrix for the corresponding $i^{th}$ pair of dimensions.

We can see that there is no constraint that $m$ must be $\leq L$. Since $\cos(m \theta),\sin(m \theta)$ is well defined for any $m \in \mathbb{N}$. In other word, even at positions longer than seen in training, for example $m > 512$, RoPE can still produce a rotation. The rotation matrix entries satisfy:
$$\cos^2(m\theta_i) + \sin^2(m\theta_i) = 1$$
So RoPE does not explode or vanish. It rotates, keeping the norm constant. This shows the first property of RoPE.

Unlike SPE and RPE, RoPE encodes relative positional information directly into the query and key vectors via position dependent rotations, avoiding the need for explicit pairwise positional biases which would break the structure required for linear-time attention. Since these rotations are linear and preserve the bilinearity of the inner product, RoPE is compatible with kernelized attention mechanisms of the form:
$$
\text{Attention}(Q, K, V) \approx \phi(Q) \left( \phi(K)^\top V \right),
$$
where $\phi(q, m) = \phi(\mathbf{R}_m q)$ is kernel feature map. This allows RoPE to integrate into linear attention architecture such as Performer and Linformer. Moreover, RoPE applies a deterministic, position-dependent rotation to queries and keys independently at inference time, cached key–value pairs can be reused without recomputation. Specifically, only the incoming query at position $m$ needs to be rotated by $R_m$, while the previously cached keys $k_n$ remain fixed with their pre-applied $R_n$ rotation. This is because the relative positional information is preserved in the dot product
$$\langle R_m q, R_n k \rangle = \langle q, k \rangle \cdot \cos((m-n)\theta)$$
which depends only on the relative distance $m - n$, not requiring full recomputation. This shows the third property of RoPE.

[Su et al., 2021] demonstrated that integrating RoPE into the Transformer (Roformer) improved translation quality, yielding higher BLEU scores compared to the original Transformer [Vaswani et al., 2017]. More notably, applying RoPE to models such as BERT and Performer resulted in faster convergence during training.