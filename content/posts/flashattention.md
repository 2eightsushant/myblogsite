+++
title = "A note on Flash Attention"
date = "2024-08-30"

[taxonomies]
tags=["flashattention","optimization","fuse"]

[extra]
comment = true
+++

Performance is one of the critical factor in training or fine tuning large scale models. The performance efficiency of a training architecture can be studied through three primary performance bottlenecks: compute bound, memory bound and overhead bound as categorized by [Horace He]. In LLMs, the self attention mechanism is the most computationally intensive component, primarily due to its quadratic time and memory complexity with respect to the sequence length. Since each token in the input sequence attends to every other token, the computational complexity scales as $O(n^2)$ , where $n$ is the sequence length.[Linformer | https://arxiv.org/pdf/2209.04881].
Furthermore, prior work [https://arxiv.org/pdf/2007.00072] have shown that tensor contractions (matrix multiplications) dominates the FLOPs compared to other operations like normalization or point wise[https://arxiv.org/pdf/2007.00072]. Modern GPUs are extremely powerful achieving up to $312$ TFLOPS in A100 SXM4 and up to $1979$ TFLOPS in H100 SXM5 for FP16 Tensor cores [https://www.nvidia.com/en-us/data-center/h100/], making compute bound a less of a concern. Also, PyTorch asynchronously executes the CUDA kernels, overlapping CPU and GPU operation(large enough) effectively hiding its overheads making the overhead bound of less of a concern too. As a result, the limiting factor in performance increasingly shift towards memory bandwidth making memory bound more of a concern. For instance, the H100 SXM GPU offers a peak memory bandwidth of 3.35 TB/sec and a computational throughput of 67 TFLOPS/sec for FP16. In other word, the GPU can perform 67 trillion operations per second, it can only load approximately 837.5 billion FP16 values per second from global memory. This means that the more we read and write to global memory (DRAM/HBM) the more performance will be under memory bound bottleneck stalling the GPU and overworking on data movement.
fig>
A key technique to reduce this memory traffic and accelerate memory bound computation is operator fusion, which reduces intermediate memory access by fusing multiple operations into a single kernel. This minimizes data materialization and exploits the GPU compute power, shifting workloads closer to the compute-bound regime from memory-bound regime.
Coming back to attention unit which holds the major compute proportion where we often have memory bound regime. Let single head scaled dot-product attention be defined as:
$$\text{Attn}(Q,K,V) = \text{softmax}\left(QK^\top \right)V$$

where, $Q, K, V \in \mathbb{R}^{n \times d}$, and the batch and head dimensions are abstracted for simplicity. Also, element-wise operator such as scaling, masking and dropout are abstracted.
The intermediate matrix $QK^\top \in \mathbb{R}^{n \times n}$ has quadratic size. To compute it, both $Q \in \mathbb{R}^{n\times d}$ and $K \in \mathbb{R}^{n\times d}$ are loaded from HBM to SRAM and the resulting matrix $QK^\top$ is written back to HBM, incurring $O(n^2)$ writes along two $O(nd)$ reads.
Then to compute safe three passes over the $n \times n$ matrix are required:
1. The first pass computes the row-wise maximum values
2. The second pass computes the exponentials after subtracting the max values and accumulates row-wise sums
3. The third pass normalizes the values by dividing by the row sums
Each pass needed incurs $O(n^2)$ for both reads and writes, leading to a total of $3 \times O(n^2)$ memory accesses.
Following the softmax, matrix multiplication between the resulting probability matrix $S \in \mathbb{R}^{n \times n}$ and value matrix $V \in \mathbb{R}^{n \times d}$ is computed. This requires $O(n^2)$ and $O(nd)$ reads for $S$ and $V$ respectively and $O(nd)$ writes for the final output.
In summary, the standard attention kernel incurs 5 full passes, where each passes moves $O(n^2)$ data, that means the total memory traffic is $O(n^2)$ per pass and $5 \times O(n^2)$ overall memory access.
We can see that memory access patterns in standard attention computation scales with both the number of head units and the size of intermediate matrices, leading to potential memory bandwidth bottlenecks. There are two major issues here, one is the quadratic size of the score matrix and other is the frequent data movement between HBM and SRAM.

### FlashAttention
With the more need to increase the context length, the more attention computation scales being a main bottleneck for the performance. Various approximation techniques such as factorization have been developed to counter. While they have increased the performance but had to trade off the precision.
Flash attention is an IO aware, fused attention algorithm introduced by Tri Dao [Dao, 2022]. It leverages tiling and scaling techniques to significantly reduce both memory usage and number of passes while preserving exactness of softmax attention. It substantially reduced the memory usage from $O(n^2)$ to $O(n)$ and speed the performance of $3 \times$ over baseline model like GPT2 on wall clock. It leverages techniques such as tiling, operator fusion and re-computation.
#### Tiling
Let $Q, K \in \mathbb{R}^{n \times d}$ be query and key matrices and score matrix be $S=QK^\top \in \mathbb{R}^{n \times n}$ . Suppose the $S$ is partitioned into blocks of size $b \times b$, such that $N = \frac{n}{b}$ and $n$ is divisible by $b$. Then, $S$ can be divided into $N \times N$ grid of blocks, where each block is $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{b \times b}$.
Here $Q_i, K_j \in \mathbb{R}^{b \times d}$ represents the $i^{th}$ and $j^{th}$ row blocks of the $Q$ and $K$, respectively.
We can represent the blocked structure as:
$$Q = \begin{bmatrix} Q_1 \\ Q_2 \\ \vdots \\ Q_{N} \end{bmatrix}, \quad Q_i \in \mathbb{R}^{b \times d}, \quad K = \begin{bmatrix} K_1 \\ K_2 \\ \vdots \\ K_{N} \end{bmatrix}, \quad K_j \in \mathbb{R}^{b \times d}$$

$$QK^\top = \begin{bmatrix} Q_1 K_1^\top & \cdots & Q_1 K_{N}^\top \\ \vdots & \ddots & \vdots \\ Q_{N} K_1^\top & \cdots & Q_{N} K_{N}^\top \end{bmatrix}, \quad \text{each block } Q_i K_j^\top \in \mathbb{R}^{b \times b}$$

For each block $S_{ij}$,
- $Q_i$ is loaded once per row block and reused across all column $N$ blocks $K_j, \ \ j =(1, \cdots , N)$, 
- $K_j$ is loaded once per column block and reused across all row blocks $Q_i  \ \ i =(1, \cdots , N)$,
- $S_{ij}$ is computed in SRAM without materialization of full score matrix,

The idea is, instead of loading full $Q$ and $K$ at once, we partition them into blocks and load each block once reusing it across multiple computation. This enables sequential computation of attention scores without materializing full $S$. The scaled softmax is then applied incrementally to these blocks.

#### Online Safe softmax
For a score block $S \in \mathbb{R}^{b \times b}$, we compute attention weights $A_i = \text{softmax}(S_i)$ row-wise. For row $S_i = [S_{i1},\cdots,S_{ib}]$, the output is,
$$A_{ij}=\frac{e^{S_{ij}-m_b}}{\sum_{k=1}^b{e^{S_{ik}-m_b}}},\quad m_b=\max_{1\leq k \leq b}S_{ik}$$
The standard safe softmax implementation ensures numerical stability, requiring three passes over each row:
Initialize: $m_0=-\infty, \quad d_0=0$
Compute maximum row-wise:
	for $j = 1, \cdots ,b:$  $m_j=\text{max}(m_{j-1},S_{ij})$
Safe softmaxing:
	for $j = 1, \cdots ,b:$ $d_j = d_{j-1}+e^{S_{ij}-m_b}$
	where $d_{j-1}=\sum_k^{j-1}{e^{S_{ik}-m_{b}}}$
Normalize:
	for $j = 1, \cdots ,b:$ $A_{ij}=\frac{(e^{S_{ij}-m_b})}{d_b}$

Each pass materializes intermediate values, such as the vector of exponentials $[e^{S_{i1}-m_b}, \cdots, e^{S_{ib}-m_b}]$, incurring $O(b)$ memory writes per row. This becomes a significant bottleneck for large block size $b$. This traffic can be reduced with a simple but cleverly exploiting the exponent rule.
$$\left(\sum_i{e^{a_i-b_j}}\right)e^c_j=\left(\sum_i{e^{a_i-c_j}}\right)e^b_j$$
The online softmax algorithm [Milakov 2018], reduces the overhead by combining the maximum and sum passes, leveraging above exponent rule. The idea is to rescale the summation as,
$$\left(\sum_k^{j-1}{e^{S_{ik}-m_{j}}}\right)e^{m_{j-1}-m_{j-1}}+e^{S_{ij}-m_j}=\left(\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}\right)e^{m_{j-1}-m_j}+e^{S_{ij}-m_j}$$
The algorithm proceeds as,
Initialize: $m_0=-\infty, \quad d_0=0$
Compute maximum row-wise and safe softmax online:
	for $j = 1, \cdots ,b:$
		$m_j=\text{max}(m_{j-1},S_{i,j})$
		$d_j = d_{j-1}e^{m_{j-1}-m_j}+e^{S_{ij}-m_j}$
Normalize:
	for $j = 1, \cdots ,b:$ $A_{ij}=\frac{(e^{S_{ij}-m_b})}{d_b}$

here, $d_{j-1}=\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}$ , so
$$d_{j}=\left(\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}\right)e^{m_{j-1}-m_j}+e^{S_{ij}-m_j}$$
Consider two cases:
If $m_j=m_{j-1}$, that means maximum is still the previous maximum:
$$d_{j}=\left(\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}\right)e^{m_{j-1}-m_{j-1}}+e^{S_{ij}-m_{j-1}}$$
$$d_{j}=\left(\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}\right)+e^{S_{ij}-m_{j-1}}$$
$$d_j=d_{j-1}+e^{S_{ij}-m_{j-1}}$$
If $m_j=S_{ij}$, that means max is the new value:
$$d_{j}=\left(\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}\right)e^{m_{j-1}-S_{ij}}+e^{S_{ij}-S{ij}}$$
$$d_{j}=\left(\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}\right)e^{m_{j-1}-S_{ij}}+1$$$$d_j=d_{j-1}e^{m_{j-1}-S_{ij}}+1$$
This formulation eliminates the need to materialize the intermediate exponent vector, thereby reducing the number of memory passes from three to two.
#### Fusing to single pass
Flash attention[Dao, 2022] extends this idea to compute the attention output $O_{i} = \sum_{j=1}^{b}{A_{ij}V_{j}}$ where $V_j$ is the $j^{th}$ row of the value matrix $V$; in a single pass, avoiding materialization of $S$ or $A$. A standard two pass approach is computed as:
Initialize: $m_0=-\infty, \quad d_0=0, \quad O_0=0$
First pass: Compute attention score, maximum row-wise and safe softmax online
	for $j = 1, \cdots ,b:$
		$S_{ij} = Q_i K_j^\top$
		$m_j=\text{max}(m_{j-1},S_{ij})$
		$d_j = d_{j-1}e^{m_{j-1}-m_j}+e^{S_{ij}-m_j}$
Second pass: Normalize and accumulate output 
	for $j = 1, \cdots ,b:$
		$A_{ij}=\frac{(e^{S_{ij}-m_b})}{d_b}$
		$O_{j} = O_{j-1}+A_{ij}V_{j}$
where $O_{j-1}=\sum_k^{j-1}{A_{ik}V_{k}}$

However, flash attention computes $O_j$ incrementally in one pass using a scaling trick similar to the online softmax with safety. The core of rescaling is:
$$\left(\sum_k^{j-1}{\frac{e^{S_{ik}-m_j}}{d_j}}V_{k}\right)\frac{d_{j-1}}{d_{j-1}}\frac{e^{m_{j-1}}}{e^{m_{j-1}}}+\frac{e^{S_{ij}-m_j}}{d_j}V_{j}=\left(\sum_k^{j-1}{\frac{e^{S_{ik}-m_{j-1}}}{d_{j-1}}}V_{k}\right)\frac{d_{j-1}}{d_{j}}\frac{e^{m_{j-1}}}{e^{m_{j}}}+\frac{e^{S_{ij}-m_j}}{d_j}$$
So we have algorithm as,
Initialize: $m_0=-\infty, \quad d_0=0, \quad O_0= \mathbf{0}$
Single pass: Compute attention score, maximum row-wise, update sum and update output:
	for $j = 1, \cdots ,b:$
		$S_{ij} = Q_i K_j^\top$
		$m_j=\text{max}(m_{j-1},S_{ij})$
		$d_j = d_{j-1}e^{m_{j-1}-m_j}+e^{S_{ij}-m_j}$
		$O_{j} = O_{j-1}\frac{d_{j-1}}{d_{j}}e^{m_{j-1}-m_{j}}+\frac{e^{S_{ij}-m_j}}{d_j}V_{j}$
where $d_{j-1}=\sum_k^{j-1}{e^{S_{ik}-m_{j-1}}}, \quad O_{j-1}=\left(\sum_k^{j-1}{\frac{e^{S_{ik}-m_{j-1}}}{d_{j-1}}V_{k}}\right)$

Note on notation: The term $e^{S_{ij} - m_j}$ in output update corresponds to $e^{\tilde m_{ij} - m_i^{\text{new}}}P_{ij}$, where $P_{ij}=e^{S_{ij} - \tilde m_{ij}}$ in [Dao, 2022].

Consider the cases:
If $m_j=m_{j-1}$ we get the usual iteration,
$$d_j=d_{j-1}+e^{S_{ij}-m_{j-1}}$$
$$O_{j} = \left(\sum_k^{j-1}{\frac{e^{S_{ik}-m_{j-1}}}{d_{j-1}}}V_{k}\right)\frac{d_{j-1}}{d_{j}}e^{m_{j-1}-m_{j-1}}+\frac{e^{S_{ij}-m_{j-1}}}{d_j}V_{j}$$
$$O_{j} = O_{j-1}\frac{d_{j-1}}{d_{j}}+\frac{e^{S_{ij}-m_{j-1}}}{d_j}V_{j}$$

If $m_j=S_{ij}$ we get the new scaled output,
$$d_j=d_{j-1}e^{m_{j-1}-S_{ij}}+1$$
$$O_{j} = \left(\sum_k^{j-1}{\frac{e^{S_{ik}-m_{j-1}}}{d_{j-1}}}V_{k}\right)\frac{d_{j-1}}{d_{j}}e^{m_{j-1}-S_{ij}}+\frac{e^{S_{ij}-S_{ij}}}{d_j}V_{j}$$
$$O_{j} = O_{j-1}\frac{d_{j-1}}{d_{j}}e^{m_{j-1}-S_{ij}}+\frac{V_{j}}{d_j}$$

Flash attention maintains running statistics $m_j$, $d_j$ and $O_j$. The update ensures:
$$O_{j}=\sum_k^{j}{\frac{e^{S_{ik}-m_{j}}}{d_{j}}V_{k}}$$
And at $j=b$:
$$O_{b}=\sum_{j=1}^{b}{\frac{e^{S_{ij}-m_{b}}}{d_{b}}V_{j}}=\sum_{j=1}^{b}{A_{ij}V_j} = O_i$$
The result $O_b$ with single pass matches the standard attention output.

#### Backward pass with recomputation
To compute gradients $\nabla_{Q_i}L$, $\nabla_{K_j}L$ and $\nabla_{V_j}L$, Flash attention recomputes $S_{ij}$ and $A_{ij}$ to avoid storing them. The forward pass stores $Q_i, K_j, V_j, O_i$ and statistics $m_b, d_b$. The backward pass proceeds block wise that uses two passes:
Initialize: $\Delta_i= 0, \nabla_{Q_i}=\mathbf{0}, \nabla_{V_j}=\mathbf{0}, \nabla_{K_j}=\mathbf{0}$ for $j=1, \cdots , b$
Pass 1: Compute intermediate gradients:
	for $j = 1, \cdots ,b:$
	    $S_{ij} = Q_i K_j^\top$
	    $A_{ij} = \frac{e^{S_{ij} - m_b}}{d_b}$
	    $\Delta_i = \Delta_i + (\nabla_{O_i} L · V_j) A_{ij}$
	    $\nabla_{V_j} = \nabla_{V_j} + A_{ij} (\nabla_{O_i} L)$
where $\Delta_i = \sum_{l=1}^b{(\nabla_{O_i}L.V_l)A_{il}}$, and $\nabla_{V_j}L=\frac{\partial L}{\partial O_i}.\frac{\partial O_i}{\partial V_j}=\nabla_{O_i}L.A_{ij}$

Pass 2: Compute score and input gradients
for $j = 1, \cdots, b$:
    $S_{ij} = Q_i K_j^\top$
    $A_{ij} = \frac{e^{S_{ij} - m_b}}{d_b}$
    $\nabla_{S_{ij}} L = A_{ij} [(\nabla_{O_i} L · V_j) - \Delta_i]$
    $\nabla_{Q_i} = \nabla_{Q_i} + (\nabla_{S_{ij}} L) K_j$
    $\nabla_{K_j} = \nabla_{K_j} + (\nabla_{S_{ij}} L) Q_i$

Since both $S_{ij}$ and $A_{ij}$ are not stored, it is recomputed in second pass. The gradient of the loss with respect to $S_{ij}$ is derived via the softmax Jacobian:
$$\frac{\partial A_{il}}{\partial S_{ij}} =  \begin{cases}  A_{ij}(1 - A_{ij}) & \text{if } l = j \\  -A_{il} A_{il} & \text{if } l \neq j  \end{cases}$$
The gradient is:
$$\nabla_{S_{ij}}L=\sum_{l=1}^b{\frac{\partial L}{\partial A_{ij}}.\frac{\partial A_{il}}{\partial S_{ij}}}=\sum_{l=1}^b{(\nabla_{O_i}L.V_l)\frac{\partial A_{il}}{\partial S_{ij}}}$$
$$=(\nabla_{O_{i}}L.V_j)A_{ij}(1-A_{ij})-\sum_{l \neq j}{(\nabla_{O_i}L.V) A_{il}A_{ij}}$$
$$=A_{ij}\left[(\nabla_{O_i}L.V_j)-\sum_{l=1}^b(\nabla_{O_i}L.V_l)A_{il}\right]$$
$$= A_{ij} \left[(\nabla_{O_i} L · V_j) - \Delta_i \right]$$
where $\Delta_i=\sum_{l=1}^b{(\nabla_{O_i}L.V_l)A_{il}}$ 

The first pass computes $\nabla_{V_j}$ while second pass computes $\nabla_{Q_i}$ and $\nabla_{K_j}$ where each pass recomputes $S_{ij}$ and $A_{ij}$, using $O(b)$ additional memory per block, reducing the memory footprint from $O(n^2)$ to $O(n)$ for the sequence length $n$.

Flash attention revolutionizes the self attention through IO aware tiling and a single pass algorithm that fuses score computation, reducing memory usage from $O(n^2)$ to $O(n)$ while maintaining numerical precision. Empirically, it achieves up to $3\times$ speedup on GPT-2, $15\%$ faster BERT-large training, and $2.4×$ speedup on Long-Range Arena (LRA) benchmark. It improves model quality with $0.7$ lower perplexity on GPT-2 and $6.4$ point gains in long-document classification [Dao, 2022].

#### Limitations
Although flash attention achieves $2 \sim 4 \times$ speed up over standard attention, it has only achieved the $30 \sim 50 \%$ in forward pass and $25 \sim 35 \%$ in backward pass of the theoretical maximum of FLOPs/sec of the A100 GPU, while GEMM can achieve $80 \sim 90 \%$ of it. [Dao, 2023] found out that it was suboptimal work partitioning between thread blocks and warps that was limiting Flash attention. Therefore, he addresses this limitation with three improvements: Reduction of non-matmul FLOPs, Parallelization and Partitioning.