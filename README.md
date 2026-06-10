# word2vec
an optimized Skip-gram with negative sampling implementation from first principles using only Python and Numpy



<img width="1536" height="754" alt="Figure_full" src="https://github.com/user-attachments/assets/15bb833d-7dbf-4363-a334-1dc8805d89d6" />








# Framework-Free Word2Vec: Tensor Geometry, Gradient Derivations, and Structural Convergence Dynamics

An academic implementation of the **Skip-gram with Negative Sampling (SGNS)** word embedding architecture, built entirely from first principles using **raw NumPy**. 

This research project bypasses abstract high-level automatic differentiation frameworks to explicitly derive and execute vectorized backpropagation, analyze coordinate-level gradient updates using **AdaGrad**, and evaluate semantic vector space convergence constraints under data-sparsity conditions.

---

## 1. Theoretical Framework & Mathematical Formulations

The core objective of the Skip-gram model is to learn distributed vector representations for words that are highly predictive of their surrounding context words within a fixed window size $T$.

### 1.1 Objective Function with Negative Sampling (SGNS)
To avoid computing the computationally intractable softmax over the entire vocabulary $V$, we optimize the negative sampling objective function. For a sequence of training words $w_1, w_2, \dots, w_T$, the localized objective function maximizing the log-probability of target-context pairs against a noise distribution is defined as:

$$\mathcal{L}_{\text{SGNS}} = -\sum_{i=1}^{T} \sum_{-C \le j \le C, j \neq 0} \left[ \log \sigma(\mathbf{v}_{w_{i+j}}^{\prime \top} \mathbf{v}_{w_i}) + \sum_{k=1}^{K} \mathbb{E}_{w_n \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_n}^{\prime \top} \mathbf{v}_{w_i}) \right] \right]$$

Where:
* $\mathbf{v}_{w}$ is the input (center) vector representation of word $w$.
* $\mathbf{v}_{w}^{\prime}$ is the output (context) vector representation of word $w$.
* $C$ is the sliding window radius.
* $K$ is the number of negative samples drawn per positive sample.
* $\sigma(x) = \frac{1}{1 + e^{-x}}$ represents the standard sigmoid non-linearity.
* $P_n(w)$ is the unigram noise distribution raised to the $3/4$ power to artificially boost the sampling frequency of rarer tokens: $P_n(w) = \frac{U(w)^{3/4}}{\sum_{j} U(w_j)^{3/4}}$.

---

## 2. 


### 2.1 Coordinate Mapping & Shape Tracking
To facilitate accelerated execution on concurrent CPU architectures, all batch calculations are fully vectorized. Let $B$ represent the mini-batch size, $D$ represent the designated embedding dimensionality, and $K$ represent the number of negative samples.

* **Target Tensors ($\mathbf{V}_{\text{target}}$):** Shape $\left(B, D\right)$
* **Context/Negative Tensors ($\mathbf{V}_{\text{context}}$):** Shape $\left(B, 1 + K, D\right)$

The forward pass calculation mapping batch-level dot products is computed cleanly via Einstein Summation notation (`np.einsum`), eliminating memory allocations for intermediate broadcasted tensors:

```python
# Fully vectorized batch evaluation of target-context dot products
# Formulates a localized matrix contraction over the embedding dimension 'd'
scores = np.einsum('bd,bkd->bk', target_vectors, context_vectors)  # Output Shape: (B, 1 + K)
```




### 2.2 AdaGrad
Traditional Stochastic Gradient Descent (SGD) applies a uniform learning rate $\eta$ across all parameter updates. In Natural Language Processing datasets, this creates a massive structural imbalance: frequent words (like `the`, `and`) generate massive, continuous gradient updates, while rare semantic words (like `king`, `queen`) receive sparse, infrequent updates.

To balance this, we implement a custom **AdaGrad (Adaptive Gradient Algorithm)** optimizer module. AdaGrad dynamically scales the learning rate for each individual feature coordinate based on the historical accumulation of squared gradients for that specific parameter.

For each parameter vector $\theta$, let $\nabla_\theta \mathcal{L}_t$ be the gradient at time step $t$. The historical running square-gradient diagonal matrix $\mathbf{G}_t \in \mathbb{R}^{D}$ accumulates variance over time:

$$\mathbf{G}_t = \mathbf{G}_{t-1} + \left(\nabla_\theta \mathcal{L}_t\right)^2$$

The parameter update rule is then formulated as a localized Hadamard (element-wise) product:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \nabla_\theta \mathcal{L}_t$$

Where:
* $\eta$ is the base global learning rate.
* $\epsilon$ is a smoothing term ($10^{-8}$) introduced to prevent catastrophic division-by-zero errors in uninitialized coordinates.
* $\odot$ represents the element-wise vector multiplication.

#### Research Implications for Embedding Convergence:
1. **Accelerated Learning for Rare Features:** Because rare words appear infrequently, their corresponding indices in $\mathbf{G}_t$ remain exceptionally small. As a result, the denominator $\sqrt{\mathbf{G}_t + \epsilon}$ remains tiny, forcing a significantly larger effective learning rate when these rare tokens *do* appear. This prevents rare word vectors from getting trapped near their random initializations.
2. **Monotonically Decreasing Step Size:** Because $\mathbf{G}_t$ accumulates squares of real numbers, every element in the cache grows monotonically over time. This naturally decays the learning rate as training progresses, allowing the model to aggressively explore the vector space early on and fine-tune its clustering boundaries smoothly in later epochs.




### 2.3 Research Hyperparameter Baseline
The empirical configurations utilized to stabilize the embedding space during our baseline testing are structured as follows:

| Hyperparameter | Baseline Value | Research Rationale |
| :--- | :--- | :--- |
| **Embedding Dim ($D$)** | `100` / `300` | Balancing spatial capacity against vocabulary density. |
| **Window Size ($C$)** | `5` | Captures a mix of structural syntax and semantic proximity. |
| **Negative Samples ($K$)** | `5` | Standard optimal trade-off boundary for smaller corpora. |
| **Sub-sampling ($t$)** | $10^{-4}$ | Artificially penalizes frequent stop-words to accelerate convergence. |
| **Base Learning Rate ($\eta$)**| `0.05` | Optimal initial step size bound for AdaGrad coordinate decay. |


---

## 3. References & Foundational Literature

This implementation is directly derived from the foundational mechanics established in the original Google Research and Stanford NLP publications. If utilizing this codebase for comparative baseline analysis, please reference the primary literature:

* **The Core Word2Vec Framework (Skip-gram Introduction):**
  Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781. 
  [Read Paper](https://arxiv.org/pdf/1301.3781)

* **Introduction of Negative Sampling (SGNS Optimization):**
  Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. In *Advances in Neural Information Processing Systems (NeurIPS)* (pp. 3111-3119).
  [Read Paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)



---
