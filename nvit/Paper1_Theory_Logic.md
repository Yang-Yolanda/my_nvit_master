# Paper 1: Mathematical Framework for KTI-Guided Intervention

## 1. The Detector: Kinematic Mutual Information (KTI)
Let $H_l^i$ be the attention head $i$ at layer $l$. The attention matrix is $A_l \in \mathbb{R}^{N \times N}$, where $N$ is the number of tokens.
We define the **Physical Connectivity Prior** as the adjacency matrix $G_{skel} \in \{0, 1\}^{N \times N}$ derived from the SMPL kinematic tree.

The **Kinematic Mutual Information (KTI)** at layer $l$ is defined as the alignment between the attention distribution and the physical tree:

$$ KTI(l) = \frac{1}{H} \sum_{h=1}^{H} \sum_{(u, v) \in \mathcal{E}_{skel}} A_l^h(u, v) \cdot \log \frac{A_l^h(u, v)}{P(u)P(v)} $$

Where $\mathcal{E}_{skel}$ is the set of edges in the kinematic tree.
*   **Peak 1 (Detection)**: High KTI at early layers indicates finding body parts.
*   **Peak 2 (Assembly)**: High KTI at deep layers indicates enforcing kinematic constraints.

## 2. The Navigator: Guidance Strategy
We propose a **KTI-Adaptive Switching Function** $\sigma(l)$ that determines the masking strategy $M_l$ for layer $l$:

$$
M_l = 
\begin{cases} 
\text{No Mask}, & \text{if } l < \tau_{start} \text{ (Global Context)} \\
\text{Soft Mask}, & \text{if } \tau_{start} \le l < \tau_{rigid} \text{ (Mamba Proxy)} \\
\text{Hard Mask}, & \text{if } l \ge \tau_{rigid} \text{ (GCN Proxy)}
\end{cases}
$$

### Definitions
1.  **Soft Mask (Mamba-like)**:
    Enforces locality physics via a Gaussian decay on the patch grid distance $d(u,v)$:
    $$ A_{soft}(u, v) = \text{Softmax}( Q K^T / \sqrt{d} + B_{gauss}(u, v) ) $$
    Where $B_{gauss}(u, v) = -\frac{d(u,v)^2}{2\sigma^2}$. This encourages local continuity (diffusion) without severing connections.

2.  **Hard Mask (GCN-like)**:
    Enforces rigid topology constraints.
    $$ A_{hard}(u, v) = \begin{cases} A_{original}(u, v), & \text{if } (u,v) \in \mathcal{N}_{skel} \\ -\infty, & \text{otherwise} \end{cases} $$
    This is equivalent to a Graph Convolution operation on the kinematic tree.

### Switching Criterion
The switching thresholds $\tau$ are determined by the derivatives of the KTI curve:
*   $\tau_{start} = \text{argmax}_l \frac{\partial KTI}{\partial l}$ (Start structure when KTI rises)
*   $\tau_{rigid} = \text{argmax}_l KTI(l)$ (Lock structure when KTI peaks)

## 3. The Proof: Effective Rank
To prove redundancy removal, we measure the **Effective Rank (ER)** of the feature covariance matrix $\Sigma_l$ at layer $l$:

$$ ER(\Sigma_l) = \exp \left( - \sum_i \bar{\sigma}_i \log \bar{\sigma}_i \right) $$

where $\bar{\sigma}_i$ are the normalized singular values.
*   **Hypothesis**: In the "Hard Mask" region ($l \ge \tau_{rigid}$), the ER should drop significantly, indicating that the high-dimensional ViT features have collapsed into a low-dimensional manifold isomorphic to the kinematic tree.
