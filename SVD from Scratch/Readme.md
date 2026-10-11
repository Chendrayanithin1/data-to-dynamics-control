# SVD via Eigendecomposition — Compared with NumPy

An implementation of the Singular Value Decomposition (SVD) built from the eigendecomposition of the correlation matrix $X^T X$, validated against `np.linalg.svd`.

Part of my self-study of *Data-Driven Science and Engineering* (Brunton & Kutz), Chapter 1.

> **Scope note:** the eigen-step uses `np.linalg.eigh`. Everything else (sorting, singular values, left singular vectors, basis completion, validation) is implemented by hand. A fully from-scratch version using power iteration is planned — see [Next steps](#next-steps).

---

## 1. Background

### The data matrix

Each column of $X$ is one snapshot of data, for example, one face image flattened into a vector:

$$
X = \begin{bmatrix} | & | & & | \\ x_1 & x_2 & \cdots & x_m \\ | & | & & | \end{bmatrix}, \qquad x_k \in \mathbb{R}^n
$$

(In the code, $X$ is called `A`.)

### The decomposition

$$
X = U \Sigma V^T
$$

- **$\Sigma$** is diagonal, with singular values ordered $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_m \ge 0$.
- **$U$** and **$V$** are unitary (orthogonal for real data):

$$
U^T U = U U^T = I_{n \times n}, \qquad V^T V = V V^T = I_{m \times m}
$$

> These identities hold for the **full** $U$. For the **economy** $\hat{U}$ ($n \times m$), only $\hat{U}^T \hat{U} = I_{m \times m}$ holds; $\hat{U}\hat{U}^T$ is a projection, not the identity. See [Section 3](#3-economy-vs-full-svd).

### Interpretation

- **Columns of $U$** — the dominant patterns in the data: "eigenfaces" for a face dataset, or eigen flow-field modes for fluid data.
- **$\Sigma$** — how important each pattern is; ordering by $\sigma$ ranks the patterns.
- **Columns of $V$** — how much of each pattern is present in each snapshot.

---

## 2. Method: SVD from the correlation matrices

The entries of $X^T X$ are inner products between snapshots:

$$
(X^T X)_{ij} = x_i^T x_j = \langle x_i, x_j \rangle
$$

Substituting the economy SVD $X = \hat{U}\hat{\Sigma}V^T$ and using $\hat{U}^T\hat{U} = I$:

$$
X^T X = V \hat{\Sigma} \hat{U}^T \hat{U} \hat{\Sigma} V^T = V \hat{\Sigma}^2 V^T
\quad\Longrightarrow\quad
X^T X \, V = V \hat{\Sigma}^2
$$

So the **columns of $V$ are the eigenvectors** of $X^T X$, and its **eigenvalues are $\sigma_i^2$**.

By the same argument for the other correlation matrix:

$$
X X^T = \hat{U} \hat{\Sigma} V^T V \hat{\Sigma} \hat{U}^T = \hat{U} \hat{\Sigma}^2 \hat{U}^T
\quad\Longrightarrow\quad
X X^T \hat{U} = \hat{U} \hat{\Sigma}^2
$$

### Algorithm

1. Form $X^T X$ ($m \times m$ — small when $n \gg m$).
2. Eigendecompose it to get eigenvalues $\lambda_i$ and eigenvectors $V$.
3. Sort eigenvalues in **descending** order and reorder the columns of $V$ to match.
4. Clip tiny negative eigenvalues (floating-point noise) to zero, then $\sigma_i = \sqrt{\lambda_i}$.
5. Recover the left singular vectors directly: $\hat{U} = X V \hat{\Sigma}^{-1}$, i.e. $u_i = X v_i / \sigma_i$.

---

## 3. Economy vs full SVD

For a tall $n \times m$ matrix, $\hat{U} = XV\hat{\Sigma}^{-1}$ gives only $m$ columns.
The full $U$ ($n \times n$) needs $n - m$ extra orthonormal columns. They are **not determined by $X$** — they multiply only zero rows of $\Sigma$ — so any orthonormal completion is valid. This implementation completes the basis with Gram–Schmidt on random vectors.

Consequence: the extra columns will not match NumPy's, and don't need to.

---

## 4. Validation

All comparisons are made against `np.linalg.svd`, with a fixed random seed.

**Sign ambiguity.** If $(u_i, v_i)$ is a valid singular pair, so is $(-u_i, -v_i)$. Element-wise differences between singular vectors are therefore meaningless. Vectors are compared with $|\langle u_i^{\text{mine}}, u_i^{\text{numpy}} \rangle|$, which should be $\approx 1$.

| Check | Metric | Result |
|---|---|---|
| Singular values | $\max_i \lvert \sigma_i - \sigma_i^{\text{np}} \rvert$ | _fill in_ |
| Left vectors | $\min_i \lvert \langle u_i, u_i^{\text{np}} \rangle \rvert$ | _fill in_ |
| Right vectors | $\min_i \lvert \langle v_i, v_i^{\text{np}} \rangle \rvert$ | _fill in_ |
| Reconstruction | $\lVert X - U\Sigma V^T \rVert_F$ | _fill in_ |
| Orthonormality of $U$ | $\lVert U^T U - I \rVert_F$ | _fill in_ |
| Orthonormality of $V$ | $\lVert V^T V - I \rVert_F$ | _fill in_ |

Figures (in `figures/`):

- Singular values: mine vs NumPy
- Absolute error per singular value (log scale)
- Error on a matrix with known singular values spanning many orders of magnitude

---

## 5. Limitations

- **Squared condition number.** Forming $X^T X$ squares the condition number of $X$, so the smallest singular values lose accuracy. This is why production SVD routines never form $X^T X$. The known-singular-value experiment above shows the effect.
- **Eigen-step uses `np.linalg.eigh`.** Not fully from scratch yet.
- **Zero singular values.** $u_i = Xv_i/\sigma_i$ is undefined when $\sigma_i = 0$ (rank-deficient input).
- **Wide matrices ($n < m$).** $X^T X$ then has $m - n$ zero eigenvalues. The fix is to decompose $X^T$ instead and swap the roles of $U$ and $V$.

---

## 6. Repository structure

```
svd-from-scratch/
├── svd.py                  # compute_svd()
├── svd_comparison.ipynb    # derivation, validation, figures
├── figures/
├── requirements.txt
└── README.md
```

## How to run

```
pip install -r requirements.txt
jupyter notebook svd_comparison.ipynb
```

---

## Next steps

- Replace `eigh` with **power iteration + deflation** for a fully from-scratch SVD.
- Use the SVD for **low-rank approximation** (image compression).
- **PCA** on a real dataset.

## Reference

S. L. Brunton and J. N. Kutz, *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*, Cambridge University Press — Chapter 1.
