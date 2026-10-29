# Part 3: Gradient Descent, Manual Calculation (matrix form)

Every number here was worked by hand and then re-checked by `part4_gd_code/gradient_descent.py`
(which also checks every gradient numerically with SciPy). Fill in the member names before submitting.

## 1. Setup

Model: y = m₁x₁ + m₂x₂ + b, written for all data points at once as

$$\hat{\mathbf y} = X\mathbf m + \mathbf b$$

$$
X=\begin{bmatrix}1 & 3\\ 4 & 10\end{bmatrix},\quad
\mathbf y=\begin{bmatrix}5\\ 6\end{bmatrix},\quad
\mathbf m^{(0)}=\begin{bmatrix}-1\\ 2\end{bmatrix},\quad
\mathbf b^{(0)}=\begin{bmatrix}1\\ 1\end{bmatrix},\quad
n=2,\quad \alpha = 0.01
$$

**Why α = 0.01?** The assignment does not give a learning rate. The curvature of J along its steepest direction is the largest eigenvalue of (2/n)AᵀA with A = [X | I], about 127.5, so gradient descent is only stable for α < 2/127.5 ≈ 0.0157. We picked α = 0.01: safely stable, but still big enough to see clear progress in three steps.

**How we treat b.** The assignment gives b as a vector [1, 1] and asks us not to treat values as scalars, so b is a 2×1 vector added row by row (b₁ belongs to data point 1, b₂ to data point 2).

## 2. Deriving the gradient of the MSE

Error vector:

$$\mathbf e = \hat{\mathbf y} - \mathbf y = X\mathbf m + \mathbf b - \mathbf y$$

Cost (Mean Squared Error):

$$J(\mathbf m,\mathbf b) = \frac1n\,\mathbf e^{\top}\mathbf e = \frac1n\sum_{i=1}^{n} e_i^2$$

**Derivative with respect to m.** By the chain rule, ∂J/∂e = (2/n)e and ∂e/∂m = X, so

$$\frac{\partial J}{\partial \mathbf m} = X^{\top}\frac{\partial J}{\partial \mathbf e} = \frac{2}{n}X^{\top}\mathbf e$$

Element by element this is ∂J/∂mⱼ = (2/n) Σᵢ xᵢⱼ eᵢ, which is the familiar scalar formula, but computed in one matrix product.

**Derivative with respect to b.** ∂e/∂b = I (the identity), so

$$\frac{\partial J}{\partial \mathbf b} = \frac{2}{n}\,\mathbf e$$

**Update rule.**

$$\mathbf m \leftarrow \mathbf m - \alpha\frac{\partial J}{\partial \mathbf m},\qquad \mathbf b \leftarrow \mathbf b - \alpha\frac{\partial J}{\partial \mathbf b}$$

With n = 2, the factor 2/n equals 1, which keeps the hand arithmetic short.

## 3. Update 1 (Member 1: ________)

**Prediction**

$$X\mathbf m=\begin{bmatrix}1(-1)+3(2)\\ 4(-1)+10(2)\end{bmatrix}=\begin{bmatrix}5\\ 16\end{bmatrix},\qquad
\hat{\mathbf y}=\begin{bmatrix}5\\ 16\end{bmatrix}+\begin{bmatrix}1\\ 1\end{bmatrix}=\begin{bmatrix}6\\ 17\end{bmatrix}$$

**Error and cost**

$$\mathbf e=\begin{bmatrix}6-5\\ 17-6\end{bmatrix}=\begin{bmatrix}1\\ 11\end{bmatrix},\qquad J=\tfrac12(1^2+11^2)=\tfrac{122}{2}=61$$

**Gradients**

$$X^{\top}\mathbf e=\begin{bmatrix}1 & 4\\ 3 & 10\end{bmatrix}\begin{bmatrix}1\\ 11\end{bmatrix}=\begin{bmatrix}1+44\\ 3+110\end{bmatrix}=\begin{bmatrix}45\\ 113\end{bmatrix}$$

$$\frac{\partial J}{\partial\mathbf m}=1\cdot\begin{bmatrix}45\\ 113\end{bmatrix},\qquad \frac{\partial J}{\partial\mathbf b}=\begin{bmatrix}1\\ 11\end{bmatrix}$$

**Update**

$$\mathbf m^{(1)}=\begin{bmatrix}-1\\ 2\end{bmatrix}-0.01\begin{bmatrix}45\\ 113\end{bmatrix}=\begin{bmatrix}-1.45\\ 0.87\end{bmatrix},\qquad
\mathbf b^{(1)}=\begin{bmatrix}1\\ 1\end{bmatrix}-0.01\begin{bmatrix}1\\ 11\end{bmatrix}=\begin{bmatrix}0.99\\ 0.89\end{bmatrix}$$

## 4. Update 2 (Member 2: ________)

**Prediction**

$$X\mathbf m^{(1)}=\begin{bmatrix}1(-1.45)+3(0.87)\\ 4(-1.45)+10(0.87)\end{bmatrix}=\begin{bmatrix}-1.45+2.61\\ -5.80+8.70\end{bmatrix}=\begin{bmatrix}1.16\\ 2.90\end{bmatrix},\qquad
\hat{\mathbf y}=\begin{bmatrix}1.16+0.99\\ 2.90+0.89\end{bmatrix}=\begin{bmatrix}2.15\\ 3.79\end{bmatrix}$$

**Error and cost**

$$\mathbf e=\begin{bmatrix}2.15-5\\ 3.79-6\end{bmatrix}=\begin{bmatrix}-2.85\\ -2.21\end{bmatrix},\qquad J=\tfrac12(8.1225+4.8841)=6.5033$$

**Gradients**

$$X^{\top}\mathbf e=\begin{bmatrix}1(-2.85)+4(-2.21)\\ 3(-2.85)+10(-2.21)\end{bmatrix}=\begin{bmatrix}-2.85-8.84\\ -8.55-22.10\end{bmatrix}=\begin{bmatrix}-11.69\\ -30.65\end{bmatrix},\qquad
\frac{\partial J}{\partial\mathbf b}=\begin{bmatrix}-2.85\\ -2.21\end{bmatrix}$$

**Update**

$$\mathbf m^{(2)}=\begin{bmatrix}-1.45\\ 0.87\end{bmatrix}-0.01\begin{bmatrix}-11.69\\ -30.65\end{bmatrix}=\begin{bmatrix}-1.3331\\ 1.1765\end{bmatrix},\qquad
\mathbf b^{(2)}=\begin{bmatrix}0.99\\ 0.89\end{bmatrix}-0.01\begin{bmatrix}-2.85\\ -2.21\end{bmatrix}=\begin{bmatrix}1.0185\\ 0.9121\end{bmatrix}$$

## 5. Update 3 (Member 3: ________)

**Prediction**

$$X\mathbf m^{(2)}=\begin{bmatrix}-1.3331+3(1.1765)\\ 4(-1.3331)+10(1.1765)\end{bmatrix}=\begin{bmatrix}-1.3331+3.5295\\ -5.3324+11.7650\end{bmatrix}=\begin{bmatrix}2.1964\\ 6.4326\end{bmatrix},\qquad
\hat{\mathbf y}=\begin{bmatrix}3.2149\\ 7.3447\end{bmatrix}$$

**Error and cost**

$$\mathbf e=\begin{bmatrix}3.2149-5\\ 7.3447-6\end{bmatrix}=\begin{bmatrix}-1.7851\\ 1.3447\end{bmatrix},\qquad J=\tfrac12(3.1866+1.8082)=2.4974$$

**Gradients**

$$X^{\top}\mathbf e=\begin{bmatrix}-1.7851+4(1.3447)\\ 3(-1.7851)+10(1.3447)\end{bmatrix}=\begin{bmatrix}-1.7851+5.3788\\ -5.3553+13.4470\end{bmatrix}=\begin{bmatrix}3.5937\\ 8.0917\end{bmatrix},\qquad
\frac{\partial J}{\partial\mathbf b}=\begin{bmatrix}-1.7851\\ 1.3447\end{bmatrix}$$

**Update**

$$\mathbf m^{(3)}=\begin{bmatrix}-1.3331-0.035937\\ 1.1765-0.080917\end{bmatrix}=\begin{bmatrix}-1.369037\\ 1.095583\end{bmatrix},\qquad
\mathbf b^{(3)}=\begin{bmatrix}1.0185+0.017851\\ 0.9121-0.013447\end{bmatrix}=\begin{bmatrix}1.036351\\ 0.898653\end{bmatrix}$$

**Check after the last update:** ŷ = [2.954063, 6.378335], e = [−2.045937, 0.378335], J = 2.1645.

## 6. Summary table

| After update | m₁ | m₂ | b₁ | b₂ | ŷ | MSE J |
|---|---|---|---|---|---|---|
| 0 (start) | −1.0000 | 2.0000 | 1.0000 | 1.0000 | [6, 17] | 61.0000 |
| 1 | −1.4500 | 0.8700 | 0.9900 | 0.8900 | [2.15, 3.79] | 6.5033 |
| 2 | −1.3331 | 1.1765 | 1.0185 | 0.9121 | [3.2149, 7.3447] | 2.4974 |
| 3 | −1.369037 | 1.095583 | 1.036351 | 0.898653 | [2.9541, 6.3783] | 2.1645 |

> If your group has a 4th or 5th member, run `python part4_gd_code/gradient_descent.py --members 4` (or 5) and copy the extra update from the printed log in exactly the same layout.

## 7. Trend and interpretation

- **The error falls every single update: 61 → 6.50 → 2.50 → 2.16.** So yes, m and b are moving in a direction that reduces the error.
- **The first step is huge and overshoots.** Both predictions start too high (e = [1, 11]), so update 1 cuts m₂ sharply from 2 to 0.87. Because x₂ values (3 and 10) are much bigger than x₁ values, m₂ gets the largest gradient and the largest correction. That pushes both predictions too low (e becomes [−2.85, −2.21]).
- **Then the parameters zig-zag back.** In update 2, m₂ rises to 1.18; in update 3 it eases down to 1.10. This back-and-forth in m₂ (and the mirror move in m₁) is the classic sign of a steep, narrow valley in J: the step bounces between its walls while the cost still drops.
- **The intercepts move slowly.** Both drop a little in update 1 (both points were over-predicted). After that they split: b₁ climbs (1.0185, then 1.0364) because point 1 stays under-predicted, while b₂ bounces (0.9121, then 0.8987) as point 2 flips from under- to over-predicted. Their gradients are 10× smaller than m₂'s, so they change by only hundredths per step.
- **Long run (from the code, 100+ updates):** MSE keeps falling towards 0. With 4 parameters and only 2 data points there are infinitely many exact fits, so gradient descent will drive the error to zero; it does not mean the model would generalise to new points.
