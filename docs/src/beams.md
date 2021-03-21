```@meta
CurrentModule = AngularPowerSpectra
```

# QuickPol

We provide utilities to compute beam matrices in the QuickPol formalism ([Hivon et al. 2017](https://arxiv.org/abs/1608.08833)). We introduce some additional steps here for computational efficiency. In this section, we use the indices ``\ell, \ell', \ell''`` such that we don't need to change indices at the end in order to match Hivon. Define a scaled version of the scan spectrum
```math
W_{\ell'}^{\nu_1,\nu_2,s_1,s_2,j_1,j_2} = \sum_{m^\prime=-\ell^\prime}^{\ell^\prime}  \left(_{s_1+\nu_1}\tilde{\omega}^{(j_1)}_{\ell^\prime m^\prime}\right)
    \left(_{s_2+\nu_2}\tilde{\omega}^{(j_2)}_{\ell^\prime m^\prime}\right)^*
```
Define the matrix,
```math
\begin{aligned}
\mathbf{\Xi}^{\nu_1,\nu_2,s_1,s_2,j_1,j_2}_{\ell^{\prime\prime},\ell} &= (-1)^{s_1 + s_2 + \nu_1 + \nu_2} \sum_{\ell^{\prime}} \, \rho_{j_1,\nu_1} \rho_{j_2, \nu_2}  W_{\ell'}^{\nu_1,\nu_2,s_1,s_2,j_1,j_2}  \\
 &\qquad\qquad \times \begin{pmatrix} \ell & \ell^{\prime} & \ell^{\prime\prime} \\
     -s_1 & s_1+\nu_1  & -\nu_1 \end{pmatrix} \begin{pmatrix}
     \ell & \ell^{\prime} & \ell^{\prime\prime} \\ -s_2 & s_2+\nu_2  & -\nu_2 \end{pmatrix}
\end{aligned}
```
This matrix is symmetric, and does not depend on ``u_1, u_2``. We can then write the beam matrix in terms of ``\mathbf{\Xi}``,
```math
\mathbf{B}_{\ell^{\prime\prime},\ell}^{\nu_1,\nu_2, u_1, u_2} \,= \sum_{j_1, j_2, s_1, s_2} \frac{2\ell + 1}{4\pi} \,_{u_1}\hat{b}^{(j_1)*}_{\ell, s_1} \,_{u_2}\hat{b}^{(j_2)*}_{\ell, s_2} \, \frac{k_{u_1} k_{u_2}}{k_{\nu_1} k_{\nu_2}} \, \mathbf{\Xi}^{\nu_1,\nu_2,s_1,s_2,j_1,j_2}_{\ell^{\prime\prime},\ell}
```
With this definition, the beam matrices ``\mathbf{B}`` are sub-blocks of the linear operator relating the cross-spectrum to the beamed cross-spectrum (Hivon+17 eq. 38),
```math
\tilde{C}^{\nu_1,\nu_2}_{\ell^{\prime\prime}} = \sum_{u_1,u_2}\left(\sum_{\ell} \mathbf{B}_{\ell^{\prime\prime},\ell}^{\nu_1,\nu_2, u_1, u_2} C_{\ell}^{u_1, u_2} \right).
```
Note that the inner sum is just a matrix-vector multiplication.



```@docs
quickpolΞ!
quickpolW
``` 
