

@doc raw"""
    clquickpol(nu₁, nu₂, b₁, b₂, ρ₁, ρ₂, ω₁, ω₂)

Compute the cross-power spectrum of two maps with spins ``\nu_1`` and ``\nu_2``
(Hivon et al. 2016, eq. 38)

```math
\begin{aligned}
\tilde{C}_{\ell^{\prime\prime}}^{\nu_1 \nu_2}  &= \sum_{u_1, u_2, j_1, j_2, \ell, s_1, s_2}
    (-1)^{s_1 + s_1 + \nu_1 + \nu_2} C_{\ell}^{u_1 u_2} \frac{2\ell+1}{4\pi}
    \,_{u_1}\hat{b}^{(j_1)*}_{\ell s_1} \,_{u_2}\hat{b}^{(j_2)*}_{\ell s_2} \\
&\qquad\qquad \times \frac{k_{u_1} k_{u_2}}{k_{\nu_1}k_{\nu_2}} \sum_{\ell^\prime m^\prime}
    \rho_{j_1, \nu_1} \rho_{j_2, \nu_2} (_{s_1+\nu_1}\tilde{\omega}^{(j_1)}_{\ell^\prime m^\prime})
    (_{s_2+\nu_2}\tilde{\omega}^{(j_2)}_{\ell^\prime m^\prime})^* \\
&\qquad\qquad \times \begin{pmatrix} \ell & \ell^{\prime} & \ell^{\prime\prime} \\
    -s_1 & s_1+\nu_1  & -\nu_1 \end{pmatrix} \begin{pmatrix}
    \ell & \ell^{\prime} & \ell^{\prime\prime} \\ -s_2 & s_2+\nu_2  & -\nu_2 \end{pmatrix}
\end{aligned}
```

# Arguments:
- `nu₁`: spin of the first map
- `nu₂`: spin of the second map
- `b₁::SpectralVector`: inverse noise-weighted beam multipoles for the first map
- `b₂::SpectralVector`: inverse noise-weighted beam multipoles for the second map
- `ρ₁`: polarization efficiency
- `ρ₂`: polarization efficiency
- `ω₁`: effective weights describing the scanning of the first map
- `ω₂`: effective weights describing the scanning of the second map


# Returns:
- `SpectralVector`: The cross-power spectrum of the two provided maps.

"""
function clquickpol(nu₁, nu₂, b₁::SV, b₂::SV, ρ₁, ρ₂, ω₁, ω₂) where {T, SV <: SpectralVector{T}}

end
