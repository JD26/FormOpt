# Level class

The `Level` class solves the diffusion-transport equation:

$$
\begin{aligned}
\partial_t \phi+\theta\cdot\nabla\phi &= h^2\Delta\phi && x\in \mathcal{D},\, t>0, \\
\partial_{n} \phi &= 0 && x\in \partial\mathcal{D},\, t>0, \\
\phi(0, x) &= \phi^{i}(x) && x\in \mathcal{D},
\end{aligned}
$$

where $\theta$ is the velocity field, $h$ the mesh diameter,
and $\phi^i$ the level set function at iteration $i$.
A time step $\Delta t$ and a final time $T$ must be provided.
We choose the next level set function as $\phi^{i+1}(x) = \phi(T,x)$.

---

## Details

::: code.formopt.Level



