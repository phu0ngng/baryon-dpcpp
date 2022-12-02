# Porting the Baryon-block Construction in the Stochastic LapH Method to Heterogeneous Systems with DPC++

This repository contains the implementation for my Master's thesis in Computational Science and Engineering at the Technical University Munich under the supervision of Dr. Ben Hörz  (Intel) and  Prof.  Michael Bader.

## Abstract
Lattice Quantum Chromodynamics (LQCD) is an important workload in particle physics that provides predictions from simulations of the strong interaction in the Standard Model of particle physics. In order to match the precision of experiments and achieve simulation breakthroughs, there is the need to achieve sustained performance in the range of multiple Exaflops. Thus, optimization of all dominant steps in the LQCD workload is needed. This work provides a high-performance implementation of the baryon-block construction, one of the key kernels in the stochastic LapH method, targeting heterogeneous systems.

The main contribution of this work consists of two parts: First, the optimization of the current implementations of CPUs; Second, porting the kernel to Intel GPUs using Data Parallel C++ (DPC++).  
  
For the optimization of the CPU implementation, new memory-data layouts are investigated, cache blocking is successfully employed, and highly optimized small matrix multiplication is used by utilizing the Intel® Math Kernel Library with Just-In-Time code generation. The GPU kernel is implemented using DPC++ based on the optimized CPU implementation. The GPU implementation is optimized using data prefetching and data pre-packing techniques.  
  
The main result of this thesis is an optimized multi-threaded implementation of the kernel which is 6.8 times faster than the kernel currently used in production systems. In addition, the work shows the successful port of the kernel to the Ponte Vecchio GPU, which has a speedup of X times over the node performance of the optimized CPU implementation (NDA).  
  
The work provides a solid foundation for modern performance-portable implementations suitable for usage in future heterogeneous HPC systems.

## Baryon-block construction in the stochastic LapH method
Baryons (such as the proton and neutron) are hadrons composed of three quarks. The following intermediary equation, which has been adapted from Eq. (23) of [2], is of great importance to construct a baryon block:

$$
\mathcal{B}^{d_1 d_2 d_3}_{\vec p} = \sum_{\vec x}^{L^3} \mathrm{e}^{-\mathrm{i} \vec p \vec x} \, \sum_{a,b,c} \varepsilon_{abc} \, q^{d_1}_{\vec x a} q'^{d_2}_{\vec x b} q''^{d_3}_{\vec x c}.
$$

 - The $q$, $q'$, and $q''$ represent the complex-based quark fields with an eigendecomposition. 
 - The $a,b,$ and $c$ denote colour indices while $d_1, d_2,$ and $d_3$ are dilution indices of $q$, $q'$, and $q''$, respectively.
 - The $\varepsilon_{abc}$ is the Levi-Civita symbol, which selects the six nonzero contributions to the colour sum according to   

$$
            \varepsilon_{abc} = \begin{cases}
                +1 \quad (abc) \in \{(123), (231),(312)\}, \\
                -1 \quad (abc) \in \{(321), (132),(213)\}, \\
                \phantom{+}0 \quad \text{otherwise}.
            \end{cases}
$$

- The $\vec p$ denotes (a set of) three-dimensional momenta, $\vec p = 2\pi / L \vec d$ with an integer vector $\vec d \in \mathbb{Z}^3$.                                                                                               The number of requested momenta is typically much smaller than the number of allowed momenta (e.g.~$33 \ll 64^3$), so that using a fast Fourier transform is not beneficial.
- The $e^{-\mathrm{i} \vec p \vec x}$ is phase factor and can be computed as $e^{-\mathrm{i} \vec p \vec x} = \cos(\vec p \vec x) - \mathrm{i} \sin(\vec p \vec x)$, for a given momentum $\vec p$ and spatial index $\vec x$.

The quark fields $q, q', q''$ are decomposed in a basis spanned by the $N_\mathrm{ev}$ low-lying eigenvectors of the Laplacian operator in the stochastic LapH method, as

$$
    q_{\vec x a}^d = \sum_{l=1}^{N_\mathrm{ev}} Q_{dl} \phi_{\vec x a}^l.
$$

- The $q_{\vec x a}^{d}$ presents the quark field with the dilution index $d=1, \dots, N_D$, the colour index $a=1,2,3$, and the spatial index $\vec x$. The dilution range $N_D$ typically has a value of 64.
- The spatial index $\vec x$ is the 'flattened' one-dimensional index in the lexicographic order of the three-dimensional lattice grids with the range of $L^3$, where $L$ is the one-dimensional lattice size. A typical lattice size nowadays is $64^3$, but there is a pressing need to scale up to $96^3$.
-  The $\phi$ is the set of $N_\mathrm{ev}$ low-lying eigenvectors of the three-dimensional Laplace operator on a given time slice which depends on the gluon field and hence has to be computed individually for each time slice of a gauge configuration. Due to the underlying physics, $N_\mathrm{ev}$ needs to be scaled proportionally to the spatial volume and is hence dependent on the problem size.
- The coefficients Q describe the quark fields as a vector in the space spanned by the Laplacian eigenvectors. In the stochastic LapH workflow, those quark fields are obtained from solutions of the Dirac equation.

## Repository Structure
```
baryon-dpcpp/
|-- src
|   |-- Makefile
|   |-- kernel_cpu_opt.cc
|   |-- kernel_cpu_ref.cc
|   |-- kernel_dpcpp.cc
|   |-- kernels_qdp3.h
|   |-- lattice.cc
|   |-- lattice.h
|   |-- main.cc
|   |-- misc.cc
|   |-- misc.h
|   |-- qdp_multi.h
|   `-- swatch.h
|-- testscripts
|   |-- scanning_blocksizes.sh
|   |-- sycl_cuda_env.sh
|   |-- sycl_hip_env.sh
|   |-- test.sh
|   `-- test_various_dx.sh
|-- Makefile
`-- README.md
```
- `kernel_cpu_ref.cpp` contains the reference implementation for CPUs as used in *Chroma_laph*.
- `kernel_cpu_opt.cpp` contains the optimized implementation for CPUs using C++.
- `kernel_dpcpp.cpp` contains the optimized implementation with DPC++.

## Compilation & Run
### Prerequisite
- Intel® oneAPI DPC++/C++ Compiler
- Intel® oneAPI Math Kernel Library (oneMKL)

### How to compile
todo

### How to run
todo

### 

## Reference:
- [1] P. Nguyen and B. Hörz. *"Performance Optimization of Baryon-block Construction in the Stochastic LapH Method"*. 2022. doi: 10.48550/ARXIV.2211.16278. [https://arxiv.org/abs/2211.16278](https://arxiv.org/abs/2211.16278).
- [2] C. Morningstar, J. Bulava, J. Foley, K. J. Juge, D. Lenkner, M. Peardon, and C. H. Wong. *“Improved stochastic estimation of quark propagation with Laplacian Heaviside smearing in lattice QCD”*. In: Phys. Rev. D 83 (11 June 2011), p. 114505. doi: 10.1103/ PhysRevD.83.114505. url: https://link.aps.org/doi/10.1103/PhysRevD.83.114505.



