# Biot Poromechanics — u-p Formulation

A parallel C++ solver for Biot's consolidation equations using a mixed displacement–pressure (u-p) finite element formulation, built on [deal.II](https://www.dealii.org/).

## What This Demonstrates

- **Multiphysics coupling** — monolithic assembly of the coupled mechanical–hydraulic system
- **Mixed finite elements** — Taylor–Hood pair Q(k+1)/Q(k) for displacement and pore pressure, satisfying the inf-sup stability condition
- **Large-scale MPI parallelism** — distributed triangulations via p4est, distributed linear algebra via Trilinos
- **Adaptive mesh refinement** — predictor-corrector local refinement to resolve pressure gradients and stress concentrations
- **deal.II** — modern C++ FEM library for research-grade scientific computing

## Physics

The code solves Biot's quasi-static consolidation model:

```
-div σ(u) + α ∇p = f                            (mechanical equilibrium)
c ∂p/∂t + α div(∂u/∂t) - div(K/μ ∇p) = g       (fluid mass balance)
```

where `u` is solid displacement, `p` is pore pressure, `α` is the Biot–Willis coefficient, `c` is specific storage, `K` is permeability, and `μ` is fluid viscosity. The monolithic u-p block system is solved with Newton's method at each time step.

## Dependencies

| Library | Role |
|---------|------|
| [deal.II](https://www.dealii.org/) ≥ 9.0 | FEM infrastructure, mesh, quadrature |
| MPI | Parallel execution |
| [Trilinos](https://trilinos.github.io/) | Distributed vectors and solvers |
| [p4est](https://p4est.github.io/) | Parallel adaptive mesh refinement |

Quick start with Docker:

```bash
docker pull tjhei/dealii:v9.0.1-full-v9.0.1-r5-gcc5
```

## Build

```bash
mkdir build && cd build
cmake -G Ninja -DDEAL_II_DIR=/path/to/dealii ..
ninja
```

Inside the Docker container:

```bash
docker run --rm -v $(pwd):/source -it tjhei/dealii:v9.0.1-full-v9.0.1-r5-gcc5
# inside the container:
mkdir build && cd build && cmake -G Ninja /source && ninja
```

## Run

```bash
# serial
./cracks parameter/parameters_sneddon_2d.prm

# parallel
mpirun -n 8 ./cracks parameter/parameters_sneddon_3d.prm
```

Running without arguments writes a `default.prm` listing all available options.

## Test Cases

- Sneddon pressurized crack (2D and 3D)
- Uniaxial tension
- Three-point bending
- Multiple interacting cracks in homogeneous and heterogeneous media

```bash
cd build && ctest -j4 -V
```

## Acknowledgements

Adapted from the crack phase-field solver by Thomas Wick and Timo Heister (2013–2020), licensed under GNU GPL v2+.
