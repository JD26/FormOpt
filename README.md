# FormOpt: A FEniCSx toolbox for level set-based shape optimization supporting parallel computing

`FormOpt` is a toolbox for two- and three-dimensional shape with parallel computing capabilities,
built on the `FEniCSx` software framework. 
The numerical shape modeling relies on a level set method, whose evolution is driven by a descent direction computed from the shape derivative.
Geometric constraints are treated accurately through a Proximal-Perturbed Lagrangian approach.
`FormOpt` leverages the powerful features of `FEniCSx`, particularly its support for weak formulations of partial differential equations, diverse finite element types, and scalable parallelism.
The implementation supports three different parallel computing modes: data parallelism, task parallelism, and a mixed mode. Data parallelism exploits `FEniCSx`'s mesh partitioning features, and we implement a task parallelism mode which is useful for problems governed by a set of partial differential equations with varying parameters. The mixed mode conveniently combines both strategies to achieve efficient utilization of computational resources.

[API reference](https://jd26.github.io/FormOpt/)

Toolbox paper: [published article](https://www.sciencedirect.com/science/article/pii/S0168874X26000879), [arXiv preprint](https://arxiv.org/abs/2601.05709) 

## Contact:
If you have any questions, suggestions, or need assistance with this project, feel free to reach out:
- **Josué:** josue.diazavalos@uni-due.de  
- **Antoine:** antoine.laurain@uni-due.de

## Repository structure

```
├── paper/				# Submitted manuscript
├── code/       		# Python scripts and numerical tests
│ 	├── formopt.py		# Classes and functions
│ 	├── models.py		# Model problems
│ 	├── test.py			# Several tests
│ 	├── load.py			# Load and show numerical results
│ 	├── plots.py		# Functions for plotting
│ 	├── Examples.ipynb	# Examples for manuscript
│	└── Tutorial.ipynb	# Examples with detailed explanations
├── arxiv/        		# LaTeX source files of the arxiv preprint
├── results/    		# Directory structure for numerical results
└── docs/       		# Markdown documentation files
```
## Installation

Currently, `FormOpt` runs under FEniCSx 0.9 (we are working in a version for FEniCSx 0.10).

1. Download and install [Anaconda](https://www.anaconda.com/) for Windows, Linux, or macOS.
2. Create a Anaconda environment and install some basic packages:
	```bash
	conda create -n fenicsx09
	```
	```bash
	conda activate fenicsx09
	```
	```bash
	conda install -c conda-forge python=3.11.10 numpy=1.26.3 scipy=1.12.0 matplotlib=3.8.3
	```
3. Install FEniCSx and its core dependencies:
	```bash
	conda install -c conda-forge fenics-dolfinx=0.9.0
	```
	```bash
	conda install mpich=4.3.0
	```
	```bash
	conda install pyvista=0.45.2
	```
4. Install additional packages (for mesh generation and working with HDF5 files):
	```bash
	conda install -c conda-forge pygmsh=7.1.17 h5py=3.10.0
	```
## Example: Compliance Minimization (L-Bracket)
This test was performed by **Max Boilledieu**
Email: **<max.boilledieu@stud.uni-due.de>**

![Optimal distribution](arxiv/L-Bracket.png)

## Example: Heat conduction

![Initial distribution](arxiv/heat1_0.png)

![Optimal distribution](arxiv/heat1_1.png)

## Example: Inverse elasticity

![Initial guess](arxiv/inv2_0.png)

![Recovered inclusions](arxiv/inv2_1.png)

