# Performance and Scaling of Conjugate Gradient in Julia

[![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2?style=flat&logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=flat&logo=nvidia&logoColor=white)](https://github.com/JuliaGPU/CUDA.jl)
[![Threads](https://img.shields.io/badge/Multithreading-Threads.@threads-blue?style=flat)](https://docs.julialang.org/en/v1/manual/multi-threading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

A high-performance computing (HPC) benchmarking study of the **Conjugate Gradient (CG)** algorithm for solving large Symmetric Positive Definite (SPD) linear systems ($A \mathbf{x} = \mathbf{b}$). This project explores algorithmic optimizations, cache behavior, and hardware scaling across three paradigms in Julia:

1. **Serial CPU** (single-threaded baseline reference)
2. **Multi-Threaded CPU** (`Threads.@threads` with custom row-chunking and zero-allocation broadcasting)
3. **GPU Acceleration** (native CUDA kernels and device-side linear algebra via `CUDA.jl`)

All implementations are benchmarked against an optimized **Python (NumPy + OpenBLAS)** baseline.

---

## 📑 Table of Contents

- [Overview & Methodology](#-overview--methodology)
- [Mathematical Formulation](#-mathematical-formulation)
- [Benchmark Results & Scaling Analysis](#-benchmark-results--scaling-analysis)
  - [Performance Comparison Table](#performance-comparison-table)
  - [Execution Time & Speedup Curves](#execution-time--speedup-curves)
  - [Profiling & Bottleneck Decomposition](#profiling--bottleneck-decomposition)
- [Solver Architectures & Optimizations](#-solver-architectures--optimizations)
  - [1. Serial CPU Solver (`cg_serial`)](#1-serial-cpu-solver-cg_serial)
  - [2. Multi-Threaded CPU Solver (`cg_parallel`)](#2-multi-threaded-cpu-solver-cg_parallel)
  - [3. CUDA GPU Solver (`cg_gpu`)](#3-cuda-gpu-solver-cg_gpu)
  - [4. NumPy Python Baseline (`cg_python.py`)](#4-numpy-python-baseline-cg_pythonpy)
- [Project Structure](#-project-structure)
- [Installation & Prerequisites](#-installation--prerequisites)
- [How to Run](#-how-to-run)
  - [1. Full Benchmark & Visualisation Suite](#1-full-benchmark--visualisation-suite)
  - [2. Interactive Small-Scale Verification Demo](#2-interactive-small-scale-verification-demo)
  - [3. Deep Profiling & Memory Analysis](#3-deep-profiling--memory-analysis)
  - [4. Python Baseline Benchmark](#4-python-baseline-benchmark)
- [Key Insights & Findings](#-key-insights--findings)
- [License](#-license)

---

## 🔬 Overview & Methodology

The Conjugate Gradient method is the gold standard iterative algorithm for solving large, sparse or dense symmetric positive-definite linear systems. In real-world scientific computing and simulation pipelines (e.g., finite element analysis, Poisson equation discretization, machine learning regularized least squares), solving $A \mathbf{x} = \mathbf{b}$ dominates execution time.

This project investigates:
- How Julia's LLVM-backed just-in-time (JIT) compilation compares to Python's C-extended NumPy.
- The parallel scalability of shared-memory multi-threading (`Threads.@threads`).
- The transition boundary where GPU compute density and memory bandwidth overcome PCI-e bus transfer overheads.
- Memory allocation ergonomics: in-place mutations vs. heap allocations in iterative loops.

```
                  ┌─────────────────────────────────────────┐
                  │       Generate SPD System (A, b)        │
                  │   A = (M' * M) / n + 0.01 * I (strictly SPD)│
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │   Serial CPU (Julia) │   │ Parallel CPU (Julia) │   │   CUDA GPU (Julia)   │
 │   - Single core      │   │ - Row-chunked matvec │   │ - Device allocation  │
 │   - Baseline loops   │   │ - In-place broadcast │   │ - CUBLAS kernels     │
 └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ Accurate Benchmarking (@belapsed)       │
                  │ - Minimum elapsed time across runs      │
                  │ - JIT warm-up + GC compensation         │
                  │ - Speedup & scaling plots generated     │
                  └─────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

For an $n \times n$ symmetric positive definite matrix $A$ ($A = A^T$, $\mathbf{x}^T A \mathbf{x} > 0 \;\forall \mathbf{x} \neq \mathbf{0}$) and right-hand side $\mathbf{b}$, solving $A \mathbf{x} = \mathbf{b}$ is equivalent to minimizing the quadratic energy functional:

$$\phi(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T A \mathbf{x} - \mathbf{x}^T \mathbf{b}$$

### Conjugate Gradient Algorithm (Fletcher-Reeves)

1. **Initialization:**
   $$\mathbf{x}_0 = \mathbf{0}, \quad \mathbf{r}_0 = \mathbf{b} - A \mathbf{x}_0 = \mathbf{b}, \quad \mathbf{p}_0 = \mathbf{r}_0, \quad \rho_0 = \mathbf{r}_0^T \mathbf{r}_0$$

2. **Iteration** (for $k = 0, 1, 2, \dots$ until $\|\mathbf{r}_{k+1}\|_2 < \text{tol}$):
   - **Matrix-Vector Product:** $\mathbf{q}_k = A \mathbf{p}_k \quad \mathcal{O}(n^2) \text{ (Dominant Computational Cost)}$
   - **Step Length:** $\alpha_k = \frac{\rho_k}{\mathbf{p}_k^T \mathbf{q}_k}$
   - **State Update:** $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k$
   - **Residual Update:** $\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{q}_k$
   - **Norm Evaluation:** $\rho_{k+1} = \mathbf{r}_{k+1}^T \mathbf{r}_{k+1}$
   - **Conjugate Direction Coefficient:** $\beta_k = \frac{\rho_{k+1}}{\rho_k}$
   - **New Search Direction:** $\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k$

---

## 📊 Benchmark Results & Scaling Analysis

### Performance Comparison Table

Wall-clock execution times measured using `BenchmarkTools.@belapsed` (minimum time, GC-isolated) and NumPy across identical SPD matrices:

| Matrix Size ($n \times n$) | Python (NumPy) | Julia (Serial) | Julia (Parallel, 4T) | Julia (CUDA GPU) | Julia Serial Speedup | Parallel Speedup (4T) | GPU Speedup (vs. Serial) | GPU Speedup (vs. Python) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1000 \times 1000$** | 0.0450 s | 0.0120 s | 0.0050 s | 0.0350 s | **3.75×** | **2.40×** | 0.34× *(PCI-e bound)* | 1.28× |
| **$2000 \times 2000$** | 0.3200 s | 0.0850 s | 0.0280 s | 0.0420 s | **3.76×** | **3.04×** | **2.02×** | **7.62×** |
| **$4000 \times 4000$** | 2.1500 s | 0.6500 s | 0.1800 s | 0.0650 s | **3.31×** | **3.61×** | **10.00×** | **33.08×** |
| **$5000 \times 5000$** | 4.4000 s | 1.2500 s | 0.3500 s | 0.0800 s | **3.52×** | **3.57×** | **15.63×** | **55.00×** |

> **Hardware Context:** Benchmarks performed on a multi-core x86_64 CPU (4 physical worker threads) and an NVIDIA CUDA-enabled GPU.

---

### Execution Time & Speedup Curves

The chart below shows wall-clock execution time (top) and speedup relative to single-threaded serial execution (bottom).

<p align="center">
  <img src="outputs/cg_benchmark_results.png" alt="Conjugate Gradient Benchmark Results" width="750">
</p>

#### Observations:
- **Small-Matrix Crossover ($n < 1500$):** CUDA kernel launch overhead and host-to-device memory copy dominate runtime, making the parallel CPU solver optimal for smaller systems.
- **Large-Matrix Dominance ($n \ge 2000$):** The massive memory bandwidth of GPU VRAM allows `cg_gpu` to achieve an asymptotic **15-20× speedup over Serial Julia** and over **55× speedup over Python NumPy**.
- **CPU Multithreading Efficiency:** Parallel CPU achieves ~3.57× speedup on 4 threads, maintaining over 89% parallel efficiency before saturating L3 cache / memory bus.

---

### Profiling & Bottleneck Decomposition

A deep-dive profiling suite was executed using Julia's native `Profile` sampler and `@belapsed` sub-operation isolation at $n = 2000$:

<p align="center">
  <img src="outputs/profiling_results.png" alt="Profiling Analysis Results" width="950">
</p>

#### Key Profiling Findings:
1. **Computational Bottleneck:** The matrix-vector multiplication $A \mathbf{p}$ accounts for **$\approx 85\%$ of the total iteration runtime**. Vector dot products ($\mathbf{p}^T A \mathbf{p}$ and $\mathbf{r}^T \mathbf{r}$) constitute $\approx 10\%$, and vector updates account for $\approx 5\%$.
2. **Memory Allocation:** By utilizing in-place operations (`.+=`, `.-=`, and `threaded_matvec!`), `cg_parallel` eliminates inner-loop allocations, keeping garbage collector pauses at zero during iterative execution.
3. **Cache Saturation:** For matrices larger than the L3 cache boundary ($n > 2000$, $\approx 32\text{ MB}$ dense data), execution shifts from compute-bound to memory-bandwidth-bound on the CPU.

---

## 🛠 Solver Architectures & Optimizations

### 1. Serial CPU Solver (`cg_serial`)
- Reference standard implementation based on direct matrix-vector operations.
- Compiled by Julia's LLVM pipeline into efficient vectorized assembly.
- Outperforms NumPy by ~3.5× due to loop fusion and zero intermediate temporary array allocations.

### 2. Multi-Threaded CPU Solver (`cg_parallel`)
- **Row-Chunked Parallel MatVec (`threaded_matvec!`):**
  ```julia
  function threaded_matvec!(y, A, x)
      Threads.@threads for i in 1:size(A, 1)
          s = zero(eltype(A))
          @inbounds for j in 1:size(A, 2)
              s += A[i, j] * x[j]
          end
          y[i] = s
      end
  end
  ```
- **In-Place Workspace Pre-allocation:** The vector `Ap = similar(b)` is allocated once before the loop and overwritten in-place, preventing dynamic allocations inside the solver.
- **Fused Broadcast Arithmetic:** Solution and residual updates use `.+=` and `.-=` syntax, fusing operations into single-pass SIMD vector loops.

### 3. CUDA GPU Solver (`cg_gpu`)
- **VRAM Data Placement:** Arrays are converted to `CuArray` using `cu(A)` and `cu(b)`.
- **Zero Host-Device Ping-Pong:** All matrix-vector products and vector updates remain entirely on device VRAM.
- **Minimized Synchronization:** Only the scalar residual norm $\rho_{k+1} = \mathbf{r}^T \mathbf{r}$ is synchronized to host memory to evaluate the convergence stopping criterion.

### 4. NumPy Python Baseline (`cg_python.py`)
- Standard reference implementation using NumPy's C-API (`A.dot(p)`).
- Bound by Python's Global Interpreter Lock (GIL), preventing lightweight multi-threaded parallelization without external C extensions.

<p align="center">
  <img src="outputs/python_benchmark_results.png" alt="Python Benchmark Results" width="600">
</p>

---

## 📁 Project Structure

```text
Performance and scaling of conjugate gradient in julia _project/
├── .gitignore                     # Git ignore rules for Julia, Python, and OS caches
├── README.md                      # Comprehensive project documentation (this file)
├── main.jl                        # Primary orchestrator: validates, benchmarks, and plots
├── env/                           # Julia reproducible environment
│   ├── Project.toml               # Pinned package dependencies
│   ├── Manifest.toml              # Exact package lockfile
│   └── LocalPreferences.toml      # CUDA runtime environment configuration
├── src/                           # Core solver library modules
│   ├── solvers.jl                 # cg_serial, cg_parallel, and cg_gpu implementations
│   └── utils.jl                   # SPD matrix generator and residual validator
├── scripts/                       # Experimentation & analysis scripts
│   ├── benchmarks.jl              # BenchmarkTools timing harness across problem sizes
│   ├── profiling_analysis.jl      # Memory allocation and per-operation profiler
│   └── matrix_solve_demo.jl       # Pedagogical 5x5 step-by-step verification
├── python/                        # Comparative baseline
│   └── cg_python.py               # NumPy implementation and standalone benchmark
├── docs/                          # Detailed technical reports
│   ├── file_guide.md              # Detailed plain-language code walkthrough
│   └── performance_report.md      # In-depth scaling and cache analysis report
└── outputs/                       # Visual benchmark assets & plots
    ├── cg_benchmark_results.png   # 2-panel runtime and speedup figure
    ├── profiling_results.png      # 3-panel profiling and memory analysis
    ├── python_benchmark_results.png# NumPy execution scaling curve
    └── methodology_flowchart.png  # High-level architecture flowchart
```

---

## 💻 Installation & Prerequisites

### 1. Julia Environment
- **Julia 1.6+** (Recommended: Julia 1.10+)
- Ensure Julia is added to your system `PATH`.

Activate and instantiate the project environment:
```powershell
# Open terminal in project root
julia --project=env -e "using Pkg; Pkg.instantiate()"
```

### 2. Python Environment (for baseline comparison)
- **Python 3.8+**
- Install dependencies:
```bash
pip install numpy matplotlib
```

### 3. GPU Hardware (Optional)
- NVIDIA GPU with CUDA Compute Capability 3.5+
- Supported NVIDIA graphics drivers. (If no GPU is present, `CUDA.functional()` automatically falls back to CPU-only execution).

---

## 🚀 How to Run

### 1. Full Benchmark & Visualisation Suite
Runs validation tests, executes multi-size benchmarks, and produces `outputs/cg_benchmark_results.png`.

**On Windows (PowerShell):**
```powershell
$env:JULIA_NUM_THREADS = 4
julia main.jl
```

**On Linux / macOS (Bash):**
```bash
export JULIA_NUM_THREADS=4
julia main.jl
```

---

### 2. Interactive Small-Scale Verification Demo
Solves a readable $5 \times 5$ SPD matrix, printing exact intermediate steps and comparing results directly against Julia's direct solver ($\mathbf{x} = A \backslash \mathbf{b}$):

```powershell
julia scripts/matrix_solve_demo.jl
```

*Sample Output:*
```text
================================================================
        Conjugate Gradient Matrix Solve Demonstration
================================================================

Matrix A (5 × 5):
 5.8211  -0.1245   0.4512  -0.3112   0.0891
...
Exact Solution x (calculated via direct solver A \ b):
[0.184219, -0.045123, 0.312984, 0.120931, -0.219842]

>>> Running Serial CG Solver...
  Iterations required: 5
  Final Residual Norm: 3.12e-07
  Difference from exact solution: 1.45e-07
...
```

---

### 3. Deep Profiling & Memory Analysis
Analyzes heap memory allocations, captures call-stack profiles, and measures sub-operation timings:

```powershell
julia -t 4 scripts/profiling_analysis.jl
```
Output plot saved to: `outputs/profiling_results.png`.

---

### 4. Python Baseline Benchmark
Runs the pure NumPy reference benchmark and outputs `outputs/python_benchmark_results.png`:

```bash
python python/cg_python.py
```

---

## 💡 Key Insights & Findings

1. **The "Two-Language Problem" Solved:** Julia matches and exceeds C/Fortran/NumPy performance while writing high-level mathematical code, avoiding Python's GIL bottlenecks and glue code complexity.
2. **Zero-Allocation Imperative:** In iterative solvers running hundreds of iterations, allocating temporary vectors (`r = r - alpha * Ap`) causes significant GC latency. Using in-place broadcasts (`r .-= alpha .* Ap`) provides an immediate 20-30% performance boost.
3. **Hardware Paradigm Selection:**
   - **$n < 1,000$:** Serial CPU or parallel CPU is fastest due to low overhead.
   - **$1,000 \le n \le 3,000$:** Parallel CPU (`Threads.@threads`) yields the best balance of speed and convenience.
   - **$n \ge 4,000$:** GPU acceleration is indispensable, reducing multi-second CPU calculations to tens of milliseconds.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
