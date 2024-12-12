# GPU-Accelerated Multi-Improvement Number Partitioning

## Overview
This project implements a parallel approach to the Number Partitioning Problem (NPP) using CUDA. It compares three different algorithms:
1. Classical Karmarkar-Karp (KK) algorithm
2. KK with GRASP and Simulated Annealing
3. CUDA-based Multi-Improvement approach

## Algorithm Description
The implementation combines multiple optimization techniques:
- **Karmarkar-Karp Base**: Uses the differencing method to generate initial solutions
- **GRASP Component**: Implements a greedy randomized adaptive search procedure
- **CUDA Parallelization**: Leverages GPU for parallel solution evaluation and improvement
- **Multi-Improvement**: Applies multiple independent moves simultaneously

## Key Features
- Parallel processing of multiple solutions using CUDA
- Support for large-scale number partitioning instances
- Efficient memory management for GPU computation
- Implementation of various neighborhood exploration strategies

## Technical Implementation
The main components include:
```cpp
// Solution representation and manipulation
struct Element {
    uint64_t value;
    std::vector<std::pair<uint64_t, int>> components;
};

// Multi-Improvement kernel functions
__global__ void calc_sums_kernel()      // Calculates subset sums
__global__ void evaluate_moves_kernel()  // Evaluates potential improvements
__global__ void apply_moves_kernel()     // Applies selected improvements
