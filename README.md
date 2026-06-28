# Data Encoding for Byzantine-Resilient Distributed Optimization

## Overview

This repository contains the Python/PyTorch implementation for the methods proposed in the paper ["Data Encoding for Byzantine-Resilient Distributed Optimization"](https://arxiv.org/pdf/1907.02664v2) by Deepesh Data, Linqi Song, and Suhas Diggavi.

The paper introduces a novel approach for enabling distributed optimization algorithms to remain resilient in the presence of Byzantine adversaries. These adversaries represent malicious or faulty worker nodes, which can arbitrarily deviate from pre-specified programs, thereby jeopardizing the integrity of the computation process. The method leverages **data encoding and error correction schemes over real numbers** to ensure robustness, without requiring probabilistic assumptions about the data or relying on randomized techniques.

The implementation provided in this repository demonstrates the use of the proposed encoding schemes for secure distributed optimization, focusing on **Proximal Gradient Descent (PGD)** and **Coordinate Descent (CD)** algorithms.

---

## Core Concepts

### Byzantine Fault Tolerance in Distributed Optimization
- **Distributed optimization** divides computation across multiple worker nodes to solve complex problems efficiently.
- **Byzantine adversaries** are worker nodes that behave maliciously or erroneously, potentially corrupting data or computation.
- The paper proposes a method to tolerate up to `t ≤ ⌊(m-1)/2⌋` corrupt nodes, where `m` is the total number of worker nodes. This threshold is **information-theoretically optimal**.

### Encoding Scheme
- The method uses **sparse data encoding** to ensure resilience against Byzantine attacks.
- Encoded data allows the master node to detect and correct errors introduced by corrupt worker nodes.
- Sparse encoding is computationally efficient, minimizing overhead in terms of storage, computation, and communication.

### Supported Algorithms
1. **Proximal Gradient Descent (PGD):**
   - Used in data-parallel settings where data is partitioned across different samples.
   - The method ensures reliable model training in the presence of adversarial attacks.
2. **Coordinate Descent (CD):**
   - Applied in model-parallel settings where data is partitioned across the parameter space.
   - The proposed scheme makes CD robust against adversarial attacks for the first time.
3. **Stochastic Gradient Descent (SGD):**
   - Extensions of the encoding scheme can be applied to scenarios involving streaming data and SGD.

### Key Contributions
- **Optimal Byzantine tolerance:** Up to `t ≤ ⌊(m-1)/2⌋` corrupt nodes can be handled.
- **Sparse encoding:** Reduces resource requirements (storage, computation, and communication) while maintaining robustness.
- **Experimental validation:** Demonstrates the trade-off between corruption tolerance and resource overhead.

---

## Repository Structure

```
├── README.md               # Project documentation
├── src/                    # Source code for the implementation
│   ├── encoding.py         # Sparse data encoding and decoding functions
│   ├── pgd.py              # Implementation of Byzantine-resilient Proximal Gradient Descent
│   ├── cd.py               # Implementation of Byzantine-resilient Coordinate Descent
│   ├── utils.py            # Utility functions for data preprocessing and simulation
│   └── experiment.py       # Script for running experiments and generating results
├── data/                   # Example datasets for testing
├── results/                # Output results from experiments
└── requirements.txt        # Python dependencies for the project
```

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch 1.9 or later

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/byzantine-resilient-optimization.git
   cd byzantine-resilient-optimization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Getting Started

### Running the Experiments
1. Prepare a dataset:
   - Place your dataset in the `data/` folder.
   - Ensure it is in a compatible format (e.g., CSV, NumPy array).

2. Run the experiment script:
   ```bash
   python src/experiment.py --algorithm pgd --data_path data/sample_dataset.csv --num_workers 10 --num_corrupt 3
   ```
   Replace `pgd` with `cd` to run experiments for the Coordinate Descent algorithm.

### Parameters
- `--algorithm`: Choose between `pgd` (Proximal Gradient Descent) or `cd` (Coordinate Descent).
- `--data_path`: Path to the input dataset file.
- `--num_workers`: Total number of worker nodes (`m`).
- `--num_corrupt`: Number of Byzantine (corrupt) nodes (`t`).

### Output
- Results are saved in the `results/` folder, including metrics such as runtime, error tolerance, and resource overhead.

---

## Key Features of the Code

1. **Encoding and Decoding (encoding.py):**
   - Implements the sparse encoding scheme for data distribution across worker nodes.
   - Decodes and corrects errors based on the received data from worker nodes.

2. **Proximal Gradient Descent (pgd.py):**
   - Simulates iterative updates in a data-parallel setting.
   - Incorporates the encoding scheme to ensure robustness against adversarial attacks.

3. **Coordinate Descent (cd.py):**
   - Simulates iterative updates in a model-parallel setting.
   - Extends the encoding scheme to secure the algorithm against Byzantine faults.

4. **Simulation Framework (experiment.py):**
   - Provides a testbed to evaluate the performance of the proposed methods.
   - Allows customization of parameters such as the number of workers, corruption threshold, and dataset.

---

## Experimental Results

The paper provides theoretical guarantees on the effectiveness of the proposed encoding schemes. The included experiments validate these findings by demonstrating:
- High resilience to Byzantine adversaries.
- Minimal overhead in terms of storage, computation, and communication.
- Stable performance across varying corruption thresholds.

Results from the experiments can be reproduced by running `experiment.py` with appropriate parameters.

---

## How to Contribute

We welcome contributions from the community. To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation of your changes.

---

## References

- [Data Encoding for Byzantine-Resilient Distributed Optimization](https://arxiv.org/pdf/1907.02664v2) by Deepesh Data, Linqi Song, and Suhas Diggavi.

For any questions or feedback, feel free to open an issue in the repository.