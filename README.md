# SPPU LP-5 Coursework Repository

Welcome to the High Performance Computing (HPC) and Deep Learning (DL) laboratory coursework repository for the Savitribai Phule Pune University (SPPU) Final Year B.E. curriculum.

This repository contains clean, standardized, and highly optimized implementations of parallel algorithms using **OpenMP** and **CUDA C++**, alongside  **Deep Learning** models developed in **Jupyter Notebooks** using **TensorFlow/Keras**.

---

## 📁 Repository Structure

```text
lp-5/
│
├── hpc/                                # High Performance Computing Practicals
│   ├── 1a-dfs.cpp                      # Parallel Depth First Search (OpenMP)
│   ├── 1b-bfs.cpp                      # Parallel Breadth First Search (OpenMP)
│   ├── 2a-merge-sort.cpp               # Parallel Merge Sort (OpenMP)
│   ├── 2b-bubble-sort.cpp              # Parallel Bubble Sort (Odd-Even Transposition)
│   ├── 3-min-max-sum-avg.cpp           # Parallel Reductions (Min, Max, Sum, Avg)
│   └── 4-cuda.cu                       # GPU-Accelerated Vector Addition (CUDA)
│
└── dl/                                 # Deep Learning Assignments
    ├── dl_assign_1.ipynb               # Boston Housing Price Prediction (DNN Regression)
    ├── dl_assign_2.ipynb               # IMDB Movie Review Sentiment Analysis (CNN-LSTM)
    └── dl_assign_3.ipynb               # Fashion MNIST Image Classification (Conv2D)
```

---

## 🚀 Part I: High Performance Computing (HPC)

### System Requirements & Environment Setup
To compile and execute the HPC C++ and CUDA programs, ensure you have the following installed:
- **GCC / G++ Compiler** with OpenMP support (`-fopenmp` flag).
- **NVIDIA CUDA Toolkit** (`nvcc` compiler) for GPU-accelerated practicals.

---

### 1. Parallel Depth First Search (DFS)
* **File:** `hpc/1a-dfs.cpp`
* **Description:** Implements graph traversal using Depth First Search parallelized via OpenMP threads.
* **Compilation:**
  ```bash
  g++ -fopenmp 1a-dfs.cpp -o 1a-dfs
  ```
* **Execution:**
  ```bash
  ./1a-dfs         # On Linux/macOS
  .\1a-dfs.exe     # On Windows
  ```
* **Ideal Input Example:**
  ```text
  Enter Number of vertices :- 6
  Enter Number of Edges: - 5
  Enter Edges (u-v)
  0 1
  0 2
  1 3
  1 4
  2 5
  Enter Starting Node 0
  ```

---

### 2. Parallel Breadth First Search (BFS)
* **File:** `hpc/1b-bfs.cpp`
* **Description:** Implements level-by-level graph traversal using a standard queue parallelized with OpenMP `parallel for` and `critical` sections.
* **Compilation:**
  ```bash
  g++ -fopenmp 1b-bfs.cpp -o 1b-bfs
  ```
* **Execution:**
  ```bash
  ./1b-bfs
  ```
* **Ideal Input Example:**
  ```text
  Enter Number of Vertices:- 6
  Enter Number of Edges : - 5
  Enter Edges (u v )
  0 1
  0 2
  1 3
  1 4
  2 5
  Enter Start Node: - 0
  ```

---

### 3. Parallel Merge Sort
* **File:** `hpc/2a-merge-sort.cpp`
* **Description:** Utilizes OpenMP `#pragma omp parallel sections` to divide and sort sub-arrays recursively across multiple threads. Includes precise time measurement using `std::chrono`.
* **Compilation:**
  ```bash
  g++ -fopenmp 2a-merge-sort.cpp -o 2a-merge-sort
  ```
* **Execution:**
  ```bash
  ./2a-merge-sort
  ```
* **Ideal Input Example:**
  ```text
  Enter the size of the array: 8
  Enter 8 elements:
  38 27 43 3 9 82 10 19
  ```

---

### 4. Parallel Bubble Sort (Odd-Even Transposition Sort)
* **File:** `hpc/2b-bubble-sort.cpp`
* **Description:** Implements parallel bubble sort by alternating between odd and even indexed comparison phases using OpenMP threads. Outputs sorted elements and elapsed execution time.
* **Compilation:**
  ```bash
  g++ -fopenmp 2b-bubble-sort.cpp -o 2b-bubble-sort
  ```
* **Execution:**
  ```bash
  ./2b-bubble-sort
  ```
* **Ideal Input Example:**
  ```text
  Enter the size of the array: 6
  Enter 6 elements:
  64 34 25 12 22 11
  ```

---

### 5. Parallel Reduction Operations (Min, Max, Sum, Average)
* **File:** `hpc/3-min-max-sum-avg.cpp`
* **Description:** Demonstrates the power of OpenMP parallel reduction clauses (`reduction(min:minValue)`, `reduction(max:maxValue)`, `reduction(+:sum)`) to aggregate statistics over arrays with maximal performance.
* **Compilation:**
  ```bash
  g++ -fopenmp 3-min-max-sum-avg.cpp -o 3-min-max-sum-avg
  ```
* **Execution:**
  ```bash
  ./3-min-max-sum-avg
  ```
* **Ideal Input Example:**
  ```text
  Enter number of elements: 10
  Enter elements:
  15 22 84 9 43 5 99 31 70 52
  ```

---

### 6. CUDA Vector Addition
* **File:** `hpc/4-cuda.cu`
* **Description:** Implements massively parallel element-wise addition of two vectors using an NVIDIA GPU kernel. Automates thread and block layout calculations, compares results against CPU execution for correctness, and tracks execution duration using native CUDA events.
* **Compilation:**
  ```bash
  nvcc 4-cuda.cu -o 4-cuda
  ```
* **Execution:**
  ```bash
  ./4-cuda
  ```
* **Ideal Input:** 
  *This program requires no interactive input.* It automatically allocates and populates host/device arrays of size $2^4 = 16$ elements with random integers, performs the GPU kernel launch, and directly displays the output comparison and timing metrics.

---

## 🧠 Part II: Deep Learning (DL)

The Deep Learning section focuses on building foundational to advanced architectures to solve real-world tasks ranging from numerical regression to computer vision and natural language processing.

### Environment Setup
We recommend using an environment like **Anaconda**, **Miniconda**, or **Google Colab** with the following packages installed:
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

---

### Assignment 1: Boston Housing Price Prediction
* **File:** `dl/dl_assign_1.ipynb`
* **Task:** Deep Neural Network (DNN) Regression.
* **Details:** Constructs a fully connected regression network to estimate median house values. Demonstrates input feature standardization, deep Dense layers, custom learning rate configurations, model training history plotting, and test set MSE/MAE evaluation.

### Assignment 2: IMDB Movie Review Sentiment Classification
* **File:** `dl/dl_assign_2.ipynb`
* **Task:** Natural Language Processing (NLP) / Binary Classification.
* **Details:** Classifies text reviews into positive or negative sentiments using a sophisticated hybrid architecture consisting of:
  1. **Word Embeddings** to generate dense vector representations.
  2. **1D Convolutional Layers (Conv1D)** for local feature extraction.
  3. **Long Short-Term Memory Layers (LSTM)** to capture sequential context.
  4. **Dense Layers with Dropout** to perform classification while avoiding overfitting.

### Assignment 3: Fashion MNIST Image Classification
* **File:** `dl/dl_assign_3.ipynb`
* **Task:** Computer Vision / Multi-Class Classification.
* **Details:** Identifies 10 categories of clothing items from 28x28 grayscale images. Implements a highly robust deep Convolutional Neural Network containing stacked **Conv2D** blocks, **Batch Normalization** for faster convergence, **MaxPooling2D** layers, and **Dropout** regularizers. Includes learning rate plateau adjustments and early stopping strategies.

---

## 📌 Usage Notes
- Make sure to activate your OpenMP flags when compiling the C++ files to take advantage of multi-core CPU parallelization.
- For the CUDA assignment, an NVIDIA GPU with appropriate drivers installed is required. If running locally without a GPU, you can leverage Google Colab by selecting the **T4 GPU** runtime environment.
