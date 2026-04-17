# Content-Aware Image Retargeting: Forward Energy & Spectral Saliency

### Advanced Algorithmic Implementation of Seam Carving
This repository contains an optimized, high-performance implementation of Content-Aware Image Retargeting (Seam Carving). Moving beyond the foundational 2007 backward-energy algorithms, this engine integrates **Forward Energy transition costs**, **Frequency-Domain Spectral Saliency**, and **Numba JIT compilation** to achieve artifact-free, real-time matrix recalculations.

---

## 🧠 Algorithmic Architecture & Upgrades

This project was built to address the specific artifacting flaws present in early image-retargeting literature. It implements three major upgrades:

### 1. Forward Energy Matrix (Rubinstein et al., 2008)
Standard seam carving utilizes a backward-looking energy map, which often creates jagged edges by ignoring the new adjacencies formed after a pixel is deleted. This algorithm implements a **Forward-Looking Dynamic Programming (DP) table**. 
By calculating the transition costs ($C_U$, $C_L$, $C_R$) of future pixel merges, the algorithm actively avoids cuts that would place highly contrasting pixels next to each other, mathematically minimizing "scar tissue."

### 2. Spectral Residual Saliency Protection
Gradient-based edge detection (Sobel) is structurally blind to semantic meaning, often carving through smooth human faces while preserving textured background noise. 
This pipeline transforms the image into the frequency domain via a **Fast Fourier Transform (FFT)** to extract the Spectral Residual. This generates a spatial attention map that identifies the true foreground subjects. The Combined Energy equation is modeled as:
$$E_{total} = (\alpha \times E_{gradient}) + (\beta \times E_{saliency})$$
By heavily weighting $\beta$, the algorithm generates a mathematical "forcefield" that forces the DP traversal safely around human subjects.

### 3. Numba JIT Hardware Optimization
Calculating dual DP matrices for every single seam deletion requires massive polynomial time $O(W \times H)$. Standard Python interpreters bottleneck this execution. By integrating the `@jit(nopython=True)` compiler, the core traversal and matrix manipulation loops are translated directly into LLVM machine code, reducing computation time from minutes to milliseconds.

---

## 📊 The Academic Diagnostic Dashboard

Instead of treating the algorithm as a black box, the pipeline generates a 4-panel diagnostic Matplotlib dashboard for every processed image to visually prove the mathematical routing:
1. **Original Input:** The normalized baseline matrix.
2. **Combined Energy Map:** Visualizes the Sobel + Saliency forcefields.
3. **Cumulative Cost Matrix:** Exposes the Dynamic Programming table. Dark paths represent optimal sub-structures, while bright regions represent the massive mathematical penalties successfully protecting the subject.
4. **Retargeted Output:** The final matrix with the background condensed and the subject perfectly preserved.

---

## 📂 Repository Structure

```text
├── src/
│   ├── seam_carver.py         # Core mathematical implementation & DP logic
│   ├── dataset_prep.py        # Automated matrix standardization script
│   └── generate_dataset.py    # Automated A/B evaluation pipeline
│
├── Evaluation_Dataset/
│   ├── 1_Original_Inputs/          # Standardized 1000x800 matrices
│   ├── 2_Baseline_2007_Results/    # Intentional failures (Backward Energy)
│   └── 3_Optimized_Hybrid_Results/ # Successful outputs (Forward + Saliency)
│
├── Dashboard_Results/         # Auto-generated 4-panel academic dashboards
└── README.md
