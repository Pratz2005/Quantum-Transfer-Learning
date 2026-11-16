# Quantum Transfer Learning for Pneumonia Classification

Does a **4-qubit variational quantum circuit** make a better classification head than a classical one when both sit on top of the same **frozen, ImageNet-pretrained ResNet-18**?

This repository compares classical and hybrid quantum-classical heads on [PneumoniaMNIST](https://medmnist.com/) chest X-rays (Normal vs. Pneumonia). All of the code is in one notebook, [`main.ipynb`](main.ipynb).

**Short answer:** no. The quantum head trains and reaches an AUC of about 0.91–0.94, but it never beats a classical head of the same size on accuracy, stability or convergence speed.

---

## Pipeline

```
28×28 X-ray ──► grayscale→RGB, resize 224×224, ImageNet norm
            ──► frozen ResNet-18 (fc removed) ──► 512-d feature  (cached once in raw_features.pt)
            ──► standardise with train-set mean/std
            ──► classification head  (one of the models below)
```

| Model | Architecture | Trainable params |
|---|---|---|
| **Linear head** | `Linear(512→2)` | 1,026 |
| **Classical bottleneck** | `Linear(512→4) → tanh → Linear(4→4) → tanh → Linear(4→2)` | 2,082 |
| **Hybrid quantum (L=2)** | `Linear(512→4) → tanh·π/2 → 4-qubit VQC → Linear(4→2)` | 2,086 |
| **Hybrid no-CNOT (L=2)** | Same as the hybrid, with the entangling gates removed (ablation) | 2,086 |

The classical bottleneck is a **parameter-matched control**. It uses the same 512→4 compression as the hybrid model, with the quantum circuit replaced by a classical 4→4 layer. Any gap between the two therefore comes from the circuit, not from the compression.

### Variational quantum circuit

Built with PennyLane (`default.qubit`, exact state-vector simulation, backprop gradients) and wrapped as a `TorchLayer`, so the whole model trains end-to-end in PyTorch.

```
AngleEmbedding (RY) on 4 qubits
  └─ repeat L times:  Rot(φ,θ,ω) on every qubit  →  ring of CNOTs (0→1→2→3→0)
⟨Z⟩ on every qubit  →  4 outputs
```

The circuit has `L × 4 × 3` trainable weights. The depth `L` and the `entangle` switch are constructor arguments.

### Training & evaluation protocol

- The official MedMNIST **train / val / test** splits are used (4,708 / 524 / 624 images).
- The same training loop is used for every model: Adam, lr = 5e-3, 15 epochs, batch size 32, **class-weighted cross-entropy** (the training set is about 74% pneumonia).
- The **best epoch is chosen by validation loss only**. The test set is used once per trained model, for reporting.
- Metrics: accuracy, ROC-AUC, macro-F1, **sensitivity** (pneumonia recall) and **specificity**.
- Diagnostics: the mean gradient norm per epoch, and the gradient norm of the **quantum weights only**, which is used to look for barren plateaus.

---

## Results

### Experiment 1: head comparison (test set, mean ± std over 5 seeds)

| Model | Accuracy | AUC | Macro-F1 | Sensitivity | Specificity | Epochs to val AUC ≥ 0.97 |
|---|---|---|---|---|---|---|
| Linear head | 0.875 ± 0.007 | 0.936 ± 0.010 | 0.860 ± 0.009 | 0.956 ± 0.007 | 0.739 ± 0.027 | 1.2 (5/5 reached) |
| Classical bottleneck | 0.871 ± 0.009 | **0.943 ± 0.006** | 0.857 ± 0.011 | 0.950 ± 0.007 | 0.740 ± 0.025 | 1.0 (5/5) |
| Hybrid quantum (L=2) | 0.851 ± 0.021 | 0.912 ± 0.025 | 0.831 ± 0.025 | 0.958 ± 0.015 | 0.673 ± 0.038 | 5.0 (3/5) |
| Hybrid no-CNOT (L=2) | 0.870 ± 0.012 | 0.934 ± 0.010 | 0.854 ± 0.015 | 0.956 ± 0.010 | 0.727 ± 0.032 | 1.6 (5/5) |

Hybrid vs. classical bottleneck, test AUC: Welch t-test p = 0.044. With only 5 seeds, treat this as indicative.

### Experiment 2: circuit depth (test set, mean over 3 seeds)

| Depth L | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| Test AUC | 0.897 | 0.917 | 0.932 | **0.936** | 0.926 |
| Quantum-weight grad norm | 0.092 | 0.106 | 0.127 | 0.130 | 0.144 |
| Epochs to val AUC ≥ 0.97 | 13.0 | 4.5 | 2.7 | 3.7 | 2.7 |

### Key findings

1. **The classical heads win.** The parameter-matched classical bottleneck has the highest AUC (0.943). The L=2 hybrid has the lowest (0.912) and is about 4× less stable across seeds.
2. **The quantum head converges more slowly.** The classical heads reach a validation AUC of 0.97 in 1–2 epochs. The L=2 hybrid needs about 5 epochs and did not reach it at all in 2 of 5 seeds.
3. **Entanglement did not help.** Removing the CNOTs *improved* the hybrid model (AUC 0.934), bringing it level with the classical heads.
4. **Depth helps up to about 3–4 layers**, then plateaus. Deeper circuits narrow the gap to the classical heads but never close it.
5. **No barren plateau was observed.** The quantum-weight gradient norm *grows* slightly with depth. This is expected at 4 qubits and says nothing about larger circuits.
6. All models have high sensitivity (about 0.95) and lower specificity (0.67–0.74), a result of the class imbalance. Most of the remaining error is false positives.

**Conclusion:** this is an honest neutral-to-negative result. A small variational quantum head on frozen CNN features can be trained, but it gives no advantage over an equally sized classical head.

---

## Reproducing the results

### Requirements

- Python 3.11 (tested)
- `pennylane`, `torch`, `torchvision`, `medmnist`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `tqdm`

```bash
python -m venv qml
source qml/bin/activate          # Windows: qml\Scripts\activate
pip install pennylane torch torchvision medmnist scikit-learn scipy pandas matplotlib tqdm jupyter
```

### Run

```bash
mkdir -p figures                 # the plotting cells save into this folder
jupyter notebook main.ipynb
```

Run the cells from top to bottom. The notebook will:

1. Download PneumoniaMNIST automatically through `medmnist`.
2. Extract ResNet-18 features once and cache them in `raw_features.pt`. This is the only slow step on a CPU.
3. Train and evaluate the single-run baselines and the hybrid model (seed 42).
4. Run **Experiment 1** (4 models × 5 seeds) and **Experiment 2** (5 depths × 3 seeds), printing tables and saving `figures/auc_boxplot.png` and `figures/depth_plots.png`.

The runtime of the experiments is controlled by one configuration cell (`EPOCHS`, `SEEDS`, `DEPTHS`, `DEPTH_SEEDS`, `AUC_TARGET`). Reduce these values for a quicker run. All randomness is seeded with `set_seed()`. Exact numbers can still vary slightly across hardware and library versions.

### Notebook layout

| Section | Contents |
|---|---|
| 1. Data & feature extraction | Loads the dataset, shows example images, runs the frozen ResNet-18, standardises features, computes class weights |
| 2. Training routine | The shared `train_and_evaluate` loop (validation-loss checkpointing, gradient-norm logging) |
| 3. Classical baselines | `LinearHead`, `BottleneckClassicalClassifier` |
| 4. Hybrid quantum classifier | `make_qnode`, `HybridQuantumClassifier`, circuit diagram |
| 5. Test-set evaluation | Classification reports, confusion matrices, ROC curves, training curves |
| 6. Experiments | Multi-seed comparison with the no-CNOT ablation, and the circuit-depth sweep |
| 7. Discussion & limitations | Interpretation of the results |

---

## Limitations

- **Simulation only.** The circuit runs on a noiseless simulator with backprop gradients. Real hardware would need parameter-shift gradients and would add shot noise and gate noise.
- **The frozen CNN does most of the work.** ResNet-18 features are already highly separable (a linear head reaches about 0.94 test AUC), which leaves little room for any head to improve.
- **Tiny circuit.** 4 qubits after a 512→4 compression is easy to simulate classically, so no quantum advantage can be claimed.
- **Low-resolution data.** The 28×28 images are upscaled to 224×224. These results say nothing about clinical usefulness.
- **Statistics.** 3–5 seeds and a 624-image test set give only rough error bars. No correction for multiple comparisons was applied.
- **No per-model hyperparameter tuning.** The quantum head might benefit from a different learning rate or a longer training schedule.

---

## Acknowledgements

- [MedMNIST v2](https://medmnist.com/): Yang et al., *Scientific Data*, 2023.
- [PennyLane](https://pennylane.ai/) for differentiable quantum circuits.
- Mari et al., *Transfer learning in hybrid classical-quantum neural networks*, Quantum 4, 340 (2020), which inspired the hybrid-head design.
