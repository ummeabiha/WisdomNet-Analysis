# WisdomNet: Implementation and Analysis

Implementation and research summary of **WisdomNet**, a method that teaches a neural network to *reject* inputs it is likely to misclassify instead of forcing a wrong prediction.

> **Paper:** T. X. Tran and R. S. Aygun, "WisdomNet: trustable machine learning toward error-free classification," *Neural Computing and Applications*, vol. 33, pp. 2719–2734, 2021. [doi:10.1007/s00521-020-05147-4](https://doi.org/10.1007/s00521-020-05147-4)

## Repository Contents

| File | Description |
|---|---|
| `WisdomNet_Implementation.ipynb` | WisdomNet reproduced on the Breast Cancer Wisconsin dataset |
| `WisdomNet Analysis Summary.pdf` | Summary of the paper's method, theory, results and limitations |


## How WisdomNet Works

1. **Pre-train** a base network *N* on the training set.
2. **Build the reject set** from the training samples *N* misclassifies.
3. **Add a conjugate (reject) neuron** to a copy of *N*, giving network *W*. Its weights and bias start at zero, so *W* initially behaves exactly like *N*.
4. **Fine-tune *W* on the reject set only**, labelling those samples as `reject`.

At inference, *W* outputs either a class or a rejection.


## Implementation

- **Dataset:** Breast Cancer Wisconsin (scikit-learn), 512 train / 57 test, stratified split
- **Base network:** 30–16–2 MLP (sigmoid, SGD), as in the paper
- **WisdomNet:** 30–16–3 (two classes + reject neuron), fine-tuned for 10 epochs

scikit-learn's binary `MLPClassifier` has a single output neuron, so it is mirrored into two equivalent class outputs (using `sigmoid(-z) = 1 - sigmoid(z)`) before the reject neuron is added. The notebook checks that *W* matches *N* before fine-tuning.

Metrics follow the paper: **Error = (FP + FN) / N** and **Reject rate = R / N**, where *N* is the full test set.

## Results

| Model | Error rate | Reject rate |
|---|---|---|
| Base network N | 1.75% | – |
| WisdomNet W | **0.00%** | 7.02% |

The paper reports 1.76% base error, 0% WisdomNet error and a 10.05% reject rate on this dataset.

### Fine-tuning sensitivity

The paper gives no stopping criterion for fine-tuning, so the notebook varies the number of epochs from the same starting weights:

| Epochs | 1 | 2 | 5 | **10** | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|---|---|
| Error (%) | 1.75 | 1.75 | 1.75 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| Reject (%) | 0.00 | 0.00 | 1.75 | **7.02** | 15.79 | 45.61 | 59.65 | 64.91 |

10 epochs is the earliest point with zero error and the lowest reject rate. Training longer only rejects more samples.


## Key Takeaways

- WisdomNet reaches 0% error on binary tasks while rejecting a small share of samples.
- The reject rate rises with lower base accuracy, more similar classes and noisier data.
- A single reject neuron scales poorly to multiclass problems.
- Fine-tuning length controls the error–rejection trade-off and must be tuned.

See the PDF for the full analysis.


## Citation

This work implements and analyses the following paper:
```bibtex
@article{tran2021wisdomnet,
  title   = {WisdomNet: trustable machine learning toward error-free classification},
  author  = {Tran, Truong X. and Aygun, Ramazan S.},
  journal = {Neural Computing and Applications},
  volume  = {33},
  number  = {7},
  pages   = {2719--2734},
  year    = {2021},
  publisher = {Springer},
  doi     = {10.1007/s00521-020-05147-4}
}
```
