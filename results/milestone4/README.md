# Milestone 4 Results

## RQ1 — Held-out comparison

All methods used settings selected through three-fold cross-validation, then were fitted on the 880 development images and evaluated on the same 160 held-out test images.

| Method | Selected setting | Mean CV macro-F1 | Test macro-F1 | Test accuracy |
|---|---|---:|---:|---:|
| Two-stage ResNet18 cascade | Learning rate `0.0001` | 65.43% | **69.61%** | **81.25%** |
| Logistic regression | L2 penalty `0.1` | 63.18% | 65.14% | 76.88% |
| k-NN | `k=5` | 58.66% | 65.06% | 77.50% |

The cascade scored 4.47 percentage points higher in test macro-F1 than logistic regression, the strongest baseline by that metric. The cascade raised 2 false alarms among 92 no-fire images and missed 12 of 68 fires.

- [Cascade test confusion matrix](cascade-test/confusion_matrix.csv)
- [Cascade test metrics](cascade-test/metrics.json)

## RQ2 — Cascade learning rate

| Learning rate | Fold 1 macro-F1 | Fold 2 macro-F1 | Fold 3 macro-F1 | Mean macro-F1 | Sample SD |
|---|---:|---:|---:|---:|---:|
| `0.0001` | 68.30% | 58.68% | 69.31% | **65.43%** | 5.87 pp |
| `0.00001` | 57.62% | 56.09% | 52.72% | 55.48% | 2.51 pp |

With five training epochs, `0.0001` scored higher on all three folds and averaged 9.95 percentage points higher macro-F1.

[Reproduction commands](../../README.md#milestone-4-evaluation)