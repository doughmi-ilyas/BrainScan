# Model card: BrainScan AI

## Summary
Convolutional neural network (TensorFlow/Keras) that classifies a 2D brain MRI image into
four classes: `glioma`, `meningioma`, `notumor`, `pituitary`. Educational project, **not a
medical device**.

## Model
- Input: RGB image resized to 224x224, values scaled to 0-1.
- Architecture: 3 x (Conv2D + MaxPooling) with 32, 64, 128 filters, Flatten, Dense 128,
  Dropout 0.5, Dense 4 (softmax). See `src/step2/build_model.py`.
- Training: Adam, batch size 16, up to 15 epochs, `EarlyStopping` (patience 3 on `val_loss`),
  best model kept by `val_accuracy`. See `src/step2/train_model.py`.

## Data
- 7,023 images in four class folders (not stored in this repository).
- Stratified split 70 % train / 15 % validation / 15 % test (test = 1,054 images).

## Results (test set, 1,054 images)
Confusion matrix (rows = true class, columns = predicted), read from
`assets/confusion_matrix.png`:

| true \ predicted | glioma | meningioma | notumor | pituitary | recall |
|---|---:|---:|---:|---:|---:|
| glioma | 233 | 8 | 0 | 2 | 95.9 % |
| meningioma | 20 | 212 | 7 | 8 | 85.8 % |
| notumor | 0 | 6 | 293 | 1 | 97.7 % |
| pituitary | 0 | 3 | 0 | 261 | 98.9 % |
| precision | 92.1 % | 92.6 % | 97.7 % | 96.0 % | |

Accuracy: (233 + 212 + 293 + 261) / 1,054 = 999 / 1,054 = **94.8 %**. The per-class numbers
above are recomputed from the matrix and match the README.

## Known limits
- Single train/validation/test split, one training run: no confidence interval, no
  cross-validation, no comparison of seeds.
- The dataset source and licence are not documented in this repository, and it is not
  known whether near-duplicate images from the same patient fall in different splits
  (possible leakage that would make the score optimistic).
- Meningioma is the weakest class (20 of 247 predicted as glioma).
- Not evaluated on images from another hospital, scanner or dataset.
- No calibration check: the confidence shown in the app is the softmax output, not a
  probability of being right.

## Intended use
Learning and demonstration of an image-classification pipeline. Not for diagnosis or any
clinical decision.
