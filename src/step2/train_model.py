from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.step1.prepare_data import make_tf_dataset, split_image_paths
from src.step2.build_model import build_model


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "brain_scan.keras"
HISTORY_PATH = MODEL_DIR / "history.json"
PLOT_PATH = MODEL_DIR / "training_curves.png"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.png"


def save_training_curves(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    tf.config.optimizer.set_jit(False)
    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        encoder,
    ) = split_image_paths()

    train_dataset = make_tf_dataset(
        train_paths,
        train_labels,
        batch_size=16,
        shuffle=True,
    )
    val_dataset = make_tf_dataset(val_paths, val_labels, batch_size=16)
    test_dataset = make_tf_dataset(test_paths, test_labels, batch_size=16)

    model = build_model()
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=15,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    duration = time.time() - start_time

    model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    predictions = np.argmax(model.predict(test_dataset, verbose=0), axis=1)

    print(f"Training time: {duration / 60:.2f} minutes")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(classification_report(test_labels, predictions, target_names=encoder.classes_))

    matrix = confusion_matrix(test_labels, predictions)
    display = ConfusionMatrixDisplay(matrix, display_labels=encoder.classes_)
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH)
    plt.close()

    save_training_curves(history)
    HISTORY_PATH.write_text(
        json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
