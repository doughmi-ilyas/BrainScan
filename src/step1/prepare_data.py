from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_DIR = Path("Data")
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_STATE = 42


def collect_image_paths(data_dir=DATA_DIR):
    """Return image paths and their folder names."""
    image_paths = []
    labels = []

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(image_path)
                labels.append(class_dir.name)

    if not image_paths:
        raise ValueError("No supported image was found in the dataset")

    return np.array(image_paths, dtype=object), np.array(labels)


def load_image(image_path):
    """Read, convert, resize and normalize one image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("image cannot be read")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        return image.astype(np.float32) / 255.0
    except (ValueError, cv2.error) as error:
        print(f"Image ignored: {image_path} ({error})")
        return None


def load_dataset(data_dir=DATA_DIR):
    """Load valid images and split them into train, validation and test sets."""
    image_paths, text_labels = collect_image_paths(data_dir)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(text_labels)

    images = []
    valid_labels = []
    for image_path, label in zip(image_paths, labels):
        image = load_image(image_path)
        if image is not None:
            images.append(image)
            valid_labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(valid_labels, dtype=np.int64)

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images,
        labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        encoder,
    )


def split_image_paths(data_dir=DATA_DIR):
    """Split image paths without loading all images into memory."""
    image_paths, text_labels = collect_image_paths(data_dir)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(text_labels)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths.astype(str),
        labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        encoder,
    )


def make_tf_dataset(image_paths, labels, batch_size=32, shuffle=False):
    """Create a small-memory TensorFlow dataset from image paths."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def read_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        return image, label

    if shuffle:
        dataset = dataset.shuffle(len(image_paths), seed=RANDOM_STATE)

    return dataset.map(
        read_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    data = load_dataset()
    print("Classes:", list(data[-1].classes_))
    print("Training:", len(data[0]))
    print("Validation:", len(data[2]))
    print("Test:", len(data[4]))
    print("Image size:", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
