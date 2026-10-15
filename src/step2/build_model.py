from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

def build_model():
    model = Sequential([
        Input(shape=(224, 224, 3)),

        Conv2D(32,(3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.5),

        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )

    return model



if __name__ == "__main__":
    model = build_model()
    model.summary()










    