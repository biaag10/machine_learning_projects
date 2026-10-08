from loguru import logger
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from module_deep_learning.config import (
    IMG_SHAPE,
    L2_WEIGHT,
    LEARNING_RATE,
    NUM_CLASSES,
)
from module_deep_learning.features import build_data_augmentation


def conv_block(filters, dropout, l2=L2_WEIGHT):

    # Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool -> Dropout
    reg = regularizers.l2(l2)

    return [
        Conv2D(filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout),
    ]


def build_model(
    input_shape=IMG_SHAPE,
    num_classes=NUM_CLASSES,
    l2=L2_WEIGHT,
    learning_rate=LEARNING_RATE,
):

    reg = regularizers.l2(l2)

    model = keras.Sequential(
        [Input(shape=input_shape), build_data_augmentation()]
        + conv_block(32, 0.2, l2)    # 32x32 -> 16x16
        + conv_block(64, 0.3, l2)    # 16x16 -> 8x8
        + conv_block(128, 0.4, l2)   # 8x8   -> 4x4
        + [
            # Transição CNN => Dense
            GlobalAveragePooling2D(),
            Dense(128, use_bias=False, kernel_regularizer=reg),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="cnn_cifar10",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():

    model = build_model()

    model.summary(print_fn=logger.info)


if __name__ == "__main__":
    main()