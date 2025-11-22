import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import japanize_matplotlib
from tensorflow.keras import layers, models

SAMPLES_PER_CLASS = 1000


def load_subset_mnist(samples_per_class: int = SAMPLES_PER_CLASS):
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    np.random.seed(42)
    x_list = []
    y_list = []

    for digit in range(10):
        idx = np.where(y_train_full == digit)[0]

        selected_idx = np.random.choice(idx, samples_per_class, replace=False)
        x_list.append(x_train_full[selected_idx])
        y_list.append(y_train_full[selected_idx])

    x_train = np.concatenate(x_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_subset_mnist()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model = build_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
    )

    epochs = range(1, len(history.history["loss"]) + 1)

    # 学習曲線
    plt.figure()
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.plot(epochs, train_acc, marker="o", label="訓練精度")
    plt.plot(epochs, val_acc, marker="o", label="検証精度")

    for x, y in zip(epochs, train_acc):
        plt.annotate(f"{y:.3f}", xy=(x, y),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=8)
    for x, y in zip(epochs, val_acc):
        plt.annotate(f"{y:.3f}", xy=(x, y),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=8)

    plt.xlabel("学習回数")
    plt.ylabel("精度")
    plt.title("学習曲線（精度）")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve_accuracy_model_1000.png")
    plt.close()

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save("model_1000.keras")
    print("モデルを保存しましたわーーー")

if __name__ == "__main__":
    main()
