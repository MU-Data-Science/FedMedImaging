import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import flwr as fl
from pathlib import Path

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy, f1, precision, recall = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


# You need to run code in like this command: python3 client_6_models_AllClasses.py -i EfficientNetB0 -cl 2
# Pass the models's name and number of classes for your data.


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, default=0, choices=range(0, 10), required=False)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-cl", "--n_classes", required=True)
    args = vars(parser.parse_args())
    alt = parser.parse_args()
    input = str(args['input'])
    n_classes = int(args['n_classes'])
    # -----------------------------------------------
    if input == "EfficientNetB0":
        print("##########/EfficientNetB0 is running\############")

        model = tf.keras.applications.efficientnet.EfficientNetB0(
            input_shape=(256, 256, 3), weights=None, classes=n_classes
        )

    elif input == "ResNet50":
        print("##########/ResNet50 is running\############")

        model = tf.keras.applications.resnet50.ResNet50(
            input_shape=(256, 256, 3), weights=None, classes=n_classes
        )

    elif input == "DenseNet121":
        print("##########/DenseNet121 is running\############")

        model = tf.keras.applications.densenet.DenseNet121(
            input_shape=(256, 256, 3), weights=None, classes=n_classes
        )


    elif input == "InceptionResNetV2":
        print("##########/InceptionResNetV2 is running\############")

        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            input_shape=(256, 256, 3), weights=None, classes=n_classes
        )

    elif input == "InceptionV3":
        print("##########/InceptionV3 is running\############")

        model = tf.keras.applications.inception_v3.InceptionV3(
            input_shape=(256, 256, 3), weights=None, classes=n_classes
        )

    elif input == "MobileNetV2":
        print("##########/MobileNetV2 is running\############")

        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(256, 256, 3), weights=None, classes=n_classes
        )

    else:
        print(
            "Please pass your model's name as an argument from this list [EfficientNetB0,ResNet50,DenseNet121,InceptionResNetV2,InceptionV3,MobileNetV2]")
        quit()

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy", f1_m, precision_m, recall_m])

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(alt.partition)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client(
        server_address="10.10.1.1:5000",
        client=client,

    )


def load_partition(idx: int):

    assert idx in range(10)
    path = '' # enter dataset path here
    datagen = ImageDataGenerator(validation_split=0.1, rotation_range=90, horizontal_flip=True, vertical_flip=True,
                                 width_shift_range=0.2, height_shift_range=0.2, fill_mode="nearest")
    train_it = datagen.flow_from_directory(path + '/train', class_mode='binary', batch_size=64)
    test_it = datagen.flow_from_directory(path + '/test', class_mode='binary', batch_size=64)
    batch_size = 64
    train_it.reset()
    x_train, y_train = next(train_it)
    for i in range(len(train_it) - 1):  # 1st batch is already fetched before the for loop.
        img, label = next(train_it)
        x_train = np.append(x_train, img, axis=0)
        y_train = np.append(y_train, label, axis=0)
    test_it.reset()
    x_test, y_test = next(test_it)
    for i in range(len(test_it) - 1):  # 1st batch is already fetched before the for loop.
        img, label = next(test_it)
        x_test = np.append(x_test, img, axis=0)
        y_test = np.append(y_test, label, axis=0)

    return (
               x_train,
               y_train,
           ), (
               x_test,
               y_test,
           )


if __name__ == "__main__":
    main()
