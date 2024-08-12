from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# You need to run code in like this command: python3 client_6_models_AllClasses.py -i EfficientNetB0 -cl 2
# Pass the models's name and number of classes for your data.

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",required=True)
    ap.add_argument("-cl","--n_classes",required=True)
    args = vars(ap.parse_args())
    input = str(args['input'])
    n_classes = int(args['n_classes'])

    if input == "EfficientNetB0":
        print("##########/EfficientNetB0 is running\############")
        model = tf.keras.applications.efficientnet.EfficientNetB0(
            input_shape=(256, 256, 3), weights=None, classes= n_classes
        )

    elif input == "ResNet50":
        print("##########/ResNet50 is running\############")
        model = tf.keras.applications.resnet50.ResNet50(
            input_shape=(256, 256, 3), weights=None, classes= n_classes
        )


    elif input == "DenseNet121":
        print("##########/DenseNet121 is running\############")
        model = tf.keras.applications.densenet.DenseNet121(
            input_shape=(256, 256, 3), weights=None, classes= n_classes
        )


    elif input == "InceptionResNetV2":
        print("##########/InceptionResNetV2 is running\############")
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            input_shape=(256, 256, 3), weights=None, classes= n_classes
        )


    elif input == "InceptionV3":
        print("##########/InceptionV3 is running\############")
        model = tf.keras.applications.inception_v3.InceptionV3(
            input_shape=(256, 256, 3), weights=None, classes= n_classes
        )


    elif input == "MobileNetV2":
        print("##########/MobileNetV2 is running\############")
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(256, 256, 3), weights=None, classes= n_classes
        )

    else:
        print("Please pass your model's name as an argument from this list [EfficientNetB0,ResNet50,DenseNet121,InceptionResNetV2,InceptionV3,MobileNetV2]")
        quit()

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        )

    
    
    fl.server.start_server(
        server_address="[::]:5000",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )




def fit_config(rnd: int):

    config = {
        "batch_size": 32,
        "local_epochs": 20,
    }
    return config


def evaluate_config(rnd: int):

    val_steps = 50 if rnd < 4 else 50
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
