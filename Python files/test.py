from model import NeuralNetwork, Optimizer, ActivationFunctions
import wandb
# wandb.login()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist

import argparse

def main():
    wandb.login()
    
    parser = argparse.ArgumentParser(description="Train a Neural Network with wandb logging")

    # Required wandb args
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Wandb project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb entity name")

    # Training hyperparameters
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU", help="Activation function")

    args = parser.parse_args()

    # Initialize wandb
    # wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    if args.dataset == "mnist":
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    split_index = int(0.9*x_train.shape[0])
    x_train, x_val = x_train[:split_index], x_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]

    # One-hot encoding labels
    def one_hot_encode(y, num_classes=10):
        return np.eye(num_classes)[y]

    y_train_ohe = one_hot_encode(y_train)
    y_val_ohe = one_hot_encode(y_val)
    y_test_ohe = one_hot_encode(y_test)

    def train_sweep(losse):
      run=wandb.init(project=args.wandb_project, entity=args.wandb_entity)
      config = wandb.config

      # Generate a custom run name
      run_name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_ls_{losse}_lr_{config.learning_rate}_opt_{config.optimizer}_init_{config.weight_init}"
      wandb.run.name = run_name
      # wandb.run.save()

      loss_function = losse

      # Initialize and train the model
      model = NeuralNetwork(
          layers=[784] + [config.layer_size]*config.hidden_layers + [10],
          learning_rate=config.learning_rate,
          optimizer=config.optimizer,
          weight_decay=config.weight_decay,
          weight_init=config.weight_init,
          activation=config.activation,
          loss=loss_function,
          wandb_run=wandb
      )
      model.train(x_train, y_train_ohe, x_val, y_val_ohe, epochs=config.epochs, batch_size=config.batch_size)

      # Evaluate on test data after training
      test_loss, test_accuracy, y_true, y_pred = model.evaluate(x_test, y_test_ohe)

      # Log final test metrics
      # wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

      print(f"Final Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

      run.finish()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "Validation Accuracy", "goal": "maximize"},
        "parameters": {
            "epochs": {"values": [args.epochs]},
            "hidden_layers": {"values": [args.num_layers]},
            "layer_size": {"values": [args.hidden_size]},
            "weight_decay": {"values": [args.weight_decay]},
            "learning_rate": {"values": [args.learning_rate]},
            "optimizer": {"values": [args.optimizer]},
            "batch_size": {"values": [args.batch_size]},
            "weight_init": {"values": [args.weight_init]},
            "activation": {"values": [args.activation]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=lambda: train_sweep(args.loss), count=1)


    # Train model
    # model.train(x_train, y_train_ohe, x_val, y_val_ohe, epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate model
    # test_loss, test_accuracy, y_true, y_pred = model.evaluate(x_test, y_test_ohe)

    # Log final results
    # wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

    # print(f"Final Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    # wandb.finish()

if __name__ == "__main__":
    main()
