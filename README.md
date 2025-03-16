[![Project Report](https://img.shields.io/badge/Project%20Report-Click%20Here-blue?style=for-the-badge)](https://wandb.ai/cs24m035-indian-institute-of-technology-madras/DA6401_Assignment_newtry/reports/DA6401-Assignment-1--VmlldzoxMTgyMzcwMg)

# Project Repository 📂

Welcome to this repository! 

## Table of Contents  
1. [Project Title & Description](#project-title--description)  
2. [Features](#features)  
3. [Installation & Dependencies](#installation--dependencies)  
4. [Usage](#usage)  
6. [Dataset](#dataset)    
7. [Results & Performance](#results--performance)
8. [License](#license)
9. [Acknowledgments](#acknowledgements)
    

# Fully Connected Feedforward Neural Network  

## Project Title & Description  

This project implements a **fully connected feedforward neural network** from scratch, including the **backpropagation algorithm** for training. The implementation strictly uses **NumPy** for all matrix and vector operations, avoiding any automatic differentiation libraries.  

The primary goal is to train the neural network for **image classification** using the **Fashion-MNIST dataset**, which consists of **grayscale images (28×28 pixels, 784 features) categorized into 10 different classes**.  

### 🔹 Key Objectives:  
- Develop a **manual backpropagation algorithm** for training the network.  
- Experiment with **hyperparameter tuning techniques**, including:  
  - Learning rate adjustments  
  - Varying the number of hidden layers  
  - Exploring different activation functions  
- Extend the implementation to the **MNIST dataset** and analyze performance variations across datasets.  
- Optimize classification accuracy using **various training strategies**.  

This project offers valuable insights into fundamental **deep learning concepts**, focusing on:  
✅ **Manual gradient computation**  
✅ **Optimization strategies**  
✅ **Hyperparameter tuning**  

By building this from scratch, we gain a deeper understanding of how neural networks operate **without relying on high-level deep learning frameworks**. 🚀  

## Features  

This project implements a fully customizable **feedforward neural network** with various tunable hyperparameters and optimization techniques.  

### 🔹 Supported Functionalities  

- **Number of epochs:** `{5, 10}`  
- **Number of hidden layers:** `{3, 4, 5}`  
- **Size of each hidden layer:** `{32, 64, 128}`  
- **Weight decay (L2 regularization):** `{0, 0.0005, 0.5}`  
- **Learning rate:** `{1e-3, 1e-4}`  
- **Optimizer choices:** `{SGD, Momentum, Nesterov, RMSprop, Adam, Nadam}`  
- **Batch size:** `{16, 32, 64}`  
- **Weight initialization strategies:** `{Random, Xavier}`  
- **Activation functions:** `{Sigmoid, Tanh, ReLU}`  

### 🔹 Hyperparameter Tuning  
To efficiently search for optimal hyperparameters, we utilized **Weights & Biases (WandB) Sweep functionality**, allowing automated experimentation with different configurations for improved performance.  

This flexibility makes the model **highly configurable and scalable**, enabling experimentation with various architectures and optimization strategies! 🚀  

## Installation & Dependencies  

### 🔹 Prerequisites  
Ensure you have **Python 3.x** installed on your system. You can check your Python version using:  

```bash
python --version
```
or  
```bash
python3 --version
```

### 🔹 Cloning the Repository  
To download this project, open a terminal and run:  

```bash
git clone https://github.com/Priyanshu999/Deep-Learning-Assignment-1.git
cd Deep-Learning-Assignment-1
```

### 🔹 Installing Dependencies  
This project includes a `requirements.txt` file containing all necessary dependencies. To install them, run:  

```bash
pip install -r requirements.txt
```
After installation, you're all set to run the project! 🚀


## Usage

### 🔹 Running Training Script
To train the neural network and track experiments using Weights & Biases, execute the following command:

```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

### 🔹 Supported Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.1 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.99 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 5 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | relu | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

## Dataset

This project supports two datasets: **MNIST** and **Fashion-MNIST**. The dataset selection can be configured using the `--dataset` argument.

### 🔹 MNIST
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size **28×28 pixels**. It is a benchmark dataset widely used for training and evaluating classification models in machine learning and deep learning.

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Classes:** 10 (digits 0-9)

### 🔹 Fashion-MNIST
Fashion-MNIST is a drop-in replacement for MNIST, designed to be more challenging. It consists of 70,000 grayscale images of **fashion items** across 10 categories, with the same image resolution (**28×28 pixels**).

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Classes:** 10 (e.g., T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

### 🔹 Dataset Integration with Weights & Biases
Both datasets can be used with Weights & Biases (WandB) for tracking model performance, visualizing accuracy, and analyzing results across different hyperparameter settings.

```bash
python train.py --dataset fashion_mnist --wandb_entity myname --wandb_project myprojectname
```

📌 **Note:** Ensure you specify the correct dataset during training to match the intended experiments!

## Results & Performance

### 🔹 Activation Function Impact
- Models using **ReLU** show a wide range of accuracy, from very poor to high accuracy.
- **Tanh-based** models tend to perform consistently well, often appearing in the higher accuracy range.
- **Sigmoid** activation generally leads to poor performance due to vanishing gradient issues.
- ReLU activation has a strong positive correlation with higher accuracy.
- Sigmoid activation shows a negative correlation, confirming suboptimal performance.

### 🔹 Batch Size and Accuracy
- Higher batch sizes (**>60**) tend to correlate with better validation accuracy.
- Smaller batch sizes (**<40**) frequently result in poor performance due to unstable gradients and lower generalization.
- Larger layers and batch sizes contribute positively, but batch size has diminishing returns.

### 🔹 Number of Hidden Layers & Layer Size
- Best models have **4-5 hidden layers** with layer sizes around **128-130**.
- Too few layers (**≤3**) underperform, while overly deep networks (**>5 layers**) show diminishing returns due to vanishing gradients.

### 🔹 Learning Rate & Optimizer Choice
- **Nadam** and **Adam** optimizers yield the highest validation accuracies.
- **SGD** and **RMSprop** lead to varied performance, with some configurations performing poorly.
- The best models use a **learning rate ~0.001**, while lower rates (**≤0.0003**) struggle to converge.
- Very high learning rates degrade performance; a moderate range (**0.001–0.01**) is optimal.
- **Adam** and **RMSprop** optimizers are positively correlated with accuracy.
- **Nesterov** and **Momentum-based** optimizers have a negative correlation, indicating struggles in comparison to adaptive optimizers.

### 🔹 Weight Initialization Impact
- **Xavier initialization** is consistently associated with high validation accuracy.
- **Random initialization** often leads to poor results due to unstable weight distributions.
- **Xavier weight initialization** is positively correlated with accuracy, reinforcing proper initialization benefits.

### 🔹 Weight Decay Regularization
- Minimal or zero weight decay (**≤0.1**) leads to better accuracy, indicating excessive regularization hinders learning.
- Weight decay negatively impacts accuracy in many cases, confirming that excessive regularization is counterproductive.

📊 **Scatter Plot Analysis**
(Attach your scatter plot image here)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
I would like to acknowledge the following resources that helped shape this project:

- [Mitesh Khapra Deep Learning Course](https://www.youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM)
- [Deep Learning - Stanford CS231N](https://www.youtube.com/playlist?list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ)


🔗 Happy Coding! 😊
