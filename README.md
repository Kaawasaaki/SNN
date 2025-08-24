# Interactive Visualization of a Convolutional Spiking Neural Network

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-important.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-orange.svg)](https://streamlit.io)

## Abstract

This project presents the development, training, and visualization of a Convolutional Spiking Neural Network (CSNN), a class of bio-inspired models that emulate the brain's event-driven and energy-efficient processing. The primary objective was to gain a practical understanding of neuromorphic computing principles. A CSNN was implemented from scratch using PyTorch and the snnTorch library, employing Leaky Integrate-and-Fire (LIF) neuron models. The model was trained on the MNIST handwritten digit dataset, achieving a validation accuracy of 97%.

To demonstrate and analyze the model's behavior, an interactive web application was developed using Streamlit. This application provides real-time inference on user-drawn digits and offers a unique view into the network's internal temporal dynamics, visualizing the flow of information as discrete spike events over time.

## Key Features

- **Convolutional SNN Implementation:** A robust CSNN architecture built with PyTorch and snnTorch.
- **High-Accuracy Training:** The model is pre-trained to 97% accuracy on the MNIST dataset.
- **Interactive Web Application:** A Streamlit-based GUI allows users to draw digits and receive instant predictions.
- **Real-Time Dynamic Visualization:** The application visualizes four key stages of the SNN's inference process:
    1.  The processed input tensor.
    2.  Spike-maps from the initial convolutional layer at a specific timestep.
    3.  A comprehensive spike raster plot of the first fully-connected layer over the entire time window.
    4.  A final bar chart aggregating the total spike counts from the output neurons, representing the network's final "vote".

## Model Architecture

The network (`MemoryOptimizedCSNN`) is a deep convolutional architecture designed for feature extraction from static images, processed over a temporal window.

- **Structure:** 4 convolutional blocks followed by 2 fully-connected layers.
- **Neuron Model:** All spiking neurons are Leaky Integrate-and-Fire (LIF) models.
- **Downsampling:** Stride-based downsampling is used in the convolutional layers to reduce spatial dimensions while increasing feature depth.
- **Regularization:** The model employs Dropout and Batch Normalization to prevent overfitting and stabilize training.

## Core Concepts Explored

This project provides a practical implementation of several advanced concepts in neural networks:

- **Event-Driven Computation:** Understanding how SNNs process information using discrete, asynchronous spike events, which is foundational to their efficiency on specialized neuromorphic hardware.
- **Temporal Information Processing:** Learning how SNNs integrate information over a defined time window, using the membrane potential of neurons as a form of short-term memory.
- **Surrogate Gradients:** Implementing a solution to train non-differentiable spiking neurons by substituting the Heaviside step function's derivative with a continuous approximation (e.g., fast sigmoid) during the backpropagation phase.

## Getting Started

Follow these instructions to set up the environment and run the project locally.

### Prerequisites

- Python 3.9 or higher
- `pip` and `venv` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Workflow: Training and Visualization

This project has two distinct operational modes: training the model from scratch and visualizing a pre-trained model.

### Step 1: Training the Model (Optional)

If you wish to retrain the model or experiment with its architecture, you can execute the training script.

-   **Command:**
    ```bash
    python train.py 
    ```
    *(Note: Please replace `train.py` with the actual name of your training script if it is different. Ensure it saves the output model to the `models_mnist_opt` directory.)*

-   **Output:** This process will generate a new model checkpoint file (e.g., `model.pth`) containing the trained weights.

### Step 2: Visualizing the Pre-Trained Model

The primary function of this repository is to visualize the workings of the SNN. The Streamlit application is for **inference and visualization only**; it does not perform training.

-   **Pre-trained Model Requirement:**
    This project includes a pre-trained model file located at `models_mnist_opt/best_model_for_42_memory_optimized.pth`. The visualization script is configured to look for the model in this specific path. **Ensure this file is present in the `models_mnist_opt` directory to run the visualization.**

-   **Launch Command:**
    The visualization script is located inside the `models_mnist_opt` directory. To run the interactive web application, first navigate into this directory and then execute the Streamlit command:
    ```bash
    cd models_mnist_opt
    streamlit run visualize.py
    ```
-   A new tab will open in your web browser with the interactive demo.

## Repository File Structure

```
.
├── models_mnist_opt/
│   ├── best_model_for_42_memory_optimized.pth  # Pre-trained model weights
│   └── visualize.py                            # Streamlit script for the visualization app
├── train.py                                    # Script for training the model from scratch
├── README.md                                   # This documentation file
├── requirements.txt                            # List of Python dependencies for pip
└── .gitignore                                  # Specifies files for Git to ignore
```

## Results

The implemented CSNN model achieves a **97% accuracy** on the MNIST validation set after 20 epochs of training, demonstrating its effectiveness in classifying static images through temporal processing.

## Future Work and Extensions

The current framework serves as an excellent baseline for further research, particularly in the domain of cybersecurity:

-   **Adversarial Attack Implementation:** Investigate the model's robustness by implementing gradient-based attacks like the Fast Gradient Sign Method (FGSM). This is possible due to the use of surrogate gradients.
-   **Inference-Time Defense Mechanisms:** Implement and evaluate post-hoc defenses against adversarial attacks, such as input preprocessing (e.g., Gaussian blurring) or temporal consistency checks, which are unique to SNNs.
-   **Quantitative Robustness Evaluation:** Generate plots of model accuracy versus attack strength (epsilon) to quantitatively measure the effectiveness of implemented attacks and defenses.
