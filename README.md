# Robust Federated Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with Fairness

This repository contains my ongoing work for my Master’s Thesis, which focuses on Robust Federated Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with Fairness. The project is built upon a base Deep Deterministic Policy Gradient (DDPG) model, evolving into a Multi-Agent DDPG (MADDPG) model. It is a work in progress, and additional enhancements and components will be added as research continues.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Training Script](#running-the-training-script)
  - [Monitoring Training with TensorBoard](#monitoring-training-with-tensorboard)
- [Project Structure](#project-structure)
- [Customization](#customization)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

This project demonstrates how to train multiple agents using the MADDPG algorithm in the **Simple Tag** environment. The agents learn to chase or evade each other in a cooperative-competitive setting. The environment and agents are simulated using the **Vectorized Multi-Agent Simulator (VMAS)**.

## Features

- Implementation of MADDPG algorithm using PyTorch and TorchRL.
- Training in a multi-agent environment with chasers and evaders.
- Logging of training metrics using TensorBoard.
- Saving training plots for visualization.
- Modular code with separate files for models and main training script.

## Prerequisites

- **Python 3.8** or higher
- **CUDA-compatible GPU** (optional but recommended for faster training)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/maddpg-simple-tag.git
   cd maddpg-simple-tag
   ```
2. **Create a Virtual Environment** (optional but recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```
3. **Install the Required Packages**
    Install the packages listed in requirements.txt:

   ```bash
    pip install -r requirements.txt
    ```
   **Note:** Ensure that you have the specified versions installed to avoid compatibility issues.

## Usage

### Running the Training Script

To train the agents in the **Simple Tag** environment, run the following command:

```bash
python train.py
```

The training script will start the training process and display the training metrics in the console. The trained models will be saved in the `models` directory.

### Monitoring Training with TensorBoard

To monitor the training progress using TensorBoard, run the following command in a separate terminal:

```bash
# tensorboard --logdir=DDPG/runs/DDPG_simple_tag # Uncomment for DDPG
tensorboard --logdir=MADDPG/runs/MADDPG_simple_tag # For MADDPG, comment out the DDPG line if not already
```

Then, open a web browser and go to `http://localhost:6006/` to view the training metrics in real-time.
You will see metrics such as episode reward mean for each agent group plotted over the training iterations.

## Project Structure

The project directory is organized as follows:

	•	main.py: The main script that sets up the environment, initializes components, and runs the training loop.
	•	models.py: Contains functions to create the policy and critic networks.
	•	requirements.txt: Lists all necessary dependencies with specific versions.
	•	README.md: This readme file.
	•	results/: Directory where training logs and plots are saved.
	•	logs/: Contains TensorBoard logs for each training run.
	•	maddpg_simple_tag/: Experiment logs directory.
	•	run1/, run2/, etc.: Subdirectories for each run.
	•	plots/: Contains training plots saved as images.

## Customization

### Hyperparameter Tuning

You can improve the model’s performance by adjusting hyperparameters in main.py and models.py. Some parameters to consider:

	•	Learning Rate (lr)
	•	Batch Size (train_batch_size)
	•	Number of Optimizer Steps (n_optimiser_steps)
	•	Discount Factor (gamma)
	•	Polyak Averaging Coefficient (polyak_tau)
	•	Exploration Noise Parameters (sigma_init, sigma_end)

### Model Architecture

Adjust the model architecture in models.py:

	•	Network Depth (depth)
	•	Number of Cells (num_cells)
	•	Activation Functions (activation_class)

For more advanced tuning, consider implementing automated hyperparameter optimization tools like Optuna.

## Results

After training, you can find the training plots in the results/plots/ directory. Each plot shows the training progress for a specific run, named as maddpg_simple_tag_training_plot_runX.png, where X is the run number.

Example of how the MADDPG directory structure looks after multiple runs:
```bash
results/
├── logs/
│   └── MADDPG_simple_tag/
│       ├── run1/
│       ├── run2/
│       └── run3/
└── plots/
    ├── MADDPG_simple_tag_training_plot_run1.png
    ├── MADDPG_simple_tag_training_plot_run2.png
    └── MADDPG_simple_tag_training_plot_run3.png
```

## References

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TorchRL Documentation](https://torchrl.readthedocs.io/en/latest/)
- [VMAS Documentation](https://vmas.readthedocs.io/en/latest/)
- [TensorBoard Documentation](https://pytorch.org/docs/stable/tensorboard.html)
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.