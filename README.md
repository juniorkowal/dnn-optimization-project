# DNN Optimization Project

This is an experimental project, created as a part of a university course, focusing on basic deep learning techniques. It explores simple model optimizations (e.g. pruning) for a coin classification task using a Kaggle dataset.

The main goal is to reduce inference time and resource consumption while maintaining reasonable accuracy.

Dataset source: [Kaggle Coin Images](https://www.kaggle.com/datasets/wanderdust/coin-images/data)

## Features

- **ResNet-50 base model**: A standard deep learning model used for image classification tasks.
- **Optimizations**: The project explores techniques like model pruning, weight quantization, and compression to improve performance.

## Project Workflow

1. **Baseline Model Training**: First, we train a baseline model using ResNet-50 on the original dataset, which contains 211 coin classes.
2. **Class Remapping**: Next, we remap the 211 classes to 32 classes based on currency categories to simplify the task for practical purposes.
3. **Fine-Tuning and Optimization**: The model is fine-tuned on the new 32-class dataset, and we apply various optimizations like pruning and quantization to improve the model's performance.
4. **Testing**: We test the effects of these optimizations on inference time and accuracy on the simplified task.

## Installation

To set up the project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/juniorkowal/dnn-optimization-project.git

# Navigate to the project directory
cd dnn-optimization-project

# Set up a virtual environment (optional but recommended)
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
```

## Usage

To run the project, simply execute main.py. The script will automatically download the dataset, perform preprocessing, train the baseline model, and run the optimization experiments.

```bash
python main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.