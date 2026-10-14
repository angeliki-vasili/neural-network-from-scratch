# Neural Network from Scratch

Academic project exploring binary classification with a neural network implemented in NumPy and a comparable model built with PyTorch.

## Technologies
Python, NumPy, PyTorch, scikit-learn and Matplotlib.

## Implementation
- Fully connected network with one hidden layer.
- ReLU and sigmoid activation functions.
- Binary cross-entropy loss.
- Forward propagation, backpropagation and gradient descent.
- Classification experiments on synthetic datasets and the scikit-learn breast cancer dataset.

## Files
- `fully_connected_nn.py`: NumPy neural network implementation.
- `experiment1.py`: Synthetic dataset experiments and decision boundary plots.
- `experiment2.py`: NumPy model evaluation over 20 runs.
- `experiment3.py`: PyTorch model evaluation over 20 runs.
- `generate_datasets.py`: Synthetic dataset generation.
- `helpers.py`: Plotting utilities.

## Running the experiments
Install the required libraries:
```bash
pip install numpy matplotlib scikit-learn torch
```

Run an experiment from the folder containing the Python files:
```bash
python experiment2.py
python experiment3.py
```

The scripts report classification accuracy and execution time.

## Limitations
The synthetic dataset experiment currently uses an ordered train/test split. Shuffling or a stratified split is needed for more reliable evaluation.

## Context
University coursework in Informatics and Telematics at Harokopio University.
