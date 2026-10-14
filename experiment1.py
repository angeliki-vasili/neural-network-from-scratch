import numpy as np
from sklearn.model_selection import train_test_split

from fully_connected_nn import FullyConnectedNN
from generate_datasets import (
    generate_binary_problem,
    generate_flower_problem,
)
from helpers import plot_decision_boundary


def run_experiment(X, y, name):
    y = y.reshape(-1, 1)

    # Keep both classes in the training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y.ravel(),
    )

    model = FullyConnectedNN(
        input_size=2,
        hidden_size=3,
        lr=0.01,
    )

    model.fit(
        X_train,
        y_train,
        iterations=5000,
        show_step=500,
    )

    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    print(f"\n{name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")

    plot_decision_boundary(model.predict, X, y.ravel())


if __name__ == "__main__":
    np.random.seed(42)

    centers = np.array([[0, 8], [0, 8]])
    X, y = generate_binary_problem(centers, N=500)
    run_experiment(X, y, "First dataset")

    centers = np.array([[0, 2], [0, 2]])
    X, y = generate_binary_problem(centers, N=500)
    run_experiment(X, y, "Second dataset")

    X, y = generate_flower_problem()
    run_experiment(X.T, y, "Flower dataset")
