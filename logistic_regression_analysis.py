import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import csv
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def sigmoid(z):
    """Convert a score into a probability using a stable sigmoid."""
    return np.exp(-np.logaddexp(0, -z))


def run_part1():
    """Calculate the four manually specified cases."""
    cases = [
        ("A", -1.0, 0.8, 2.0, 0.50),
        ("B", -1.0, 0.8, 2.0, 0.70),
        ("C", -1.0, -0.8, 2.0, 0.50),
        ("D", 1.0, 0.0, 2.0, 0.50),
    ]

    print("\nPart 1: Score, Probability, and Predicted Class")
    print(f"{'Case':<6}{'Score':>10}{'Probability':>14}"
          f"{'Threshold':>12}{'Class':>8}")

    for name, beta0, beta1, x, threshold in cases:
        # Calculate the linear score.
        z = beta0 + beta1 * x

        # Convert the score into the probability of class 1.
        probability = sigmoid(z)

        # Compare the unrounded probability with the threshold.
        predicted_class = int(probability >= threshold)

        print(
            f"{name:<6}{z:>10.4f}{probability:>14.4f}"
            f"{threshold:>12.2f}{predicted_class:>8}"
        )


def load_and_inspect_data():
    """Load the required features and encode malignant as class 1."""
    dataset = load_breast_cancer()

    # Keep the two required features in the specified order.
    feature_names = ["mean radius", "mean texture"]
    feature_indices = [
        list(dataset.feature_names).index(name)
        for name in feature_names
    ]
    X = dataset.data[:, feature_indices]

    # Identify malignant from the dataset's original class names.
    malignant_code = list(dataset.target_names).index("malignant")

    # Recode the target: 1 = malignant, 0 = benign.
    y = (dataset.target == malignant_code).astype(int)

    print("\nPart 2: Dataset Inspection")
    print("Features:", feature_names)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target coding: 0 = benign, 1 = malignant")

    # Count missing values separately for each selected feature.
    print("\nMissing values:")
    for name, count in zip(feature_names, np.isnan(X).sum(axis=0)):
        print(f"{name}: {int(count)}")

    print("Missing target values:", int(np.isnan(y).sum()))

    # Report the class counts after recoding.
    print("\nFull dataset class counts:")
    print("0 = benign:", int(np.sum(y == 0)))
    print("1 = malignant:", int(np.sum(y == 1)))
    print("Total:", len(y))

    # Display a few rows to check feature order and target coding.
    print("\nFirst 5 rows:")
    print(f"{'mean radius':>14}{'mean texture':>15}{'target':>10}")
    for features, target in zip(X[:5], y[:5]):
        print(f"{features[0]:>14.4f}{features[1]:>15.4f}"
              f"{target:>10}")

    # Stop execution if the selected data contain missing values.
    assert not np.isnan(X).any(), "Missing feature values detected."

    return X, y, feature_names

def split_and_scale_data(X, y):
    """Create the required partitions and fit scaling on training only."""
    indices = np.arange(len(y))

    # First split: reserve 20% of all rows for the final test.
    development_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Second split: reserve 25% of development rows for validation.
    # Preserve the order returned by both split operations.
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=0.25,
        stratify=y[development_indices],
        random_state=42,
    )

    # Verify that the three partitions do not overlap.
    assert np.intersect1d(
        train_indices, validation_indices
    ).size == 0, "Training and validation overlap."

    assert np.intersect1d(
        train_indices, test_indices
    ).size == 0, "Training and test overlap."

    assert np.intersect1d(
        validation_indices, test_indices
    ).size == 0, "Validation and test overlap."

    # Verify that every original row appears exactly once.
    combined_indices = np.concatenate(
        [train_indices, validation_indices, test_indices]
    )
    assert combined_indices.size == len(y), "Incorrect total row count."
    assert np.unique(combined_indices).size == len(y), (
        "Some rows are duplicated or missing."
    )

    print("\nPart 2: Data Splitting")
    print(f"{'Split':<14}{'Total':>8}{'Benign (0)':>14}"
          f"{'Malignant (1)':>16}")

    for name, partition_indices in [
        ("training", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ]:
        labels = y[partition_indices]
        print(
            f"{name:<14}{len(labels):>8}"
            f"{int(np.sum(labels == 0)):>14}"
            f"{int(np.sum(labels == 1)):>16}"
        )

    print("Partition checks passed: no overlap and no missing rows.")

    # Extract features and labels without changing row order.
    X_train = X[train_indices]
    X_validation = X[validation_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_validation = y[validation_indices]
    y_test = y[test_indices]

    # Learn the scaling parameters from training features only.
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Apply the same fitted scaler to all three partitions.
    X_train_scaled = scaler.transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # Confirm that the scaler was fitted on training rows only.
    assert int(scaler.n_samples_seen_) == len(train_indices)

    # Training features should have mean 0 and standard deviation 1.
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-12)

    print("\nPart 2: Scaling")
    print("Scaler fitted on training rows:", scaler.n_samples_seen_)
    print("Feature order: mean radius, mean texture")
    print(
        "Scaled training means:",
        np.round(X_train_scaled.mean(axis=0), 6),
    )
    print(
        "Scaled training standard deviations:",
        np.round(X_train_scaled.std(axis=0), 6),
    )
    print("Scaling checks passed.")

    return {
        "X_train": X_train_scaled,
        "X_validation": X_validation_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "scaler": scaler,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
    }

def binary_cross_entropy_from_scores(z, y):
    """Calculate mean binary cross-entropy using stable score arithmetic."""
    return np.mean(np.logaddexp(0, z) - y * z)


def train_numpy_model(X_train, y_train, feature_names):
    """Train logistic regression using full-batch gradient descent."""
    learning_rate = 0.1
    epochs = 20000
    n_train = len(y_train)

    # Add a leading column of ones for the intercept.
    X_aug = np.column_stack(
        [np.ones(n_train), X_train]
    )

    # Initialize the intercept and both feature coefficients to zero.
    beta = np.zeros(X_aug.shape[1], dtype=float)

    # Record the loss before any coefficient updates (epoch 0).
    initial_scores = X_aug @ beta
    loss_history = [
        binary_cross_entropy_from_scores(initial_scores, y_train)
    ]

    for epoch in range(1, epochs + 1):
        # Calculate probabilities using the current coefficients.
        scores = X_aug @ beta
        probabilities = sigmoid(scores)

        # Calculate one full-batch gradient for all coefficients.
        gradient = X_aug.T @ (probabilities - y_train) / n_train

        # Update all coefficients simultaneously.
        beta = beta - learning_rate * gradient

        # Record loss after the update, including the final update.
        updated_scores = X_aug @ beta
        updated_loss = binary_cross_entropy_from_scores(
            updated_scores, y_train
        )
        loss_history.append(updated_loss)

    loss_history = np.asarray(loss_history)

    # Check that training produced finite coefficients and losses.
    assert np.isfinite(beta).all(), "Non-finite coefficients detected."
    assert np.isfinite(loss_history).all(), "Non-finite loss detected."
    assert len(loss_history) == epochs + 1
    assert loss_history[-1] < loss_history[0], (
        "Training loss did not decrease."
    )

    # Measure the gradient at the final coefficients.
    final_probabilities = sigmoid(X_aug @ beta)
    final_gradient = (
        X_aug.T @ (final_probabilities - y_train) / n_train
    )

    print("\nPart 3: NumPy Logistic Regression")
    print("Training method: full-batch gradient descent")
    print("Learning rate:", learning_rate)
    print("Epochs:", epochs)
    print("Regularization: none")
    print("Class weights: none")
    print(f"Initial training loss: {loss_history[0]:.9f}")
    print(f"Final training loss:   {loss_history[-1]:.9f}")
    print("Recorded loss values:", len(loss_history))
    print(
        f"Final gradient norm: {np.linalg.norm(final_gradient):.9e}"
    )

    print("\nLearned coefficients (standardized-unit coefficients):")
    coefficient_names = ["intercept", *feature_names]
    for name, value in zip(coefficient_names, beta):
        print(f"{name}: {value:.9f}")

    # Resolve the output folder relative to this script.
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_path = reports_dir / "training_loss.png"

    # Include epoch 0 and every subsequent update in the plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(epochs + 1), loss_history)
    ax.set_title("NumPy Logistic Regression: Training Loss")
    ax.set_xlabel("Epoch (0 = before training)")
    ax.set_ylabel("Mean Binary Cross-Entropy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\nTraining loss plot saved to:", plot_path)

    return beta, loss_history

def positive_class_probability(model, X):
    """Return the probability column corresponding to class 1."""
    positive_column = np.flatnonzero(model.classes_ == 1)

    assert positive_column.size == 1, (
        "The model must contain exactly one class labeled 1."
    )

    return model.predict_proba(X)[:, positive_column[0]]


def fit_and_compare_reference_models(data, numpy_beta, feature_names):
    """Fit reference models and compare validation probabilities."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_validation = data["X_validation"]

    # Fit the required unregularized reference model.
    # Use the same scaled training features as the NumPy model.
    # Let scikit-learn fit its own intercept; do not add a ones column.
    reference_model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-9,
    )
    reference_model.fit(X_train, y_train)

    # Fit the baseline using training class proportions.
    baseline_model = DummyClassifier(strategy="prior")
    baseline_model.fit(X_train, y_train)

    # Combine the reference intercept and coefficients for comparison.
    reference_beta = np.concatenate(
        [reference_model.intercept_, reference_model.coef_[0]]
    )

    print("\nPart 4: Coefficient Comparison")
    print("Units: standardized-unit coefficients")
    print(
        f"{'Coefficient':<18}{'NumPy':>15}"
        f"{'scikit-learn':>18}{'Absolute diff':>18}"
    )

    coefficient_names = ["intercept", *feature_names]

    for name, numpy_value, reference_value in zip(
        coefficient_names, numpy_beta, reference_beta
    ):
        print(
            f"{name:<18}{numpy_value:>15.9f}"
            f"{reference_value:>18.9f}"
            f"{abs(numpy_value - reference_value):>18.9e}"
        )

    # Generate NumPy validation probabilities without retraining.
    numpy_validation_probabilities = sigmoid(
        numpy_beta[0] + X_validation @ numpy_beta[1:]
    )

    # Identify the class 1 probability column using classes_.
    reference_validation_probabilities = positive_class_probability(
        reference_model, X_validation
    )

    mean_absolute_difference = np.mean(
        np.abs(
            numpy_validation_probabilities
            - reference_validation_probabilities
        )
    )

    print("\nValidation Probability Comparison")
    print("Reference classes_:", reference_model.classes_)
    print("Positive class: 1 = malignant")
    print(
        "Mean absolute probability difference: "
        f"{mean_absolute_difference:.12e}"
    )
    print("Reference optimizer iterations:", reference_model.n_iter_)

    # Inspect the baseline probabilities learned from training labels.
    baseline_validation_probabilities = positive_class_probability(
        baseline_model, X_validation
    )

    print("\nMajority Baseline")
    print("Baseline classes_:", baseline_model.classes_)
    print(
        "Training malignant proportion: "
        f"{np.mean(y_train):.9f}"
    )
    print(
        "Baseline class 1 probability: "
        f"{baseline_validation_probabilities[0]:.9f}"
    )
    print(
        "Baseline predicted class at threshold 0.50:",
        int(baseline_validation_probabilities[0] >= 0.50),
    )

    return {
        "reference_model": reference_model,
        "baseline_model": baseline_model,
        "numpy_validation_probabilities":
            numpy_validation_probabilities,
        "reference_validation_probabilities":
            reference_validation_probabilities,
        "baseline_validation_probabilities":
            baseline_validation_probabilities,
        "reference_beta": reference_beta,
        "mean_absolute_probability_difference":
            mean_absolute_difference,
    }

def calculate_classification_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a given threshold."""
    predictions = (probabilities >= threshold).astype(int)

    # Rows = actual classes; columns = predicted classes.
    # Class 0 = benign; class 1 = malignant.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def select_validation_threshold(y_validation, probabilities):
    """Select the threshold using NumPy validation F1 only."""
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    results = []

    for threshold in thresholds:
        metrics = calculate_classification_metrics(
            y_validation,
            probabilities,
            threshold,
        )
        results.append({"threshold": threshold, **metrics})

    # Maximize F1; if tied, prefer the smallest threshold.
    # Selection uses full precision, not the rounded display values.
    best_result = max(
        results,
        key=lambda row: (row["f1"], -row["threshold"]),
    )
    selected_threshold = best_result["threshold"]

    print("\nPart 5: NumPy Validation Threshold Selection")
    print("Positive class: 1 = malignant")
    print(
        f"{'Threshold':>10}{'TN':>5}{'FP':>5}{'FN':>5}{'TP':>5}"
        f"{'Accuracy':>12}{'Precision':>12}"
        f"{'Recall':>12}{'F1':>12}"
    )

    for row in results:
        print(
            f"{row['threshold']:>10.2f}"
            f"{row['tn']:>5}{row['fp']:>5}"
            f"{row['fn']:>5}{row['tp']:>5}"
            f"{row['accuracy']:>12.6f}"
            f"{row['precision']:>12.6f}"
            f"{row['recall']:>12.6f}"
            f"{row['f1']:>12.6f}"
        )

    print(f"\nSelected threshold: {selected_threshold:.2f}")
    print(f"Selected validation F1: {best_result['f1']:.9f}")
    print("Selection rule: highest F1, then smallest threshold.")
    print("Threshold fixed before test evaluation.")
    print("No models or scaler are refitted.")

    # Export the seven candidates in the required column order.
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "threshold_results.csv"

    fieldnames = [
        "threshold", "tn", "fp", "fn", "tp",
        "accuracy", "precision", "recall", "f1",
    ]

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            formatted_row = row.copy()
            formatted_row["threshold"] = f"{row['threshold']:.2f}"

            for metric in ["accuracy", "precision", "recall", "f1"]:
                formatted_row[metric] = f"{row[metric]:.9f}"

            writer.writerow(formatted_row)

    print("Threshold results saved to:", output_path)

    return selected_threshold, results

def evaluate_and_export_results(data, models, numpy_beta, selected_threshold):
    """Evaluate the fixed policies on validation, then test."""
    results = []

    print("\nPart 5: Final Evaluation")
    print(f"Fixed selected threshold: {selected_threshold:.2f}")
    print("Class 0 = benign; class 1 = malignant")
    print("Confusion matrix rows = actual; columns = predicted")

    # Finish validation evaluation before evaluating the test partition.
    for split in ["validation", "test"]:
        X_split = data[f"X_{split}"]
        y_split = data[f"y_{split}"]

        # Generate probabilities using the already fitted models.
        # No fitting, scaling updates, or threshold selection occurs here.
        numpy_probabilities = sigmoid(
            numpy_beta[0] + X_split @ numpy_beta[1:]
        )
        reference_probabilities = positive_class_probability(
            models["reference_model"], X_split
        )
        baseline_probabilities = positive_class_probability(
            models["baseline_model"], X_split
        )

        configurations = [
            ("baseline", "default", 0.50, baseline_probabilities),
            ("numpy", "default", 0.50, numpy_probabilities),
            ("numpy", "selected", selected_threshold, numpy_probabilities),
            ("sklearn", "default", 0.50, reference_probabilities),
            (
                "sklearn",
                "selected",
                selected_threshold,
                reference_probabilities,
            ),
        ]

        print(f"\nSplit: {split}")
        print(
            f"{'Model':<10}{'Policy':<10}{'Threshold':>10}"
            f"{'TN':>5}{'FP':>5}{'FN':>5}{'TP':>5}"
        )

        split_results = []

        for model_name, policy, threshold, probabilities in configurations:
            metrics = calculate_classification_metrics(
                y_split, probabilities, threshold
            )

            row = {
                "split": split,
                "model": model_name,
                "policy": policy,
                "threshold": threshold,
                **metrics,
                # Log loss uses probabilities, not predicted labels.
                "log_loss": float(
                    log_loss(y_split, probabilities, labels=[0, 1])
                ),
            }

            assert (
                row["tn"] + row["fp"] + row["fn"] + row["tp"]
                == len(y_split)
            ), "Confusion counts do not match the partition size."

            results.append(row)
            split_results.append(row)

            print(
                f"{model_name:<10}{policy:<10}{threshold:>10.2f}"
                f"{row['tn']:>5}{row['fp']:>5}"
                f"{row['fn']:>5}{row['tp']:>5}"
            )

        # Print a separate metrics table to keep the output readable.
        print()
        print(
            f"{'Model':<10}{'Policy':<10}"
            f"{'Accuracy':>12}{'Precision':>12}"
            f"{'Recall':>12}{'F1':>12}{'Log loss':>12}"
        )

        for row in split_results:
            print(
                f"{row['model']:<10}{row['policy']:<10}"
                f"{row['accuracy']:>12.6f}"
                f"{row['precision']:>12.6f}"
                f"{row['recall']:>12.6f}"
                f"{row['f1']:>12.6f}"
                f"{row['log_loss']:>12.6f}"
            )

    # Verify that all ten required configurations are present.
    assert len(results) == 10
    assert sum(row["split"] == "validation" for row in results) == 5
    assert sum(row["split"] == "test" for row in results) == 5

    fieldnames = [
        "split", "model", "policy", "threshold",
        "tn", "fp", "fn", "tp",
        "accuracy", "precision", "recall", "f1", "log_loss",
    ]

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "assignment_results.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            formatted_row = row.copy()
            formatted_row["threshold"] = f"{row['threshold']:.2f}"

            for metric in [
                "accuracy", "precision", "recall", "f1", "log_loss"
            ]:
                formatted_row[metric] = f"{row[metric]:.9f}"

            writer.writerow(formatted_row)

    print("\nPart 6: Export Complete")
    print("Assignment result rows:", len(results))
    print("Assignment results saved to:", output_path)

    return results


if __name__ == "__main__":
    run_part1()
    X, y, feature_names = load_and_inspect_data()
    data = split_and_scale_data(X, y)

    numpy_beta, loss_history = train_numpy_model(
        data["X_train"],
        data["y_train"],
        feature_names,
    )

    models = fit_and_compare_reference_models(
        data,
        numpy_beta,
        feature_names,
    )

    selected_threshold, threshold_results = select_validation_threshold(
        data["y_validation"],
        models["numpy_validation_probabilities"],
        
    )

    assignment_results = evaluate_and_export_results(
        data,
        models,
        numpy_beta,
        selected_threshold,
    )