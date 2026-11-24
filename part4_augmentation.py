import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import time
from joblib import Parallel, delayed

start_time = time.perf_counter()

def process_batch(X, y, model, sd, size, start, end, k, seed):
    '''Generate random indices, make predictions and calculate accuracy scores'''
    rng = np.random.default_rng(seed)
    b = end - start # batch size

    # Random sampling
    random_indices = rng.choice(len(X), size=(b, size))
    X_sample = X[random_indices]
    y_sample = y[random_indices]

    X_augmented = X_sample + rng.normal(0, sd, size=X_sample.shape) # add Gaussian noise

    # Predict all samples in the batch
    preds = model.predict(X_augmented.reshape(-1, X.shape[1])).reshape(b, size)

    # Calculate accuracy and F1-score
    accuracies = np.mean(y_sample == preds, axis=1)

    tp = np.sum((y_sample == 1) & (preds == 1), axis=1)
    fp = np.sum((y_sample == 0) & (preds == 1), axis=1)
    fn = np.sum((y_sample == 1) & (preds == 0), axis=1)

    f1_scores = 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    two_augs = []

    # Save the last two augmented datasets
    for j in range(max(0, k - 2 - start), b):
        two_augs.append([X_augmented[j].copy(), y_sample[j].copy()])

    return accuracies, f1_scores, two_augs


def data_augmentation(X, y, model, sd, size=1000, k=100000, batch_size=100, nprocs=16):
    '''Sample, augment and evaluate multiple batches in parallel'''

    starts = list(range(0, k, batch_size)) # create batches
    seeds = np.random.SeedSequence(42).spawn(len(starts)) # independent child seeds generation

    results = Parallel(n_jobs=nprocs, verbose=10)(
        delayed(process_batch)(X, y, model, sd, size, start, min(start + batch_size, k), k, seed)
        for start, seed in zip(starts, seeds)
    )

    accuracies = np.concatenate([result[0] for result in results])
    f1_scores = np.concatenate([result[1] for result in results])
    two_augs = [aug for result in results for aug in result[2]]

    return accuracies, f1_scores, two_augs

def plot_decision_boundary(ax, model, X, y, title):
    '''Plot the model's best separator by applying the model over a dense grid'''

    # Set min and max values and add some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create the plot grid
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    preds = model.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    preds = preds.reshape(grid_x.shape)

    ax.contourf(grid_x, grid_y, preds, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor='k', s=15, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")


np.random.seed(42)
n = 10000
n_half = n // 2

# Circular Dataset (Archimedean spirals)
# Spiral 1 (Class 0)
theta1 = np.linspace(1, 3 * np.pi, n_half) + np.random.normal(0, 0.15, n_half)
r1 = 0.65 * theta1 + np.random.normal(0, 0.15, n_half)
X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)]) + np.random.normal(0, 0.25, size=(n_half, 2))

# Spiral 2 (Class 1)
theta2 = np.linspace(1, 3 * np.pi, n_half) + np.random.normal(0, 0.15, n_half)
r2 = 0.65 * theta2 + np.random.normal(0, 0.15, n_half)
X2 = np.column_stack([r2 * np.cos(theta2 + np.pi), r2 * np.sin(theta2 + np.pi)]) + np.random.normal(0, 0.25, size=(n_half, 2))

X_circular = np.vstack([X1, X2])
y_circular = np.array([0] * n_half + [1] * n_half)

# Initialize and train the SVM model
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_rbf.fit(X_circular, y_circular)

# Run data augmentation
augment_accuracies, augment_f1, augments = data_augmentation(X_circular, y_circular, svm_rbf, sd=0.3)

print(f"Mean accuracy: {np.mean(augment_accuracies):.4f}")
print(f"Mean F1-score: {np.mean(augment_f1):.4f}")


# Plot two augmentations
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, (augment_X, augment_y) in enumerate(augments):
    plot_decision_boundary(axes[i], svm_rbf, augment_X, augment_y, f"Augmented Dataset {i+1}")

axes[0].set_ylabel("$X_2$")
axes[1].tick_params(labelleft=False)

plt.tight_layout()
plt.savefig("augmented_datasets_numpy_batch.png", dpi=300)
plt.close()

print(f"Total runtime: {time.perf_counter() - start_time:.2f} seconds")
