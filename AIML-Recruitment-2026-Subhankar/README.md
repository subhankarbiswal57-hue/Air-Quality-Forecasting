# AIML Recruitment 2026 — Subhankar

## Candidate Details
- **Name:** Subhankar
- **Institution:** SRM Institute of Science and Technology, Kattankulathur
- **Program:** B.Tech CSE (AI & ML), 2nd Year

## Tasks Completed
- **Task 1: Air Quality Forecasting**
- **Task 2: Neural Network (MNIST Digit Classification)**

## Problem Statement
- **Task 1**: Analyse historical air-quality sensor data and build a model to predict a future air-quality measurement (next-hour pollutant concentration).
- **Task 2**: Build and train a simple neural network to classify handwritten digits (0–9) from the MNIST dataset, understand the role of each network component (layers, activations), and analyse how changing a hyperparameter affects performance.

## Approach

### Task 1 — Air Quality Forecasting
1. Loaded the UCI Air Quality dataset, parsed Date/Time into a datetime index, and replaced the dataset's `-200` missing-value marker with NaN.
2. Dropped near-empty columns (>50% missing) and interpolated remaining gaps using time-based interpolation to preserve the time-series trend.
3. Explored pollutant distributions, daily/hourly patterns, and correlations between variables.
4. Engineered lag features, rolling averages, and time-of-day/week/month features (all computed using `.shift()` to avoid leakage).
5. Split data chronologically (80/20, no shuffling) and trained Linear Regression and Random Forest models to predict the next hour's pollutant reading.
6. Evaluated with MAE/MSE/RMSE/R², analysed feature importance, and discussed data leakage risks specific to time-series forecasting.

### Task 2 — Neural Network (MNIST)
1. Loaded and inspected the MNIST dataset (60,000 train / 10,000 test images, 28×28 grayscale).
2. Normalized pixel values to [0, 1] and flattened images to 784-length vectors.
3. Built a simple feed-forward neural network: Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax).
4. Trained the model using the Adam optimizer and sparse categorical crossentropy loss, tracking training/validation loss and accuracy.
5. Evaluated the model on the test set using accuracy, a confusion matrix, and a classification report (precision/recall/F1).
6. Ran an experiment by changing the hidden layer size and compared results against the baseline model.

## Technologies Used
- Python
- pandas, NumPy
- TensorFlow / Keras
- scikit-learn
- Matplotlib, Seaborn

## Results

### Task 1 — Air Quality Forecasting
- See `results/task1_metrics.json` for full MAE/MSE/RMSE/R² for both Linear Regression and Random Forest.
- Random Forest outperformed Linear Regression, indicating non-linear relationships between engineered features and the target.
- Most recent lag features were the strongest predictors (see `results/task1_feature_importance.png`).

### Task 2 — MNIST Neural Network
- Baseline model test accuracy: **97.71%** (see `results/metrics.json` and notebook output for full details).
- Confusion matrix shows most confusion between visually similar digits (e.g. 4/9, 3/5).
- Experiment (modified hidden layer size) results and comparison are shown in the notebook and `results/experiment_comparison.png`.

## Key Learnings
1. Normalizing input data significantly stabilizes and speeds up neural network training.
2. ReLU avoids the vanishing gradient problem common with sigmoid/tanh in deeper networks, while Softmax is well suited for multi-class output since it produces a proper probability distribution over classes.
3. Model capacity (number of neurons/layers) directly trades off between underfitting and overfitting — more neurons improved fit but increased overfitting risk on this simple dataset.
4. In time-series forecasting, features must be constructed using only past values (via `.shift()`) — otherwise it's easy to accidentally leak future information into training.
5. A chronological train/test split (no shuffling) is essential for time-series problems, unlike typical random splits used for i.i.d. data.

## Challenges
- **Challenge (Task 2):** Deciding how to evaluate whether the model was overfitting or underfitting.
  **Solution:** Plotted training vs validation loss/accuracy curves across epochs — a widening gap between training and validation curves confirmed overfitting, which guided the choice of hyperparameter to experiment with.
- **Challenge (Task 1):** Handling a large fraction of missing/invalid sensor readings (marked as -200) without breaking the time-series structure.
  **Solution:** Dropped columns that were mostly missing, and used time-based interpolation (rather than mean-fill or row-dropping) for the rest, preserving temporal continuity.

## Repository Structure
```
AIML-Recruitment-2026-Subhankar/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── task1_air_quality_forecasting.ipynb
│   └── task2_mnist_neural_network.ipynb
├── results/
│   ├── task1_distributions.png
│   ├── task1_timeseries.png
│   ├── task1_hourly_pattern.png
│   ├── task1_correlation_heatmap.png
│   ├── task1_actual_vs_predicted.png
│   ├── task1_feature_importance.png
│   ├── task1_metrics.json
│   ├── task1_cleaned_air_quality.csv
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── experiment_comparison.png
│   └── metrics.json
└── models/
    └── mnist_model.h5
```
