"""
Project Configuration File
Stores paths, constants, and tunable parameters for model training and data handling
"""

# === File Paths ===
RAW_TRAIN_PATH = "dataset/PM_train.txt"
RAW_TEST_PATH = "dataset/PM_test.txt"
TRUTH_PATH = "dataset/PM_truth.txt"

TRAIN_WITH_RUL = "dataset/df_train_with_rul.csv"
TRAIN_SELECTED = "dataset/df_train_selected.csv"
TEST_SELECTED = "dataset/df_test_selected.csv"

MODEL_OUTPUT_PATH = "outputs/rf_rul_model.joblib"

# === Feature Selection ===
TOP_N_FEATURES = 20

# === Preprocessing ===
SCALING_METHOD = "minmax"  # options: 'minmax', 'standard'

# === Model Hyperparameters (for baseline RF) ===
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}
