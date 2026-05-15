import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data(filepath):
    """
    Load dataset from CSV file.
    """

    try:
        df = pd.read_csv(filepath)

        logging.info("Dataset loaded successfully.")
        logging.info(f"Dataset Shape: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


# =========================================================
# SPLIT FEATURES AND TARGET
# =========================================================

def split_data(df, target_column='loan_status'):
    """
    Split dataset into train and test sets.
    """

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        logging.info("Train-test split completed.")

        logging.info(f"X_train Shape: {X_train.shape}")
        logging.info(f"X_test Shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        raise


# =========================================================
# APPLY SMOTE
# =========================================================

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance classes.
    """

    try:
        logging.info("Applying SMOTE...")

        smote = SMOTE(random_state=42)

        X_train_smote, y_train_smote = smote.fit_resample(
            X_train,
            y_train
        )

        logging.info("SMOTE applied successfully.")

        logging.info("Class Distribution After SMOTE:")
        logging.info(f"\n{pd.Series(y_train_smote).value_counts()}")

        return X_train_smote, y_train_smote

    except Exception as e:
        logging.error(f"Error applying SMOTE: {e}")
        raise


# =========================================================
# COMPUTE CLASS WEIGHTS
# =========================================================

def compute_class_weights_func(y_train):
    """
    Compute class weights for imbalanced classification.
    """

    try:
        classes = np.unique(y_train)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )

        class_weights = dict(zip(classes, weights))

        logging.info("Class weights computed successfully.")
        logging.info(f"Class Weights: {class_weights}")

        return class_weights

    except Exception as e:
        logging.error(f"Error computing class weights: {e}")
        raise


# =========================================================
# SAVE BALANCED DATASET
# =========================================================

def save_balanced_data(X_train_smote, y_train_smote, output_path):
    """
    Save balanced dataset to CSV file.
    """

    try:
        balanced_df = pd.concat(
            [
                pd.DataFrame(X_train_smote),
                pd.DataFrame(y_train_smote, columns=['loan_status'])
            ],
            axis=1
        )

        balanced_df.to_csv(output_path, index=False)

        logging.info(f"Balanced dataset saved to: {output_path}")

    except Exception as e:
        logging.error(f"Error saving balanced dataset: {e}")
        raise


# =========================================================
# MAIN PIPELINE
# =========================================================

def imbalance_pipeline():
    """
    Complete imbalance handling pipeline.
    """

    try:

        # -------------------------------------------------
        # LOAD DATA
        # -------------------------------------------------

        df = load_data(
            "data/processed/final_feature_engineered_data.csv"
        )

        # -------------------------------------------------
        # SPLIT DATA
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = split_data(df)

        # -------------------------------------------------
        # ORIGINAL CLASS DISTRIBUTION
        # -------------------------------------------------

        logging.info("Original Class Distribution:")
        logging.info(f"\n{y_train.value_counts()}")

        # -------------------------------------------------
        # APPLY SMOTE
        # -------------------------------------------------

        X_train_smote, y_train_smote = apply_smote(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # COMPUTE CLASS WEIGHTS
        # -------------------------------------------------

        class_weights = compute_class_weights_func(y_train)

        # -------------------------------------------------
        # SAVE BALANCED DATA
        # -------------------------------------------------

        save_balanced_data(
            X_train_smote,
            y_train_smote,
           "data/processed/balanced_train_data.csv"
        )

        logging.info("Imbalance handling pipeline completed.")

        return (
            X_train_smote,
            X_test,
            y_train_smote,
            y_test,
            class_weights
        )

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


# =========================================================
# RUN PIPELINE
# =========================================================

if __name__ == "__main__":

    (
        X_train_balanced,
        X_test,
        y_train_balanced,
        y_test,
        class_weights
    ) = imbalance_pipeline()

    print("\nPipeline Executed Successfully.")