import pandas as pd
import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split 

# Configuration
DATASET_FILENAMES = [
    "1-42-neris-single.csv", 
    "2-43-neris-single.csv", 
    "3-44-rbot-single.csv", 
    "4-45-rbot-single.csv", 
    "5-46-virut-single.csv", 
    "6-47-donbot-single.csv", 
    "7-48-sogou-single.csv", 
    "8-49-murlo-single.csv", 
    "9-50-neris-multi.csv", 
    "10-51-rbot-multi.csv", 
    "11-52-rbot-multi.csv", 
    "12-53-nsis-multi.csv", 
    "13-54-virut-single.csv"
]

PATH = "data/CTU-datasets"
BASE_OUTPUT_DIR = "data/results_EXP1"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Split configuration
TEST_SIZE_RATIO = 0.2 
RANDOM_STATE = 42 


def setup_logger(output_folder, filename):
    """Configures a logger that writes to a specific log file for the test."""
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root loggers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    # File handler (writes to log.txt)
    file_handler = logging.FileHandler(os.path.join(output_folder, 'log.txt'), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def generate_folder_name(filename, test_number):
    """Extracts numeric prefix and merges with test number."""
    parts = []
    match = re.search(r'^(\d+-\d+)', filename)
    if match:
        parts.append(match.group(1).replace('-', '_'))

    prefix = f"test{test_number}_"
    return prefix + '_'.join(parts)


def plot_label_count(data, output_path, filename):
    """Generates and saves a bar chart of the class distribution."""
    counts = data['Label'].value_counts().sort_index()
    total_rows = len(data)
    
    botnet_count = counts.get(1, 0)
    
    labels = ['Background/Safe (0)', 'Botnet (1)']
    colors = ['skyblue', 'salmon']
    
    plt.figure(figsize=(7, 6))
    bars = plt.bar(counts.index, counts.values, color=colors)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval}', ha='center', va='bottom', fontsize=12)

    plt.title(f'Class Distribution - {filename}', fontsize=14)
    plt.xticks(counts.index, labels, rotation=0) 
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"   -> Label count plot saved: {output_path}")


def load_and_clean_data(file_path, output_folder, filename, logger):
    """Loads data, applies cleaning, logs statistics, and generates label plot."""
    data = pd.read_csv(file_path, sep=",")

    initial_shape = data.shape[0]
    initial_cols = data.columns.tolist()
    
    logger.info(f"\n--- Data Cleaning Start: {filename} ---")
    logger.info(f"Initial shape: {initial_shape} rows, {len(initial_cols)} columns")

    # Type conversion
    data['sTos'] = data['sTos'].astype('Int64')
    data['dTos'] = data['dTos'].astype('Int64')

    # Checking and fixing bad labels (Dir column)
    data['Dir'] = data['Dir'].str.strip()
    data_before_who_drop = data.shape[0]
    data = data[data['Dir'] != 'who'].copy()
    rows_dropped_who = data_before_who_drop - data.shape[0]

    # Swap IP/Port columns if direction indicates reverse flow
    mask = data['Dir'].isin(['<-', '<?'])
    data.loc[mask, ['SrcAddr', 'DstAddr']] = data.loc[mask, ['DstAddr', 'SrcAddr']].values
    data.loc[mask, ['Sport', 'Dport']] = data.loc[mask, ['Dport', 'Sport']].values
    
    data['Dir'] = data['Dir'].replace({
            '->': 'mono', '?>': 'mono', '<-': 'mono', '<?': 'mono', '<->': 'bi', '<?>': 'bi'
        })

    # Removing Duplicates
    rows_before_dup = data.shape[0]
    data.drop_duplicates(inplace=True)
    rows_dropped_dup = rows_before_dup - data.shape[0]

    # Drop useless columns
    cols_to_drop = ['sTos', 'dTos', 'StartTime']
    cols_dropped = [c for c in cols_to_drop if c in data.columns]
    data.drop(columns=cols_dropped, inplace=True, axis=1)

    # Uniform labels into botnet (1) and background (0)
    data['Label'] = data['Label'].apply(lambda x: 1 if 'botnet' in str(x).lower() else 0)
    
    final_shape = data.shape[0]
    total_rows_dropped = initial_shape - final_shape
    
    # Post-cleaning statistics
    botnet_count = data['Label'].sum()
    total_rows = data.shape[0]
    botnet_percentage = (botnet_count / total_rows) * 100 if total_rows > 0 else 0

    logger.info("\n--- Cleaning Statistics ---")
    logger.info(f"Columns removed: {', '.join(cols_dropped) if cols_dropped else 'None'}")
    logger.info(f"Rows removed total: {total_rows_dropped} (who: {rows_dropped_who}, duplicates: {rows_dropped_dup})")
    logger.info(f"Final shape: {data.shape[0]} rows, {data.shape[1]} columns")
    logger.info(f"Botnet Label (1) Percentage: {botnet_percentage:.4f}% ({botnet_count} / {total_rows})")

    print(f"   Initial shape: {initial_shape}, Final shape: {data.shape[0]}, Botnet: {botnet_percentage:.2f}%")
    
    plot_path = os.path.join(output_folder, 'label_count_distribution.png')
    plot_label_count(data, plot_path, filename)

    return data


def encode_features(X_train, X_test):
    """Applies Label Encoding and Frequency Encoding."""

    # Label Encoding for categorical columns
    cat_cols = ['Proto', 'Dir', 'State']
    for col in cat_cols:
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        codes, uniques = pd.factorize(combined)
        X_train[col] = codes[:len(X_train)]
        X_test[col] = codes[len(X_train):]

    # Frequency Encoding for high-cardinality columns
    freq_cols = ['SrcAddr', 'DstAddr', 'Sport', 'Dport']
    for col in freq_cols:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq_map)
        # Fill unseen values in test with 0
        X_test[col] = X_test[col].map(freq_map).fillna(0)

    # Ensure numeric types
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    return X_train, X_test


# Main Execution Loop
print("Starting process for individual datasets with internal split.")

for i, filename in enumerate(DATASET_FILENAMES):
    test_number = i
    
    output_folder_name = generate_folder_name(filename, test_number)
    output_folder = os.path.join(BASE_OUTPUT_DIR, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Logger setup
    logger = setup_logger(output_folder, filename)
    logger.info(f"### Start Test {test_number} on Dataset: {filename} ###")
    
    print(f"\nTest {test_number}: Dataset {filename}")
    print("---")

    full_path = os.path.join(PATH, filename)
    
    print(f"1. Loading and cleaning data for {filename}...")
    full_data = load_and_clean_data(full_path, output_folder, filename, logger)
    
    Y = full_data['Label']
    X = full_data.drop(columns=['Label'])

    print(f"2. Splitting into Train ({100-TEST_SIZE_RATIO*100:.0f}%) and Test ({TEST_SIZE_RATIO*100:.0f}%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, 
        test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE,
        stratify=Y
    )
    
    print(f"   Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    print("3. Applying Encoding...")
    X_train_encoded, X_test_encoded = encode_features(X_train.copy(), X_test.copy()) 

    logger.info("\n--- Split/Encoding Statistics ---")
    logger.info(f"Train Shape (encoded): ({X_train_encoded.shape[0]}, {X_train_encoded.shape[1]})")
    logger.info(f"Test Shape (encoded): ({X_test_encoded.shape[0]}, {X_test_encoded.shape[1]})")

    print("4. Training Random Forest...")
    rf_model = RandomForestClassifier(
        criterion='gini',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        rf_model.fit(X_train_encoded, y_train)
    except Exception as e:
        print(f"ERROR during model training: {e}. Skipping test.")
        logger.error(f"\nERROR during model training: {e}")
        logging.shutdown()
        continue

    # Evaluation
    y_pred = rf_model.predict(X_test_encoded)
    avg_accuracy = rf_model.score(X_test_encoded, y_test)

    print(f"5. Results saved in: {output_folder}/")
    print(f"   Average Accuracy: {avg_accuracy:.4f}")

    logger.info("\n--- Model Results ---")
    logger.info(f"Accuracy (Test Set): {avg_accuracy:.4f}")

    # Save Classification Report
    report_path = os.path.join(output_folder, 'report.txt')
    target_names = ['Background/Safe', 'Botnet']

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=False
    )

    with open(report_path, 'w') as f:
        f.write(f"Processed Dataset: {filename}\n")
        f.write(f"Split ratio: Train {1-TEST_SIZE_RATIO:.2f} / Test {TEST_SIZE_RATIO:.2f}\n")
        f.write(f"Accuracy: {avg_accuracy:.4f}\n\n")
        f.write(report)

    # Save Confusion Matrix
    cm_path = os.path.join(output_folder, 'confMatrix.png')
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(values_format='d', cmap=plt.cm.Blues) 
    plt.title(f"Confusion Matrix (Test {test_number})")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close() 

    # Save Feature Importance Plot
    fi_path = os.path.join(output_folder, 'featureImportance.png')
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances (Test {test_number})")
    plt.bar(range(X_train_encoded.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train_encoded.shape[1]), X_train_encoded.columns[indices], rotation=45, ha='right')
    plt.ylabel('Importance (Gini)')
    plt.tight_layout()
    plt.savefig(fi_path, bbox_inches='tight')
    plt.close() 

    # Release file handlers
    logging.shutdown()

print("\nProcess completed for all individual datasets with internal split.")