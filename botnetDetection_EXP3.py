import pandas as pd
import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Configuration
DATASET_GROUPS = [
    # ss
    ['1-42-neris-single.csv', '3-44-rbot-single.csv'],
    # sss
    ['1-42-neris-single.csv', '3-44-rbot-single.csv', '5-46-virut-single.csv'],
    # ssss w/double neris
    ['1-42-neris-single.csv', '2-43-neris-single.csv', '3-44-rbot-single.csv', '5-46-virut-single.csv'],
    # sssss
    ['1-42-neris-single.csv', '3-44-rbot-single.csv', '5-46-virut-single.csv', '7-48-sogou-single.csv', '8-49-murlo-single.csv'],
    # mm
    ['10-51-rbot-multi.csv', '12-53-nsis-multi.csv'],
    # mmm
    ['9-50-neris-multi.csv', '10-51-rbot-multi.csv', '12-53-nsis-multi.csv'],
    # mmmm
    ['9-50-neris-multi.csv', '10-51-rbot-multi.csv', '11-52-rbot-multi.csv', '12-53-nsis-multi.csv'],
    # ssmm
    ['1-42-neris-single.csv', '13-54-virut-single.csv', '10-51-rbot-multi.csv', '12-53-nsis-multi.csv'],
    # ssmm w/double rbot
    ['1-42-neris-single.csv', '3-44-rbot-single.csv', '10-51-rbot-multi.csv', '12-53-nsis-multi.csv'],
]

PATH = "data/CTU-datasets"
BASE_OUTPUT_DIR = "data/results_EXP3"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Split configuration
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42


def setup_logger(output_folder, test_number):
    """Configures a logger that writes to a specific log file for the test."""
    logger_name = f"test_{test_number}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root loggers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_folder, 'log.txt'), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def generate_folder_name(filenames, test_number):
    """Extracts numeric prefixes and merges them for the folder name."""
    parts = []
    for filename in filenames:
        match = re.search(r'^(\d+-\d+)', filename)
        if match:
            parts.append(match.group(1).replace('-', '_'))

    prefix = f"test{test_number}_"
    return prefix + '_'.join(parts)


def load_and_merge_datasets(filenames):
    """
    Loads multiple datasets and assigns progressive labels.
    0 = Background/Normal
    1 = Botnet from first file
    2 = Botnet from second file, etc.
    """
    df_list = []
    class_names = ['Background'] 
    
    for index, filename in enumerate(filenames):
        file_path = os.path.join(PATH, filename)
        data = pd.read_csv(file_path, sep=",")
        
        # Extract botnet name for reporting
        parts = filename.split('-')
        botnet_name = parts[2] if len(parts) > 2 else f"Botnet_{index+1}"
        class_names.append(botnet_name)

        # Multi-Class Logic:
        # If 'botnet' -> index + 1
        # Else (Background) -> 0
        data['Label'] = data['Label'].apply(lambda x: index + 1 if 'botnet' in str(x).lower() else 0)

        df_list.append(data)

    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df, class_names


def clean_data(data, logger=None):
    """Applies data cleaning: fixes Dir column, swaps IP/Ports, removes duplicates."""
    initial_shape = data.shape[0]

    # Type conversion
    data['sTos'] = data['sTos'].astype('Int64')
    data['dTos'] = data['dTos'].astype('Int64')

    # Fix 'Dir' column
    data['Dir'] = data['Dir'].str.strip()
    data = data[data['Dir'] != 'who'].copy()

    # Swap IP/Port columns if direction indicates reverse flow
    mask = data['Dir'].isin(['<-', '<?'])
    data.loc[mask, ['SrcAddr', 'DstAddr']] = data.loc[mask, ['DstAddr', 'SrcAddr']].values
    data.loc[mask, ['Sport', 'Dport']] = data.loc[mask, ['Dport', 'Sport']].values

    # Normalize tokens
    data['Dir'] = data['Dir'].replace({
            '->': 'mono', '?>': 'mono', '<-': 'mono', '<?': 'mono',
            '<->': 'bi', '<?>': 'bi'
        })

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Drop useless columns
    cols_to_drop = ['sTos', 'dTos', 'StartTime']
    data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True, axis=1)

    if logger:
        logger.info(f"Initial shape: {initial_shape}, Final shape: {data.shape[0]}")
    
    print(f"   Initial shape: {initial_shape}, Final shape: {data.shape[0]}")
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
        X_test[col] = X_test[col].map(freq_map).fillna(0)

    # Ensure numeric types
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    return X_train, X_test


# Main Execution Loop
print("Starting process for multi-class dataset pairs.")

for i, filenames in enumerate(DATASET_GROUPS):
    test_number = i
    
    output_folder_name = generate_folder_name(filenames, test_number)
    output_folder = os.path.join(BASE_OUTPUT_DIR, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Logger setup
    logger = setup_logger(output_folder, test_number)
    logger.info(f"### Start Test {test_number} ###")
    logger.info(f"Files: {', '.join(filenames)}")

    print(f"\nTest {test_number}: {filenames}")
    print("---")
    
    print("1. Loading and merging datasets...")
    data, current_target_names = load_and_merge_datasets(filenames)

    print("2. Cleaning data...")
    data = clean_data(data, logger)

    X = data.drop(columns=['Label'])
    y = data['Label']
    
    unique_classes = sorted(y.unique())
    logger.info(f"Classes present: {unique_classes}")
    logger.info(f"Mapped class names: {current_target_names}")
    
    print(f"   Classes present: {unique_classes}")
    print(f"   Mapped names: {current_target_names}")

    print("3. Splitting Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    print("4. Applying Encoding...")
    X_train_encoded, X_test_encoded = encode_features(X_train.copy(), X_test.copy())

    print("5. Training Random Forest...")
    rf_model = RandomForestClassifier(
        criterion='gini',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    try:
        rf_model.fit(X_train_encoded, y_train)
    except Exception as e:
        print(f"ERROR during training: {e}. Skipping test.")
        logger.error(f"ERROR during training: {e}")
        logging.shutdown()
        continue

    # Evaluation
    y_pred = rf_model.predict(X_test_encoded)
    avg_accuracy = rf_model.score(X_test_encoded, y_test)

    print(f"6. Results saved in: {output_folder}/")
    print(f"   Average Accuracy: {avg_accuracy:.4f}")

    logger.info("\n--- Model Results ---")
    logger.info(f"Accuracy: {avg_accuracy:.4f}")

    # Save Classification Report
    report_path = os.path.join(output_folder, 'report.txt')

    # Use the dynamic class names extracted during loading
    report = classification_report(
        y_test,
        y_pred,
        target_names=current_target_names, 
        output_dict=False
    )

    with open(report_path, 'w') as f:
        f.write(f"Input Files: {', '.join(filenames)}\n")
        f.write(f"Accuracy: {avg_accuracy:.4f}\n\n")
        f.write(report)

    # Save Confusion Matrix
    cm_path = os.path.join(output_folder, 'confMatrix.png')
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_target_names)
    disp.plot(values_format='d', cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix (Test {test_number})")
    plt.tight_layout()
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

print("\nProcess completed for all multi-class dataset groups.")