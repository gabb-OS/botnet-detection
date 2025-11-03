import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuration
# Choose Model: 'DT', 'RF', 'SVM' or 'MLP'
MODEL_TYPE = 'DT' 

# SVM Specific Configuration
SVM_MAX_TRAIN_SAMPLES = 20000 
ENABLE_PCA_SVM = True   # True => Enable PCA on SVM

# Global Random State for reproducibility
RANDOM_STATE = 42

# Dataset pairs to test
DATASET_PAIRS = [
    # TRAINIP/TESTIP    | BOTNET    | TYPEofATTACK
    # single/single     | same      | same
    ['1-42-neris-single.csv', '2-43-neris-single.csv'],

    # single/single     | same      | different
    ['3-44-rbot-single.csv', '4-45-rbot-single.csv'],

    # single/single     | different | same
    ['5-46-virut-single.csv', '6-47-donbot-single.csv'],

    # single/single     | different | different
    ['1-42-neris-single.csv', '8-49-murlo-single.csv'],

    # single/multi      | different | common
    ['1-42-neris-single.csv', '10-51-rbot-multi.csv'],

    # multi/single      | same      | different
    ['10-51-rbot-multi.csv', '4-45-rbot-single.csv'],

    # multi/single      | different | different
    ['10-51-rbot-multi.csv', '6-47-donbot-single.csv'],

    # multi/multi       | same      | same
    ['10-51-rbot-multi.csv', '11-52-rbot-multi.csv'],

    # multi/multi       | different | common
    ['9-50-neris-multi.csv', '13-54-virut-single.csv']
]

path = "data/CTU-datasets"
base_output_dir = f"data/results_EXP2_{MODEL_TYPE.lower()}"
os.makedirs(base_output_dir, exist_ok=True)


# Helper Functions

def generate_folder_name(filenames, test_number):
    """Generates a folder name based on dataset prefixes and test number."""
    parts = []
    for filename in filenames:
        match = re.search(r'^(\d+-\d+)', filename)
        if match:
            parts.append(match.group(1).replace('-', '_'))
    prefix = f"test{test_number}_"
    return prefix + '_'.join(parts)


def load_and_clean_data(file_path):
    """Loads data, cleans string columns, handles IP/Port direction swapping, and removes duplicates."""
    data = pd.read_csv(file_path, sep=",")
    data['sTos'] = data['sTos'].astype('Int64')
    data['dTos'] = data['dTos'].astype('Int64')

    initial_shape = data.shape[0]

    # Clean 'Dir' column
    data['Dir'] = data['Dir'].str.strip()
    data = data[data['Dir'] != 'who'].copy()

    # Swap IP/Port columns if direction indicates reverse flow
    mask = data['Dir'].isin(['<-', '<?'])
    data.loc[mask, ['SrcAddr', 'DstAddr']] = data.loc[mask, ['DstAddr', 'SrcAddr']].values
    data.loc[mask, ['Sport', 'Dport']] = data.loc[mask, ['Dport', 'Sport']].values

    # Normalize 'Dir' tokens
    data['Dir'] = data['Dir'].replace({
            '->': 'mono', '?>': 'mono', '<-': 'mono', '<?': 'mono',
            '<->': 'bi', '<?>': 'bi'
        })

    # Remove duplicates and useless columns
    data.drop_duplicates(inplace=True)
    cols_to_drop = ['sTos', 'dTos', 'StartTime']
    data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True, axis=1)

    # Label generation: 1 if botnet, 0 otherwise
    data['Label'] = data['Label'].apply(lambda x: 1 if 'botnet' in str(x).lower() else 0)

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
        # Handle unseen values in test set by filling with 0
        X_test[col] = X_test[col].map(freq_map).fillna(0)

    # Ensure numeric types
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    return X_train, X_test


def drop_nans_aligned(X, y):
    """Drops rows with NaN values while keeping X and y aligned."""
    combined = pd.concat([X, y], axis=1).dropna()
    return combined.drop(columns=['Label']), combined['Label']


def balance_training_data(X, y, max_samples=None):
    """
    Downsamples the majority class (Background) to match the minority class (Botnet).
    If max_samples is set (for SVM), it ensures the total dataset size does not exceed it
    while maintaining a 50/50 split.
    """
    # Combine X and y so that sampling affects both simultaneously
    train_data = pd.concat([X, y], axis=1)
    
    botnets = train_data[train_data['Label'] == 1]
    background = train_data[train_data['Label'] == 0]
    
    n_botnets = len(botnets)
    n_background = len(background)
    
    print(f"   Original Distribution -> Background: {n_background}, Botnets: {n_botnets}")
    
    if n_botnets == 0:
        print("   WARNING: No Botnets in training data! Returning sampled background only.")
        if max_samples and len(train_data) > max_samples:
            train_data = train_data.sample(n=max_samples, random_state=RANDOM_STATE)
        return train_data.drop(columns=['Label']), train_data['Label']

    n_target_per_class = n_botnets

    # If a limit is enforcing (e.g. SVM max 20k), we can't exceed max_samples / 2 per class
    if max_samples is not None:
        limit_per_class = max_samples // 2
        if n_target_per_class > limit_per_class:
            print(f"   [SVM Limit] Reducing botnets from {n_target_per_class} to {limit_per_class} to fit max_samples.")
            n_target_per_class = limit_per_class

    # Sample both classes to the target size
    botnets_sampled = botnets.sample(n=n_target_per_class, random_state=RANDOM_STATE)
    background_sampled = background.sample(n=n_target_per_class, random_state=RANDOM_STATE)
    
    # Recombine and Shuffle
    balanced_data = pd.concat([botnets_sampled, background_sampled])
    balanced_data = balanced_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"   Final Training Size: {len(balanced_data)} (Botnets: {len(botnets_sampled)}, Background: {len(background_sampled)})")
    
    # Split back into X and y
    return balanced_data.drop(columns=['Label']), balanced_data['Label']


# Main Execution Loop

print(f"Starting Process. Selected Model: {MODEL_TYPE}")

for i, filenames in enumerate(DATASET_PAIRS):
    test_number = i
    train_file, test_file = filenames
    print(f"\nTest {test_number}: {train_file} (Train) vs {test_file} (Test)")
    print("---")

    # Load and Clean 
    full_train_path = os.path.join(path, train_file)
    full_test_path = os.path.join(path, test_file)
    
    print("1. Loading and cleaning data...")
    data_train = load_and_clean_data(full_train_path)
    data_test = load_and_clean_data(full_test_path)

    y_train_raw = data_train['Label']
    X_train_raw = data_train.drop(columns=['Label'])
    y_test_raw = data_test['Label']
    X_test_raw = data_test.drop(columns=['Label'])

    # Encode 
    print("2. Applying feature encoding...")
    X_train_enc, X_test_enc = encode_features(X_train_raw, X_test_raw)

    # Model-Specific Preprocessing & Initialization 
    model = None
    X_train_final, y_train_final = None, None
    X_test_final, y_test_final = None, None
    
    # To track PCA components
    n_pca_components = None 

    if MODEL_TYPE == 'MLP':
        print("3. Preprocessing for MLP (Drop NaNs, Balance, Scale)...")
        # Drop NaNs
        X_train_clean, y_train_clean = drop_nans_aligned(X_train_enc, y_train_raw)
        X_test_clean, y_test_clean = drop_nans_aligned(X_test_enc, y_test_raw)
        
        # Balance Training Data
        X_train_bal, y_train_bal = balance_training_data(X_train_clean, y_train_clean)
        
        # Scale Data
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_bal)
        X_test_final = scaler.transform(X_test_clean)
        y_train_final = y_train_bal
        y_test_final = y_test_clean

        model = MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            activation='relu',
            solver='adam',
            max_iter=1000,
            learning_rate_init=0.001,
            random_state=RANDOM_STATE,
            early_stopping=True
        )
        
    elif MODEL_TYPE == 'SVM':
        print(f"3. Preprocessing for SVM (Drop NaNs, Balance, Limit <= {SVM_MAX_TRAIN_SAMPLES}, Scale)...")
        # Drop NaNs
        X_train_clean, y_train_clean = drop_nans_aligned(X_train_enc, y_train_raw)
        X_test_clean, y_test_clean = drop_nans_aligned(X_test_enc, y_test_raw)
        
        # Balance Training Data
        X_train_bal, y_train_bal = balance_training_data(
            X_train_clean, 
            y_train_clean, 
            max_samples=SVM_MAX_TRAIN_SAMPLES
        )
        
        # Scale Data
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_bal)
        X_test_final = scaler.transform(X_test_clean)
        
        # Apply PCA
        if ENABLE_PCA_SVM:
            print("   Applying PCA (0.95 variance)...")
            pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
            X_train_final = pca.fit_transform(X_train_final)
            X_test_final = pca.transform(X_test_final)
            n_pca_components = X_train_final.shape[1]
            print(f"   PCA reduced features to: {n_pca_components}")

        y_train_final = y_train_bal
        y_test_final = y_test_clean 

        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=RANDOM_STATE,
            cache_size=2000
        )

    elif MODEL_TYPE in ['RF', 'DT']: 
        print(f"3. Preprocessing for {MODEL_TYPE} (None required)...")
        
        X_train_final = X_train_enc
        y_train_final = y_train_raw
        X_test_final = X_test_enc
        y_test_final = y_test_raw

        if MODEL_TYPE == 'RF':
            model = RandomForestClassifier(
                criterion='gini',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        elif MODEL_TYPE == 'DT':
            model = DecisionTreeClassifier(
                criterion='gini',
                random_state=RANDOM_STATE
            )
    
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose 'DT', 'RF', 'SVM' or 'MLP'.")

    # Training 
    print(f"4. Training {MODEL_TYPE} model...")
    try:
        model.fit(X_train_final, y_train_final)
    except Exception as e:
        print(f"ERROR during training: {e}. Skipping test.")
        continue

    # Evaluation 
    y_pred = model.predict(X_test_final)
    avg_accuracy = model.score(X_test_final, y_test_final)

    output_folder = os.path.join(base_output_dir, generate_folder_name(filenames, test_number))
    os.makedirs(output_folder, exist_ok=True)
    print(f"5. Results saved in: {output_folder}/")
    print(f"   Accuracy: {avg_accuracy:.4f}")

    # Save Report
    with open(os.path.join(output_folder, 'report.txt'), 'w') as f:
        f.write(f"Model: {MODEL_TYPE}\n")
        f.write(f"Dataset Train: {train_file}\nDataset Test: {test_file}\n")
        if MODEL_TYPE == 'SVM':
            f.write(f"Training Samples (Limited): {len(X_train_final)}\n")
            if ENABLE_PCA_SVM and n_pca_components:
                 f.write(f"PCA Enabled: True (Components: {n_pca_components})\n")
        f.write(f"Accuracy: {avg_accuracy:.4f}\n\n")
        f.write(classification_report(y_test_final, y_pred, target_names=['Background', 'Botnet']))

    # Save Confusion Matrix
    cm = confusion_matrix(y_test_final, y_pred, labels=model.classes_)
    plt.figure(figsize=(6, 6))
    
    ConfusionMatrixDisplay(cm, display_labels=['Background', 'Botnet']).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({MODEL_TYPE} - Test {test_number})")
    plt.savefig(os.path.join(output_folder, 'confMatrix.png'), bbox_inches='tight')
    plt.close()

    # RF/DT-Specific Plotting (Feature Importance) 
    if MODEL_TYPE in ['RF', 'DT']:
        print(f"6. Generating Feature Importance and Distribution plots ({MODEL_TYPE})...")
        
        # Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances (Test {test_number})")
        plt.bar(range(X_train_final.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train_final.shape[1]), X_train_final.columns[indices], rotation=45, ha='right')
        plt.ylabel('Importance (Gini)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'featureImportance.png'), bbox_inches='tight')
        plt.close()

        # KDE Distribution Plots for Top 3 Features
        top_3_indices = indices[:3]
        top_3_names = X_train_raw.columns[top_3_indices]

        for i, feature_name in enumerate(top_3_names):
            safe_name = feature_name.replace('/', '_').replace(' ', '_')
            plt.figure(figsize=(8, 5))
            sns.kdeplot(data=X_train_raw, x=feature_name, label='Train Data', color='blue', fill=True, alpha=0.3)
            sns.kdeplot(data=X_test_raw, x=feature_name, label='Test Data', color='orange', fill=True, alpha=0.3)
            plt.title(f"Distribution: {feature_name} (Rank #{i+1})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'dist_top{i+1}_{safe_name}.png'), bbox_inches='tight')
            plt.close()

print("\nProcess completed for all dataset pairs.")