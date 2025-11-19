import pandas as pd
import sys

def check_column_values(input_file, columns=['sTos', 'dTos']):
    """
    Check if values in specified columns are only 0.0 or NaN.
    
    Args:
        input_file: Path to the input CSV file
        columns: List of column names to check
    """
    print(f"Reading file: {input_file}\n")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df)}\n")
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in the dataset")
            continue
        
        print(f"=== Analysis for column: {col} ===")
        
        # Count NaN values
        nan_count = df[col].isna().sum()
        
        # Count 0.0 values (excluding NaN)
        zero_count = (df[col] == 0.0).sum()
        
        # Get unique values (excluding NaN)
        unique_values = df[col].dropna().unique()
        
        # Count non-zero and non-NaN values
        non_zero_non_nan = df[(df[col] != 0.0) & (df[col].notna())]
        
        print(f"NaN values: {nan_count} ({nan_count/len(df)*100:.2f}%)")
        print(f"Zero (0.0) values: {zero_count} ({zero_count/len(df)*100:.2f}%)")
        print(f"Total zeros + NaN: {nan_count + zero_count} ({(nan_count + zero_count)/len(df)*100:.2f}%)")
        print(f"Non-zero, non-NaN values: {len(non_zero_non_nan)} ({len(non_zero_non_nan)/len(df)*100:.2f}%)")
        
        print(f"\nUnique values (excluding NaN): {len(unique_values)}")
        print(f"Unique values: {sorted(unique_values)}")
        
        if len(non_zero_non_nan) > 0:
            print(f"\nSample non-zero values:")
            print(non_zero_non_nan[col].head(10).tolist())
        
        # Final verdict
        if len(unique_values) == 1 and unique_values[0] == 0.0:
            print(f"\n✓ Column '{col}' contains ONLY 0.0 and/or NaN values")
        else:
            print(f"\n✗ Column '{col}' contains OTHER values besides 0.0 and NaN")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_columns.py <input_file>")
        print("Example: python check_columns.py data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        check_column_values(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
