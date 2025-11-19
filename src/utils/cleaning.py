import pandas as pd
import sys
import os

def clean_network_data(input_file, output_file=None):
    """
    Clean network flow data by standardizing direction values and removing duplicates.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (optional, defaults to 'cleaned_<input_file>')
    """
    # Generate output filename if not provided
    if output_file is None:
        # Get the directory and filename separately
        input_dir = os.path.dirname(input_file)
        input_filename = os.path.basename(input_file)
        
        # Create cleaned filename
        name, ext = os.path.splitext(input_filename)
        cleaned_filename = f"cleaned_{name}{ext}"
        
        # Join with the original directory
        if input_dir:
            output_file = os.path.join(input_dir, cleaned_filename)
        else:
            output_file = cleaned_filename
    
    print(f"Reading file: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    original_rows = len(df)
    
    # Substitute direction values
    df['Dir'] = df['Dir'].str.strip()  # Remove any whitespace
    df['Dir'] = df['Dir'].replace({
        '->': 'mono',
        '?>': 'mono',
        '<->': 'bi',
        '<?>': 'bi'
    })
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Original rows: {original_rows}")
    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Duplicates removed: {original_rows - len(df)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [output_file]")
        print("Example: python script.py data.csv")
        print("Example: python script.py /path/to/data.csv")
        print("Example: python script.py data.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        clean_network_data(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
