import pandas as pd

# Read the CSV file
df = pd.read_csv('capture20110810.csv')

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
df.to_csv('cleaned_file.csv', index=False)

print(f"Original rows: {len(pd.read_csv('capture20110810.csv'))}")
print(f"Rows after removing duplicates: {len(df)}")
print(f"Duplicates removed: {len(pd.read_csv('capture20110810.csv')) - len(df)}")
