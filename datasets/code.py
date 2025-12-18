import pandas as pd
import os

# List of CSV files (adjust paths if needed, assuming they are in the current directory)
files = [
    'True.csv',
    'Fake.csv',
    'fake_or_real_news.csv',
    'gossipcop_fake.csv',
    'gossipcop_real.csv',
    'politifact_fake_real_news_dataset.csv'
]

# Function to normalize labels to 'real' or 'fake'
def normalize_label(label):
    if pd.isna(label):
        return None
    label_str = str(label).strip().lower()
    if label_str in ['real', 'true', '1']:
        return 'real'
    if label_str in ['fake', 'false', '0', 'faux']:
        return 'fake'
    return None  # Unknown labels will be dropped later

# List to hold processed DataFrames
dfs = []

# Process each file
for file in files:
    if not os.path.exists(file):
        print(f"Warning: File {file} not found. Skipping.")
        continue
    
    df = pd.read_csv(file)
    
    # Extract title and text based on available columns
    if 'title' in df.columns and 'text' in df.columns:
        title_col = 'title'
        text_col = 'text'
    elif 'title' in df.columns and 'full_text' in df.columns:
        title_col = 'title'
        text_col = 'full_text'
    elif 'claim' in df.columns:
        # For politifact, use claim as title and full_text as text
        title_col = 'claim'
        text_col = 'full_text'
    else:
        print(f"Warning: {file} does not have recognizable title/text columns. Skipping.")
        continue
    
    # Extract label
    if 'label' in df.columns:
        df['normalized_label'] = df['label'].apply(normalize_label)
    else:
        print(f"Warning: {file} has no 'label' column. Skipping.")
        continue
    
    # Select and rename columns
    processed_df = df[[title_col, text_col, 'normalized_label']].rename(
        columns={title_col: 'title', text_col: 'text', 'normalized_label': 'label'}
    )
    
    # Optional: add a source column for traceability
    processed_df['source_file'] = file
    
    dfs.append(processed_df)

# Concatenate all processed DataFrames
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Drop rows with invalid/unknown labels
    merged_df = merged_df.dropna(subset=['label'])
    
    # Basic preprocessing on text and title
    def clean_text(t):
        if pd.isna(t):
            return ""
        return str(t).strip()
    
    merged_df['title'] = merged_df['title'].apply(clean_text)
    merged_df['text'] = merged_df['text'].apply(clean_text)
    
    # Combine title and text into a single content column (common for fake news detection)
    merged_df['content'] = merged_df['title'] + " " + merged_df['text']
    merged_df['content'] = merged_df['content'].apply(clean_text)
    
    # Keep relevant columns
    final_df = merged_df[['title', 'text', 'content', 'label', 'source_file']]
    
    # Save to new CSV
    final_df.to_csv('processed_data.csv', index=False)
    print("Preprocessing complete. Merged dataset saved as 'processed_data.csv'.")
    print(f"Total samples: {len(final_df)} ({final_df['label'].value_counts().to_dict()})")
else:
    print("No data processed. Check file paths and structures.")