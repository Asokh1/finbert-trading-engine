import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_massive_data():
    print("Downloading the modernized Parquet dataset...")
    
    # Using a natively secure dataset (Parquet format) to bypass the script ban
    # Contains nearly 12,000 financial headlines/tweets
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    df = pd.DataFrame(dataset['train'])
    
    df = df[df['label'] != 2]
    
    train, test = train_test_split(df[['text', 'label']], test_size=0.2, random_state=42)
    
    train.to_json('data/train.json', orient='records', lines=True)
    test.to_json('data/test.json', orient='records', lines=True)
    
    print(f"\nSUCCESS! Your AI will now train on real data.")
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    print(f"Label distribution - Positive: {(df['label']==1).sum()}, Negative: {(df['label']==0).sum()}")

if __name__ == '__main__':
    prepare_massive_data()