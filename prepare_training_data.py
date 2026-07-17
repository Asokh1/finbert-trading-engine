import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_massive_data():
    print("Downloading the modernized Parquet dataset...")
    
    # Using a natively secure dataset (Parquet format) to bypass the script ban
    # Contains nearly 12,000 financial headlines/tweets
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    df = pd.DataFrame(dataset['train'])

    # Dataset labels (0=Bearish, 1=Bullish, 2=Neutral) don't match FinBERT's own
    # id2label (0=positive, 1=negative, 2=neutral), so remap to keep the pretrained
    # classification head's positive/negative/neutral slots aligned with training data.
    LABEL_TO_FINBERT = {0: 1, 1: 0, 2: 2}
    df['label'] = df['label'].map(LABEL_TO_FINBERT)

    train, test = train_test_split(df[['text', 'label']], test_size=0.2, random_state=42)
    
    train.to_json('data/train.json', orient='records', lines=True)
    test.to_json('data/test.json', orient='records', lines=True)
    
    print(f"\nSUCCESS! Your AI will now train on real data.")
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    print(f"Label distribution - Positive: {(df['label']==0).sum()}, Negative: {(df['label']==1).sum()}, Neutral: {(df['label']==2).sum()}")

if __name__ == '__main__':
    prepare_massive_data()