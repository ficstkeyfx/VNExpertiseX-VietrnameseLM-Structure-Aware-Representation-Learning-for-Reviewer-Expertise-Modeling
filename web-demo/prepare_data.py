import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import faiss
import pickle
import os
import torch

def main():
    print("Loading dataset...")
    csv_path = '../articles_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Clean data
    df = df.dropna(subset=['title', 'author'])
    
    # Subsample for demo purposes to avoid hours of CPU processing
    if len(df) > 500:
        print("Subsampling 500 papers for fast demo evaluation...")
        df = df.sample(n=500, random_state=42).copy()
        
    print(f"Total papers to process: {len(df)}")
    
    # Prepare text for embedding
    df['text_to_embed'] = df['title'] + " " + df['abstract'].fillna('')
    sentences = df['text_to_embed'].tolist()
    
    # Load BGE-M3 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading BGE-M3 model on {device}...")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True if device == 'cuda' else False, device=device)
    
    print("Computing dense embeddings (this may take a while)...")
    try:
        embeddings_output = model.encode(sentences, batch_size=12, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_embeddings = embeddings_output['dense_vecs']
    except Exception as e:
        print(f"Error during embedding: {e}")
        return
        
    # Normalize for cosine similarity
    faiss.normalize_L2(dense_embeddings)
    
    print("Building FAISS index...")
    dimension = dense_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(dense_embeddings)
    
    print("Saving index and metadata...")
    faiss.write_index(index, 'paper_index.faiss')
    
    # Save metadata so we can map indices back to paper info
    metadata = df[['title', 'abstract', 'author', 'url', 'research_field', 'english_title', 'english_abstract']].to_dict('records')
    with open('paper_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print("Data preparation complete!")

if __name__ == '__main__':
    main()
