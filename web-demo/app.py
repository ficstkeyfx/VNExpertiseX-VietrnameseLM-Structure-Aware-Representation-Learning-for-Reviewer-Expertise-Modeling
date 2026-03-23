import streamlit as st
import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import faiss
import pickle
import os
import torch
import gc

# Filter out noisy author entries (institutions, faculties, etc.)
INSTITUTION_KEYWORDS = [
    'đại học', 'trường', 'viện', 'khoa', 'học viện', 'phân hiệu',
    'university', 'college', 'institute', 'faculty', 'center',
    'trung tâm', 'bộ môn', 'phòng', 'ban', 'sở', 'bệnh viện',
    'công ty', 'tập đoàn', 'chi nhánh', 'phân viện', 'academy',
    'department', 'school of', 'lab ', 'laboratory',
]

def is_institution_name(name):
    """Return True if the name looks like an institution, not a person."""
    name_lower = name.lower().strip()
    if len(name_lower) < 3:
        return True
    for kw in INSTITUTION_KEYWORDS:
        if kw in name_lower:
            return True
    return False

# 1. Page Configuration
st.set_page_config(page_title="VNExpertiseLM Demo", layout="wide")

# Navigation
page = st.sidebar.selectbox("Navigation", ["Recommend Reviewers", "View Indexed Papers"])

# 2. Caching Models & Data
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True if device == 'cuda' else False, device=device)
    return model

@st.cache_resource
def load_index_and_metadata():
    if not os.path.exists('paper_index.faiss') or not os.path.exists('paper_metadata.pkl'):
        return None, None
    index = faiss.read_index('paper_index.faiss')
    with open('paper_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

@st.cache_resource
def load_full_dataset():
    csv_path = '../articles_data.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

if page == "Recommend Reviewers":
    st.title("VNExpertiseLM Reviewer Recommendation")

    st.info("Loading model and building index... please wait (this is cached).")
    model = load_model()
    index, metadata = load_index_and_metadata()
    full_df = load_full_dataset()
    
    if index is None or metadata is None:
        st.error("Index or metadata not found. Please run `python prepare_data.py` first.")
        st.stop()
    
    st.success("System is ready!")
    
    # 3. User Inputs
    with st.sidebar:
        st.header("Search Parameters")
        top_k_retrieve = st.slider("Initial Dense Retrieval (K)", 10, 500, 100)
        top_k_author = st.slider("Recommend Scientists (Top K)", 1, 20, 5)
    
    st.subheader("Input Article Details")
    title_input = st.text_input("📝 Title", placeholder="Enter the title of the paper...")
    author_input = st.text_input("👤 Author(s) (Optional)", placeholder="Enter author names...")
    abstract_input = st.text_area("📄 Abstract", placeholder="Enter the abstract of the paper...", height=150)
    
    if st.button("Recommend Reviewers"):
        if not title_input.strip() and not abstract_input.strip():
            st.warning("Please provide a title or abstract.")
        else:
            with st.spinner("Retrieving and calculating scores..."):
                query_text = f"{title_input} {abstract_input}".strip()
                
                # --- PHASE 1: Dense Retrieval ---
                query_embedding_dict = model.encode([query_text], batch_size=1, max_length=512, return_dense=True)
                query_dense = query_embedding_dict['dense_vecs']
                faiss.normalize_L2(query_dense)
                
                # Search FAISS
                D, I = index.search(query_dense, top_k_retrieve)
                retrieved_indices = I[0]
                retrieved_scores_dense = D[0]
                
                retrieved_docs = [metadata[i] for i in retrieved_indices if i != -1]
                
                # --- PHASE 2: VNExpertiseLM Reranking ---
                # Prepare pairs of (query, doc_text)
                sentence_pairs = []
                for doc in retrieved_docs:
                    doc_text = f"{doc['title']} {doc.get('abstract', '')}"
                    sentence_pairs.append([query_text, doc_text])
                
                # Compute score (dense + sparse + colbert combined)
                # weights_for_different_modes: [dense_weight, sparse_weight, colbert_weight]
                rerank_scores_dict = model.compute_score(sentence_pairs, max_passage_length=512, weights_for_different_modes=[0.4, 0.2, 0.4])
                
                # Extract combined scores
                if 'colbert+sparse+dense' in rerank_scores_dict:
                    rerank_scores = rerank_scores_dict['colbert+sparse+dense']
                else:
                    # Fallback if dictionary structure is different, sometimes compute_score returns simple list if modes are single
                    rerank_scores = rerank_scores_dict['colbert+sparse+dense'] if isinstance(rerank_scores_dict, dict) else rerank_scores_dict
                
                # Combine docs with rerank scores
                for doc, score in zip(retrieved_docs, rerank_scores):
                    doc['rerank_score'] = float(score)
                    
                # Sort by rerank score descending
                retrieved_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # --- PHASE 3: Author Aggregation ---
                author_scores = {}
                author_papers = {}
                for doc in retrieved_docs:
                    authors_str = str(doc.get('author', ''))
                    if not authors_str or authors_str == 'nan':
                        continue
                    # Split authors (assuming comma separation, adjust if dataset uses semicolon)
                    authors = [a.strip() for a in authors_str.split(',') if a.strip()]
                    authors = [a for a in authors if not is_institution_name(a)]
                    for a in authors:
                        # Accumulate score (sum of reranked scores of their papers in top K)
                        if a not in author_scores:
                            author_scores[a] = 0.0
                            author_papers[a] = []
                        author_scores[a] += doc['rerank_score']
                        if doc not in author_papers[a]:
                            author_papers[a].append(doc)
                
                # Sort authors by score
                sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
                top_authors = sorted_authors[:top_k_author]
                
                st.success("Recommendation Complete!")
                
                # --- DISPLAY RESULTS ---
                st.header(f"Top {top_k_author} Recommended Scientists")
                for rank, (author, score) in enumerate(top_authors, 1):
                    with st.expander(f"🏅 #{rank}: {author} (Score: {score:.2f})"):
                        st.write(f"**Total Rerank Score:** {score:.2f}")
                        
                        # Get 5-6 recent papers for this author
                        if full_df is not None:
                            author_mask = full_df['author'].astype(str).str.contains(author, regex=False, na=False)
                            all_author_papers = full_df[author_mask]
                            recent_papers = all_author_papers.head(6).to_dict('records')
                        else:
                            recent_papers = author_papers[author][:6]
                        
                        st.write(f"### Sample / Recent Papers (Found {len(recent_papers)} papers for this author)")
                        for p in recent_papers:
                            st.markdown(f"- **{p.get('title', 'No Title')}**")
                            if 'research_field' in p and pd.notna(p['research_field']):
                                st.caption(f"Field: {p['research_field']}")
                            abs_text = str(p.get('abstract', ''))
                            if abs_text and abs_text != 'nan':
                                st.write(f"> {abs_text[:200]}...")
                            if 'url' in p and pd.notna(p['url']):
                                st.markdown(f"[Link to article]({p['url']})")
                            st.divider()

elif page == "View Indexed Papers":
    st.title("Indexed Papers Database")
    st.write("View the papers currently loaded in the VNExpertiseLM index.")
    
    full_df = load_full_dataset()
    if full_df is not None:
        st.write(f"**Total Papers in Dataset:** {len(full_df)}")
        
        # Search functionality
        search_query = st.text_input("🔍 Search Database (Title, Author, Abstract, Keywords...)", placeholder="Type to search...")
        
        if search_query:
            # Case insensitive search across multiple columns
            mask = np.column_stack([full_df[col].astype(str).str.contains(search_query, case=False, na=False) for col in full_df.columns]).any(axis=1)
            filtered_df = full_df[mask]
            st.write(f"Found {len(filtered_df)} matches:")
            st.dataframe(filtered_df)
        else:
            st.write("Sample of the first 50 papers:")
            st.dataframe(full_df.head(50))
            
    else:
        st.error("No dataset found (`articles_data.csv`).")
