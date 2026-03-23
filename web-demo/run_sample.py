import pickle; import faiss; import torch; import pandas as pd; import numpy as np
from FlagEmbedding import BGEM3FlagModel

query_title = 'Các yếu tố tác động đến lòng trung thành của khách hàng: Trường hợp nghiên cứu các siêu thị tại tỉnh Đồng Nai'
query_abstract = 'Lòng trung thành của khách hàng là yếu tố rất quan trọng và cần thiết mà mọi nhà kinh doanh luôn phải quan tâm đo sự cạnh tranh khốc liệt của ngành bán lẻ hiện nay.'
query_text = query_title + ' ' + query_abstract

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)
index = faiss.read_index('paper_index.faiss')
with open('paper_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Dense Retrieval
query_dense = model.encode([query_text], batch_size=1, max_length=512, return_dense=True)['dense_vecs']
faiss.normalize_L2(query_dense)
D, I = index.search(query_dense, 100)

retrieved_docs = [metadata[i] for i in I[0] if i != -1]
sentence_pairs = [[query_text, f"{doc['title']} {doc.get('abstract', '')}"] for doc in retrieved_docs]

rerank_scores_dict = model.compute_score(sentence_pairs, max_passage_length=512, weights_for_different_modes=[0.4, 0.2, 0.4])
rerank_scores = rerank_scores_dict['colbert+sparse+dense'] if isinstance(rerank_scores_dict, dict) else rerank_scores_dict

for doc, score in zip(retrieved_docs, rerank_scores):
    doc['rerank_score'] = float(score)

retrieved_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

author_scores = {}
for doc in retrieved_docs:
    authors_str = str(doc.get('author', ''))
    if authors_str == 'nan': continue
    authors = [a.strip() for a in authors_str.split(',') if a.strip()]
    for a in authors:
        author_scores[a] = author_scores.get(a, 0.0) + doc['rerank_score']

sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print('TOP RECOMMENDED AUTHORS:')
for a, s in sorted_authors:
    print(f'{a}: {s:.2f}')
