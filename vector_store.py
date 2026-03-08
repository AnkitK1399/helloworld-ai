import faiss
import numpy as np

def create_vector_store(embeddings):
     dimension = len(embeddings[0])
     index = faiss.IndexFlatL2(dimension)
     embeddings_array = np.array(embeddings)
     index.add(embeddings_array)
     return index

def search_vector(index, query_embedding, k=1):
    query_vector =  np.array([query_embedding])
    distance, indices = index.search(query_vector, k)
    return distance, indices


