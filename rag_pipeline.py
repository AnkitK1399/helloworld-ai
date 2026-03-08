from embeddings import load_docs, create_embeddings, model
from vector_store import create_vector_store, search_vector

def initialize_rag():

    docs = load_docs()
    embeddings = create_embeddings(docs)
    index = create_vector_store(embeddings)
    return docs, index

def query_rag(question, docs, index):

    query_embedding = model.encode(question)
    indices, distances = search_vector(index, query_embedding)
    results = []

    for indx in indices[0]:
        results.append(docs[int(indx)])
    return results