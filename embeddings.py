from sentence_transformers import SentenceTransformer
import os # help in reading file from the folder

#loading of embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_docs(data_folder='data'):
    documents = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_folder,filename)
            with open(filepath,'r', encoding='utf-8') as file:
                text = file.read()
                documents.append((filename,text))

    
    return documents
def create_embeddings(documents):
    texts = []

    for doc in documents:
        texts.append(doc[1])
    
    embedding = model.encode(texts) # returns numpy array

    return embedding

if __name__ == '__main__':
    docs = load_docs()
    embedding = create_embeddings(docs)

    for i , doc in enumerate(docs):
        print('File Name: ', doc[0])
        print('size of vector: ', len(embedding[0]))
        print()

   

