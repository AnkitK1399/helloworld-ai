from rag_pipeline import initialize_rag, query_rag


# Initialize RAG system
docs, index = initialize_rag()

print("RAG system is ready!")
print("Ask a question about accounting documents.")
print("Type 'exit' to quit.")


while True:

    question = input("\nYour question: ")

    if question.lower() == "exit":
        break

    # Retrieve relevant document
    result = query_rag(question, docs, index)

    filename, content = result

    print("\nMost relevant document:", filename)
    print("Content:", content[:300], "...")