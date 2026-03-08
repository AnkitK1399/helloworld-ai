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
    results = query_rag(question, docs, index)

   

    print("\nTop relevant documents:\n")

    for filename, content in results:
        print("Document:", filename)
        print("Content:", content[:200])
        print()