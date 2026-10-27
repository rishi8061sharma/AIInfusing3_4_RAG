from langchain_text_splitters import RecursiveCharacterTextSplitter



# Step 1: Define your document
your_long_document = """
LangChain is a framework for developing applications
powered by large language models.

Recursive character splitting divides a long document
into smaller chunks while trying to preserve paragraphs
and sentences.

Chunking is important in Retrieval-Augmented Generation
(RAG), where documents are split, converted into embeddings,
and stored in a vector database for semantic search.
""" * 20


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # aim for ~800 chars per chunk
    chunk_overlap=100,     # adjacent chunks share 100 chars (context glue)
)

chunks = splitter.split_text(your_long_document)
print(len(chunks))    # → e.g. 47 chunks ready to embed