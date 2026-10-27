from langchain_text_splitters import CharacterTextSplitter

# 1. Define the input document
document = """
LangChain helps developers build applications powered by
large language models. Text splitting divides long documents
into smaller chunks for embedding and retrieval.
""" * 50

# 2. Configure the splitter
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=200,
    chunk_overlap=0,
    length_function=len,
)

# 3. Split the document
chunks = splitter.split_text(document)

# 4. Display the results
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {len(chunk)} characters")
    print(chunk[:100])