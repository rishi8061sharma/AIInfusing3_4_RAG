
# Install dependencies:
# python -m pip install langchain langchain-openai langchain-chroma
# python -m pip install langchain-huggingface langchain-community
# python -m pip install langchain-text-splitters pypdf sentence-transformers
# python -m pip install beautifulsoup4 gradio python-dotenv

import os
import gradio as gr

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Configure the LLM
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 2. Load multiple PDF and HTML files into one index
def build_index(file_paths):
    documents = []

    for file_path in file_paths:
        extension = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        # Load PDF files
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
            file_documents = loader.load()

        # Load HTML files
        elif extension in (".html", ".htm"):
            loader = BSHTMLLoader(
                file_path,
                open_encoding="utf-8",
                bs_kwargs={"features": "html.parser"}
            )
            file_documents = loader.load()

        else:
            raise ValueError(f"Unsupported file type: {filename}")

        # Preserve source filename and document type
        for document in file_documents:
            document.metadata["source"] = filename
            document.metadata["file_type"] = extension.lstrip(".")

        documents.extend(file_documents)

    if not documents:
        raise ValueError("No document content was loaded.")

    # 3. Split all documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # 4. Generate embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Create one vector index for all uploaded files
    db = Chroma.from_documents(chunks, embedder)

    return db


# 6. RAG prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the question using ONLY the supplied context.
The context can come from PDF and HTML documents.

If the answer is not present in the context, say:
"I couldn't find that in the uploaded documents."

Cite the source filenames used in your answer.
For PDF sources, include the page number when available.
Do not invent filenames or page numbers.

Context:
{context}

Question:
{question}

Answer:
""")


# 7. Retrieve relevant chunks and generate an answer
def ask(db, question):
    chunks = db.similarity_search(question, k=4)

    context_parts = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown document")
        page = chunk.metadata.get("page")

        if page is not None:
            source_label = f"{source}, page {page + 1}"
        else:
            source_label = source

        context_parts.append(
            f"[Source: {source_label}]\n{chunk.page_content}"
        )

    context = "\n\n".join(context_parts)

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


# 8. Store the current index
state = {"db": None}


# 9. Upload and index multiple files
def upload(files):
    if not files:
        state["db"] = None
        return "Please upload at least one PDF or HTML file."

    try:
        # With Gradio type="filepath", files are normally path strings
        file_paths = [
            file.name if hasattr(file, "name") else str(file)
            for file in files
        ]

        # Validate file types
        allowed_extensions = {".pdf", ".html", ".htm"}

        for path in file_paths:
            extension = os.path.splitext(path)[1].lower()

            if extension not in allowed_extensions:
                raise ValueError(
                    f"Unsupported file type: {os.path.basename(path)}"
                )

        # Build one index from every uploaded file
        new_db = build_index(file_paths)
        state["db"] = new_db

        filenames = [
            os.path.basename(path) for path in file_paths
        ]

        return (
            f"✅ Successfully indexed {len(filenames)} file(s):\n\n"
            + "\n".join(f"- {name}" for name in filenames)
            + "\n\nYou can now ask questions across all uploaded files."
        )

    except Exception as error:
        state["db"] = None
        return f"❌ Failed to index files: {error}"


# 10. Chat handler
def chat(message, history):
    if state["db"] is None:
        return "Please upload and index your PDF or HTML files first."

    try:
        return ask(state["db"], message)
    except Exception as error:
        return f"❌ Failed to answer your question: {error}"


# 11. Gradio interface
with gr.Blocks(title="Chat with PDF and HTML Files") as demo:
    gr.Markdown(
        "# 📚 Chat with PDF and HTML Files\n"
        "Upload multiple PDF and HTML documents, index them, "
        "and ask questions across all of them."
    )

    files = gr.File(
        label="Upload PDF and HTML files",
        file_count="multiple",
        file_types=[".pdf", ".html", ".htm"],
        type="filepath"
    )

    status = gr.Markdown()

    index_button = gr.Button("Index Uploaded Files")

    index_button.click(
        fn=upload,
        inputs=files,
        outputs=status
    )

    gr.ChatInterface(fn=chat)


# 12. Launch with public sharing
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True
)
