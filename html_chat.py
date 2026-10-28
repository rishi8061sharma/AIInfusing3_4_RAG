# pip install langchain langchain-openai langchain-chroma langchain-huggingface \
#             langchain-community pypdf sentence-transformers gradio python-dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader



def build_index(html_paths):
    documents = []

    # Process each HTML file individually
    for html_path in html_paths:
        loader = BSHTMLLoader(
            html_path,
            open_encoding="utf-8",
            bs_kwargs={"features": "html.parser"}
        )

        file_documents = loader.load()

        # Store the source filename in metadata
        for document in file_documents:
            document.metadata["source"] = os.path.basename(html_path)

        documents.extend(file_documents)

    if not documents:
        raise ValueError("No HTML content was loaded.")

    # Split all loaded documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Generate embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build a single index containing all HTML files
    db = Chroma.from_documents(chunks, embedder)

    return db


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful HTML document assistant.
Answer the question using ONLY the context below.

If the answer is not present in the context, say:
"I couldn't find that in the uploaded HTML documents."

At the end, list the source filenames used, for example:
Sources: java.html, spring.html

Context:
{context}

Question: {question}
""")





def ask(db, question):
    chunks = db.similarity_search(question, k=4)

    context = "\n\n".join(
        f"[Source: {c.metadata.get('source', 'HTML document')}] "
        f"{c.page_content}"
        for c in chunks
    )

    chain = prompt | model

    return chain.invoke({
        "context": context,
        "question": question
    }).content



import gradio as gr

state = {"db": None}                    # ① remember the index across turns

def upload(html_files):
    if not html_files:
        state["db"] = None
        return "Please upload at least one HTML file."

    try:
        html_paths = [
            file.name if hasattr(file, "name") else str(file)
            for file in html_files
        ]

        state["db"] = build_index(html_paths)

        filenames = [
            os.path.basename(path) for path in html_paths
        ]

        return (
            f"✅ Indexed {len(filenames)} HTML file(s):\n\n"
            + "\n".join(f"- {name}" for name in filenames)
            + "\n\nYou can now ask questions across all uploaded files."
        )

    except Exception as error:
        state["db"] = None
        return f"❌ Failed to index HTML files: {error}"

def chat(message, history):
    if state["db"] is None:
        return "Please upload HTML files first."

    return ask(state["db"], message)

with gr.Blocks(title="📄 Chat with Multiple HTML Files") as demo:
    gr.Markdown("## 📄 Chat with your multiple HTML files (powered by RAG)")
    html_files    = gr.File(
        label="Upload HTML files",
        file_count="multiple",
        file_types=[".html", ".htm"],
        type="filepath"
    )
    status = gr.Markdown()
    
    index_button = gr.Button("Index HTML Files")
    index_button.click(
        fn=upload,
        inputs=html_files,
        outputs=status
    )
    gr.ChatInterface(fn=chat)

demo.launch(share=True)              # share=True → public link!