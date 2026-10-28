
# Install:
# pip install langchain langchain-openai langchain-chroma langchain-huggingface
# pip install langchain-community langchain-text-splitters beautifulsoup4
# pip install sentence-transformers gradio python-dotenv

from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

def build_index(html_path):
    # 1. Load the HTML file
    documents = BSHTMLLoader(html_path).load()

    # 2. Split HTML text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Build the in-memory vector store
    return Chroma.from_documents(chunks, embedder)


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a helpful HTML document assistant.
Answer using ONLY the context below.
If the context does not contain the answer, say:
"I couldn't find that in the document."

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


state = {"db": None}


def upload(html_file):
    if html_file is None:
        return "Please upload an HTML file."

    state["db"] = build_index(html_file.name)
    return "HTML file indexed successfully! Ask your questions."


def chat(message, history):
    if state["db"] is None:
        return "Please upload an HTML file first."

    return ask(state["db"], message)


with gr.Blocks(title="Chat with HTML") as demo:
    gr.Markdown("## Chat with your HTML page (RAG)")

    html_file = gr.File(
        label="Upload HTML file",
        file_types=[".html", ".htm"],
        type="filepath"
    )

    status = gr.Markdown()
    html_file.upload(upload, inputs=html_file, outputs=status)

    gr.ChatInterface(fn=chat)

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True
)
