from langchain_community.llms import Ollama
from flask import Flask, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask( __name__)

folder_path = "db"
catched_llm = Ollama(model="llama3")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompts= PromptTemplate.from_template(
    """
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
    Context: {context}
    Answer: 
    [\INST]
    """
)

# print(response)

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /Ai Called")
    json_content = request.json
    query = json_content.get("query")
    
    print("query: ", query)
    response = catched_llm.invoke(query)
    print(response)
    
    response_answer = {"answer": response}
    return response_answer
    
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name=file.filename
    save_file = "pdf/"+file_name
    file.save(save_file)
    print("filename: ", file_name)
    
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory = folder_path
    )
    vector_store.persist()
    
    response = {"status: ": "Suceesfully Uploaded", "Filename: ": file_name, "doc_len: ": len(docs), "chunks: ": len(chunks) }
    return response

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /askpdf called")
    json_content = request.json
    query = json_content.get("query")
    
    print(f"query: {query}")
    
    print("loading vector store")
    vectore_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    
    print("Creating Chain")
    retriever = vectore_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":20,
            "score_threshold": 0.1,
        },
    )
    
    document_chain = create_stuff_documents_chain(catched_llm, raw_prompts)
    chain = create_retrieval_chain(retriever,document_chain)
    
    result = chain.invoke({"input": query})
    print(result)
    
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )
    
    response_answer = {"answer: ": result["answer"], "sources": sources}
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()