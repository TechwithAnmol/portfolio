from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

load_dotenv()
os.getenv("API_KEY")
genai.configure(api_key=os.getenv("API_KEY"))

# Global variable to store chat history
chat_history = []

app = Flask(__name__)

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are AnmoLLM, a personal assistant for Anmol Singh. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just use your own understanding of Anmol's resume to reply. Every question will be related to Anmol, and you have the details.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_response(question):
    global chat_history  # Access global chat history variable
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    
    # Append user question to chat history
    chat_history.append(question)
    
    # Append chat history to context
    context = "\n".join(chat_history)
    
    response = chain({"input_documents": docs, "question": question, "context": context}, return_only_outputs=True)
    
    # Append model response to chat history
    chat_history.append(response["output_text"])

    return response["output_text"]

@app.route('/')
def index():
    return render_template('index.html', css_file="static/css/style.css")

@app.route('/index.html')
def indexreturn():
    return render_template('index.html', css_file="static/css/style.css")

@app.route('/portfolio-details.html')
def portfolio_details():
    return render_template('portfolio-details.html', css_file="static/css/style.css")

@app.route('/portfolio-details copy.html')
def portfolio_details_copy():
    return render_template('portfolio-details copy.html', css_file="static/css/style.css")

@app.route('/portfolio-details copy 2.html')
def portfolio_details_copy2():
    return render_template('portfolio-details copy 2.html', css_file="static/css/style.css")



@app.route('/get_response', methods=['POST'])
def get_user_question():
    data = request.get_json()
    question = data['question']
    response = get_response(question)
    return response

if __name__ == '__main__':
    pdf_path = "Anmol Singh Resume.pdf"  # Hardcoded path to PDF file
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    app.run(debug=True)  # Run the Flask app



#Working code with command line I/P O/P
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

# load_dotenv()
# os.getenv("API_KEY")
# genai.configure(api_key=os.getenv("API_KEY"))

# # Global variable to store chat history
# chat_history = []

# def get_pdf_text(pdf_path):
#     text = ""
#     pdf_reader = PdfReader(pdf_path)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     You are AnmoLLM, a personal assistant for Anmol Singh. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just use your own understanding of Anmol's resume to reply. Every question will be related to Anmol, and you have the details.\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input():
#     global chat_history  # Access global chat history variable
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     while True:
#         user_question = input("Ask a Question from the PDF Files (Type 'exit' to quit): ")
        
#         if user_question.lower() == 'exit':
#             break
        
#         docs = new_db.similarity_search(user_question)
#         chain = get_conversational_chain()
        
#         # Append user question to chat history
#         chat_history.append(user_question)
        
#         # Append chat history to context
#         context = "\n".join(chat_history)
        
#         response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
        
#         # Append model response to chat history
#         chat_history.append(response["output_text"])
        
#         print("Reply: ", response["output_text"])

# def main():
#     print("Chat with PDF using Gemini💁")
    
#     pdf_path = "Anmol Singh Resume.pdf"  # Hardcoded path to PDF file
    
#     raw_text = get_pdf_text(pdf_path)
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)
#     print("Processing Done!")
    
#     user_input()

# if __name__ == "__main__":
#     main()























































#Working resume on streamlit
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

# load_dotenv()
# os.getenv("API_KEY")
# genai.configure(api_key=os.getenv("API_KEY"))

# # Global variable to store chat history
# chat_history = []

# def get_pdf_text(pdf_path):
#     text = ""
#     pdf_reader = PdfReader(pdf_path)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     You are AnmoLLM, a personal assistant for Anmol Singh. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just use your own understanding of Anmol's resume to reply. Every question will be related to Anmol, and you have the details.\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     global chat_history  # Access global chat history variable
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
    
#     # Append user question to chat history
#     chat_history.append(user_question)
    
#     # Append chat history to context
#     context = "\n".join(chat_history)
    
#     response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
    
#     # Append model response to chat history
#     chat_history.append(response["output_text"])

#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     pdf_path = "Anmol Singh Resume.pdf"  # Hardcoded path to PDF file

#     with st.spinner("Processing..."):
#         raw_text = get_pdf_text(pdf_path)
#         text_chunks = get_text_chunks(raw_text)
#         get_vector_store(text_chunks)
#     st.success("Done")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

# if __name__ == "__main__":
#     main()
