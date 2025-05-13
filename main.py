import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# To maintain chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def display_chat():
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(
                f'<div class="chat-bubble user-bubble">'
                f'{message["content"]}'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bubble assistant-bubble">'
                f'{message["content"]}'
                f'</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":book:")
    st.markdown("""
        <style>
        .chat-bubble {
            display: inline-block;
            padding: 10px 20px;
            margin-bottom: 10px;
            border-radius: 15px;
            max-width: 70%;
            line-height: 1.5;
            font-size: 16px;
        }
        .user-bubble {
            background-color: #dcf8c6;
            color: #000;
            text-align: right;
            margin-left: auto;
        }
        .assistant-bubble {
            background-color: #f1f1f1;
            color: #333;
            text-align: left;
            margin-right: auto;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        </style>
        """, unsafe_allow_html=True)

    st.header("PDF Reader")

    with st.sidebar:
        st.title("Menu:")
        api_key = st.text_input("Enter your Google API Key", type="password")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Process PDFs", key="process_pdfs"):
            if api_key:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Done")
            else:
                st.error("Please enter your Google API Key")

        st.info("This chatbot uses Google Generative AI model for conversational responses.")
        st.info("Created By Pranav Lejith(Amphibiar)")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Submit Question", key="submit_question"):
        if user_question and api_key:
            # Add user question to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})

            # Get response from the model
            response = user_input(user_question, api_key)
            
            # Add model response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display chat history
            display_chat()
        elif not api_key:
            st.error("Please enter your Google API Key")

if __name__ == "__main__":
    main()
