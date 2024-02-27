import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from HTMLTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


# 함수명 : get_pdf_text, 기능 : 각 pdf에서 각 페이지의 text를 추출해 'text'변수에 추가
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 함수명 : get_text_chunks, 기능 : 우리가 얻은 text를 chunks나 pieces로 얻어냄
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
                separator = "\n", 
                chunk_size = 1000,
                chunk_overlap =200,
                length_function = len
            )
    chunks = text_splitter.split_text(text)
    return chunks

# 함수명 get_vectorstore, 기능 : chunks를 vectorstore에 넣어 vectorstore를 만듦
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "Clinical-AI-Apollo/Medical-NER")
    vectorstores = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstores

# 함수명 : get_conversation_chain, 기능 : 
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs = {"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever,
        memory = memory
    )
    return conversation_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question' : user_question})
    st.write(response)
    
    
def main():
    # dotenv에서 api_key를 load
    load_dotenv()
    
    # 연결되는 웹사이트의 config
    st.set_page_config(page_title = "Chat with multiple PDFs", page_icon = ":books:")
    
    st.write(css, unsafe_allow_html = True)
    
    # intialize "conversation"
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
    st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html = True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html = True)
    
    
    
    with st.sidebar:
        st.subheader("Your documents")
        pdfs_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            # get pdf text
            raw_text = get_pdf_text(pdfs_docs)
            st.write(raw_text)
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)
            
            # create vector store
            vectorstore = get_vectorstore(text_chunks)
            
            # create conversation chain
            conversation = get_conversation_chain(vectorstore)
     
    st.session_state.conversation
            
if __name__ == '__main__':
    main()
    
