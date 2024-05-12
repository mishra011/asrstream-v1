import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup



load_dotenv()


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    load_dotenv()
    # Create llm
    
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    """
    llm = Replicate(
        streaming = True,
        model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        callbacks=[StreamingStdOutCallbackHandler()],
        input = {"temperature": 0.01, "max_length" :500,"top_p":1})
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    #uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
    submitted = False
    with st.sidebar.form("data_form"):
        st.write("Enter the Url")
        uploaded_url = st.text_input("Enter the Url Here")
        slider_val = st.slider("Form slider")
        #checkbox_val = st.checkbox("Form checkbox")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        print("SUBMITTED :: ", submitted, uploaded_url, slider_val)

    if uploaded_url:
        text = []
        print("URL :: ", uploaded_url)
        if True:
            loader = None
            #loader = SitemapLoader(uploaded_url)
            #loader = WebBaseLoader(uploaded_url)
            #loader = WebBaseLoader(["https://www.espn.com/", "https://google.com"])
            #uploaded_url = "https://www.tatacommunications.com/"
            urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
                ]
            
            #loader = SeleniumURLLoader(urls=urls)
            #loader = UnstructuredURLLoader(urls=urls)

            #loader = SitemapLoader(uploaded_url)
            #loader.requests_per_second = 2
            #loader.requests_kwargs = {"verify": False}
            #loader = WebBaseLoader(uploaded_url)
            if slider_val:
                if slider_val in [1,2]:
                    slider_val = slider_val
                else:
                    slider_val = 1
            #loader = RecursiveUrlLoader(url=uploaded_url, max_depth=slider_val, extractor=lambda x: Soup(x, "html.parser").text)
            loader = SitemapLoader(uploaded_url)
            documents = loader.load()

            if loader:
                text.extend(documents)
                uploaded_url = None
                
        
        #text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        text_chunks = text_splitter.split_documents(text)
        print("CHUNKS :: ", text_chunks)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        print("embeddings :: ")
        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        print("vector_store :: ")
        # Create the chain object
        chain = create_conversational_chain(vector_store)
        print("chain :: ")

        
        display_chat_history(chain)
        print("display_chat_history")


if __name__ == "__main__":
    main()

