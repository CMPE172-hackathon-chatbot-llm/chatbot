# Import Langchain dependenices
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import  RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Bring in Streamlit for UI dev
import streamlit as st
# Bring in watson interface
from watsonxlangchain import LangChainInterface

# setup the app title
st.title('Ask watsonx')

#Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt Here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)