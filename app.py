# Import Langchain dependenices
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import  RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Bring in Streamlit for UI dev
import streamlit as st
# Bring in watson interface
#from watsonxlangchain import LangChainInterface

#create llm thru langchain
creds = {
    'apikey' : 'cpd-apikey-IBMid-691000CM03-2024-04-13T21:31:37Z',
    'url':'https://us-south.ml.cloud.ibm.com'
}
llm = LangChainInterface(
    credentials = creds,
    model = 'meta-llama/llama-2-70b-chat',
    params = {
        'decoding_method':'sample',
        'max_new_tokens':200,
        'temperature':0.5
    },
    project_id='baa5d59f-e936-4dfa-9db1-6fac7fbef880')


# setup the app title
st.title('State of Food Security and Nutrition in the World 2023')

#show prev prompts
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

#Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt Here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    #for prev prompts
    st.session_state.messages.append({'role':'user', 'content':prompt})