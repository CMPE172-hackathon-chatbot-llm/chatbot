# Import Langchain dependenices
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import  RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Bring in Streamlit for UI dev
import streamlit as st

# Niti's API key: EzjmGauMG1UYYzBu27xpIZR6EsLIx-uGOvAcEc7qOspW
import os
from getpass import getpass

# Prompt the user for the API key
watsonx_api_key = 'EzjmGauMG1UYYzBu27xpIZR6EsLIx-uGOvAcEc7qOspW'

# Set the API key as an environment variable
os.environ["WATSONX_APIKEY"] = watsonx_api_key

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 999,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# Bring in watson interface
#from watsonxlangchain import LangChainInterface
from langchain_ibm import WatsonxLLM

watsonx_llm = WatsonxLLM(
    model_id="meta-llama/llama-2-70b-chat",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="e49cf2c9-0f52-481a-af0f-f310e2f2eb35",
    params=parameters,
)

# setup the app title
st.title('Ask F.O.O.D. Bot Anything!')

# Adding a subheader as subtitle
st.subheader('The F.O.O.D. Bot is here to help you with your food-related questions!')

# Setup a session state message variable  to hold all old messages
if 'messages' not in st.session_state:
    st.session_state.messages= []

# Display all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


#Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt Here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content':prompt})
    # send the prompt to llm
    response = watsonx_llm(prompt)
    # show llm response
    st.chat_message('assistant').markdown(response)
    # store llm response in state
    st.session_state.messages.append(
        {'role':'assistant', 'content':response}
    )