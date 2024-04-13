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

# Setup credentials dictionary
cred = {
    'apikey':'cpd-apikey-IBMid-691000CM03-2024-04-13T21:31:37Z',
    'url':'https://us-south.ml.cloud.ibm.com'
}
# Create LLM using Langchain
llm = LangChainInterface(
    credentials = cred,
    model = 'meta-llama/llama-2-70b-chat',
    params = {
        'decoding_method':'sample',
        'max_new_tokens':200,
        'temperature':0.5
    },
    project_id='baa5d59f-e936-4dfa-9db1-6fac7fbef880')

# This function laods a Food PDF
def load_pdf():
    # Update the PDF
    pdf = 'SOFI-2023.pdf'
    loaders = [PyPDFLoader(pdf)]
    # Create index - aka vector database - ask chromadb
    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    # Return the vector database
    return index
# Load the PDF
index = load_pdf()

# Create a Q&A chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

# Setup the app title
st.title('Ask Food Bot')

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content']) 

# Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt Here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content':prompt})
    # Send the prompt to the PDF Q&A CHAIN
    response = chain.run(prompt)
    # Show the LLM response
    st.chat_message('assistant').markdown(response)
    # Store the LLM response in state
    st.session_state.messages.append(
        {'role':'assistant', 'content':response}
    )