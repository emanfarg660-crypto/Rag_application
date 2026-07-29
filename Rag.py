
import google.generativeai as genai
import streamlit as st
import os
import faiss
import numpy as np
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

api = 'Google_Api_Key'
st.title('RAG Application Using Gemini AI')

#configure google generative ai
if api:
    genai.configure(api_key= api)
else:
    st.error('Your API is Not Found')

#function to generate text using google API
def Generate_Text(text):
    #call generative model from google AI
    model = genai.GenerativeModel('gemini-3-flash-preview')
    #generate response from Gemini
    response = model.generate_content(text)
    return response.text

if 'messages' not in st.session_state:
    st.session_state.messages = []

#display all messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):  #role: user / assistant
        st.markdown(message['content']) #content : user_input / response from model


#upload pdf file
upload_file = st.file_uploader('Choose a file...', type = ['pdf'])

if upload_file is not None:
    with tempfile.NamedTemporaryFile(delete= False, suffix= '.pdf') as tempfile:
        tempfile.write(upload_file.read()) #read and write a copy from the pdf
        tempfile_path = tempfile.name

    loader = PyPDFLoader(tempfile_path)
    documents = loader.load()

    #st.write(documents)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    #create embedding
    text = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(text, show_progress_bar= True)
    embedding_matrix = np.array(embeddings)
    #st.write(embedding_matrix)

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    st.success('PDF Processed Successfully.')

    user_input = st.chat_input('Please Enter Your Text...')

    if user_input:
    #display user message
        with st.chat_message('user'):
            st.markdown(user_input)

        st.session_state.messages.append({'role' : 'user', 'content': user_input})

        question_embedding = embedding_model.encode([user_input])

        k = 1
        distances, indicies = index.search(question_embedding, k)
        similar_doc = [documents[i] for i in indicies[0]]

        context = ""

        for i, doc in enumerate(similar_doc):
            context += doc.page_content + '\n'

        prompt = f'You are an assistant who retrieves answer based in the following content: {context}\nQuestion:{user_input}'

        response_text = Generate_Text(prompt)

        #generate response from google gemini API
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            with st.spinner('Generating Response....'):
                response_text = Generate_Text(prompt)
                message_placeholder.markdown(f'{response_text}')
        st.session_state.messages.append({'role' : 'assistant', 'content' : response_text})

else:
    st.write('Please, Upload a PDF File.')
