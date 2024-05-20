## General Imports
import os
from time import sleep
from dotenv import load_dotenv

## UI & MongoDB Library
import streamlit as st
from pymongo import MongoClient

## Library Required for Vector Store
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
# Since the Vector Database is already created, there is no need to regenerate the vector store
# from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader


## Library Required for LLMs and Chats
from langchain.llms import AzureOpenAI                                          ## This object is a connector/wrapper for OpenAI LLM engine
from langchain_openai import AzureChatOpenAI                                    ## This object is a connector/wrapper for ChatOpenAI engine
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage      ## These are the commonly used chat messages

##

load_dotenv(override=True)

st.title("Simple Chatbot for SC1015")
st.text("This is just a short description")


def search_chunks(query):
    search_result = st.session_state['retrieval'].invoke(query)
    context = []
    for r in search_result:
        context.append(r.page_content)

    instruction = "try to understand the userquery and answer based on the context given below:\n"
    return SystemMessage(content=f"{instruction}'context':{context}, 'userquery':{query}")

if "text_embedding" not in st.session_state:
    st.session_state['text_embedding'] =  AzureOpenAIEmbeddings(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment=os.environ["AZURE_TEXT_EMBEDDING"],
        model='text-embedding-ada-002'
    )
    vectorDB = FAISS.load_local("db/sc1015", st.session_state['text_embedding'] , allow_dangerous_deserialization=True)
    st.session_state['retrieval'] = vectorDB.as_retriever(search_kwargs={"k": 5})

    st.session_state['llm'] = AzureChatOpenAI(
        openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment=os.environ['DEPLOYMENT_NAME'],
        temperature=1
    )

    persona = "You are a teaching assistant at for the course SC1015 at NTU."
    task ="your task is to answer student query about the data science and ai course."
    context = "the context will be provided based on the course information and FAQ along with the user query"
    condition = "If user ask any query beyond data science and ai, tell the user you are not an expert of the topic the user is asking and say sorry. If you are unsure about certain query, say sorry and advise the user to contact the instructor at instructor@ntu.edu.sg"
    ### any other things to add on

    ## Constructing initial system message
    sysmsg = f"{persona} {task} {context} {condition}"
    st.session_state['conversations'] = [SystemMessage(content=sysmsg)]

    greetings = '''Hello my name is Alice, and I am a Automated Teaching Assistant for SC1015 - Data Science & AI. I am here to help, feel free to ask any questions.
    '''
    st.session_state['conversations'].append(AIMessage(content=greetings))
    st.session_state['msgtypes'] = {HumanMessage: "Human", AIMessage:"AI", SystemMessage:"System"}





if 'conversations' in st.session_state:
    for conv in st.session_state['conversations']:
        if isinstance(conv, SystemMessage):
            continue
        role = st.session_state.msgtypes[type(conv)]
        with st.chat_message(role):
            st.markdown(conv.content)

if query:= st.chat_input("Your Message"):
    st.chat_message("Human").markdown(query)
    st.session_state['conversations'].append(HumanMessage(content=query))


    context = search_chunks(query)
    templog = st.session_state['conversations'] + [context]
    response = st.session_state['llm'].invoke(templog)
    # response = st.session_state['llm'].invoke(st.session_state['conversations'])
    st.chat_message("AI").markdown(response.content)
    st.session_state['conversations'].append(response)

# st.markdown(st.session_state['conversations'])

