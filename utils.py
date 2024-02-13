from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
#ssfrom openai import Op
import os
import streamlit as st 

#############################
with open('openai_api.txt', 'r', encoding='utf-8') as f:
    for line in f:
        api_key = (line)

os.environ['OPENAI_API_KEY']=api_key

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len 
    ) 

    chunks = text_splitter.split_text(text) 

    embeddings = OpenAIEmbeddings()
    knowledgeBase = Chroma.from_texts(chunks, embeddings)

    return knowledgeBase


BASE_INSTRUCTION = """
Only use the data present in the document, do not make an answer. only answer to question if the answer is present in the retreival file.
"""

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


def create_agent_with_chat_history():
    from langchain.prompts import (
        SystemMessagePromptTemplate,
    )
    from langchain_openai import ChatOpenAI 
    RTOKEN= api_key

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=RTOKEN)
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = docsearch.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "search_available_doc",
        "Searches and returns excerpts from the available document.",
    )
    tools = [tool]
    from langchain import hub
    from langchain.prompts import PromptTemplate
    from langchain.prompts import SystemMessagePromptTemplate

    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages[0]=SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=BASE_INSTRUCTION))

    # LLM
    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613",
        openai_api_key= RTOKEN
    )


    from langchain.agents import AgentExecutor, create_openai_tools_agent

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    message_history = ChatMessageHistory()


    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def setup():
    #print('setup function called')

    # Retrieve the uploaded file
    uploaded_file = st.session_state.get('uploaded_file', None)

    # text preprocess and chain setup
    if uploaded_file is not None:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

        agent_with_chat_history = create_agent_with_chat_history()
        
        if 'agent_with_chat_history' not in st.session_state:
            st.session_state['agent_with_chat_history'] = agent_with_chat_history 
            #print('qa chain added to state')
    #st.write(st.session_state)