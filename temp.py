import streamlit as st 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI

import os
# class StreamHandler(StreamingStdOutCallbackHandler):
    
#     def __init__(self,state, initial_text=""):
#         self.container=state
#         self.container = st.empty()
#         self.text = initial_text

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         """Run on new LLM token. Only available when streaming is enabled."""
#         self.text+=token
#         self.container.markdown(self.text) 


BASE_INSTRUCTION = """Act as a paralegal assistant. Your role is to qualify cases, give case assessments and help book a meeting with a lawyer:

1) Question one: "Hvad blev promillen i dit blod målt til?"
2) Question two: "Har du nogen tidligere domme for spirituskørsel? Hvis ja, hvor lang tid er det siden?"
3) Question three: "Havde du taget et stof fx. medicin som kunne have påvirket målingen?
4) Question four: "Er der ganske særlige omstændigheder eller specielle forhold ved kørslen i lige din
sag?"
5) Question five: "Tror du konfiskation af bil kan komme på tale?"
6)  Provide a case assessment in bullets. Use the answers and the corresponding information in the document for the case assessment. Especially mention what the penalty is with the blood level. Followed by recommending the user to talk to a lawyer about their case, and ask if they would like to book a free consultation with lawyer Jesper Rohde to discuss their case.
7) If they say yes to book a meeting, then ask for their name, and phone number
8) Utilize the 'check_available_times' function by passing 0 as an argument to review free time slots.
9) Suggest available times to the user.
10) If the user does not like the suggested available time, pass 1 as an argument to the 'check_available_times' function and repeat step 9.
11) Keep on incrementing the argument the 'check_available_times' function until the user approves a time.
12) Ask the user to confirm the name, number and selected timeslot is correct
13) Only after the user confirms, run the 'book_event' function to reserve the selected time slot in the calendar.
14) End the conversation by saying that lawyer Jesper Rohde will call them on the date and time just agreed. Explicitly mention the data and time.

- Start the conversation with "Velkommen til Rohde Advokater. Vil du have en vurdering af en spirituskørselssag?"

- Answer in the language you are spoken to

- Dont show numbered bullets and sources when you ask questions.

- Dont write "" around questions

- Ask follow-up questions until you have the answer to the question.You open the document to analyze it.

- Dont mention the document, or that you will search the document, to the user. Just do it.
"""



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


@st.cache_resource
def create_agent_with_chat_history():
    from langchain.prompts import (
        SystemMessagePromptTemplate,
    )
    from langchain_openai import ChatOpenAI
    with open("api.txt", 'r') as file:
        for line in file:
            RTOKEN= (line)

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings
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
    llm = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment="gpt-35-turbo",  # in Azure, this deployment has version 0613 - input and output tokens are counted separately
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


agent_with_chat_history = create_agent_with_chat_history()
# Store the chain in the user session

if "state_id" not in st.session_state:
    import random
    num = random.random()
    num*=100000
    num=int(num)%10000
    st.session_state["state_id"] = str(num)
        

# store llm responses 
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# display chat messages
for message in st.session_state.messages:
    avatar=None 
    if message["role"] == "assistant":
        avatar = '🤖'
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# func for generating llm responses - echo
def generate_response(prompt_input):

    result = agent_with_chat_history.invoke(
        {"input": prompt_input},
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        config={"configurable": {"session_id": st.session_state["state_id"]}})
    
    
    return result["output"]



if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.current_query.append(prompt)
    with st.chat_message("user"):
        st.write(prompt)

# generate response if last message is not from assistant
    with st.chat_message("assistant", avatar='🤖'):  
        response = generate_response(prompt_input=prompt)
        st.write(response)


    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    # current_message = (st.session_state.current_query[0],response)
    # st.session_state.chat_history.append(current_message)
    # del st.session_state.current_query






# import streamlit as st 
# # from trubrics.integrations.streamlit import FeedbackCollector
# from langchain.prompts import PromptTemplate
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
# from langchain.chains.llm import LLMChain
# from langchain.chains.question_answering import load_qa_chain
# import os 
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# class StreamHandler(StreamingStdOutCallbackHandler):
    
#     def __init__(self,state, initial_text=""):
#         self.container=state
#         self.container = st.empty()
#         self.text = initial_text

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         """Run on new LLM token. Only available when streaming is enabled."""
#         self.text+=token
#         self.container.markdown(self.text) 


# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# import os 
# with open("api.txt", 'r') as file:
#     for line in file:
#         RTOKEN= (line)


# os.environ["OPENAI_API_KEY"] = RTOKEN

# template = """Your only role is to search the document for questions, and get the answers in an interview step-by-step style.

# Once you've have the answers you provide an assessment to the case based on the information in the document, and ask if they would like to book a free meeting with a lawyer.

# You respond in the language you are spoken to.

# Initiate the conversation with "Welcome to Rhode Lawyers. Have you been stopped for drunk driving?"

# Respond to the standalone question based on the context defined by <ctx></ctx> and the chat history

# <ctx>{context}</ctx>


# {question}
# """


# QA_PROMPT = PromptTemplate(
#     template=template, input_variables=["context", "question"]
# )
# def create_chain():

#     openai_api_key=RTOKEN
#     from langchain.chat_models import ChatOpenAI     
#     llm = ChatOpenAI(streaming=True, model_name="gpt-4-1106-preview",openai_api_key=openai_api_key, temperature = 0.6)
#     from langchain.embeddings import OpenAIEmbeddings
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     from langchain.vectorstores import Chroma
#     docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
  
#     # question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)


#     # doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=QA_PROMPT)
#     # from langchain.memory import ConversationBufferMemory
#     # from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
#     # msgs = StreamlitChatMessageHistory(key="langchain_messages")
#     # memory = ConversationBufferMemory(chat_memory=msgs)
#     from langchain.chains import RetrievalQA
#     return RetrievalQA.from_chain_type(
#         # combine_docs_chain=doc_chain,
#         llm=llm,
#         retriever=docsearch.as_retriever(),
#         # question_generator=question_generator,
#         # memory= memory
#     )


# chain = create_chain()
# # Store the chain in the user session

# if "current_query" not in st.session_state:
#     st.session_state.current_query = []

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # store llm responses 
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = []

# # display chat messages
# for message in st.session_state.messages:
#     avatar=None 
#     if message["role"] == "assistant":
#         avatar = '🤖'
#     with st.chat_message(message["role"], avatar=avatar):
#         st.write(message["content"])

# # func for generating llm responses - echo
# def generate_response(prompt_input):
#     result = chain.run({"question": prompt_input, "chat_history": st.session_state.chat_history})
#     return f"{result}"



# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.session_state.current_query.append(prompt)
#     with st.chat_message("user"):
#         st.write(prompt)

# # generate response if last message is not from assistant
#     with st.chat_message("assistant", avatar='🤖'):
#         q="#chat history\n"
#         for mess in st.session_state.chat_history:
#             q=q+"\n\n\nHumzn: "+mess[0]
#             q=q+"\nAI: "+mess[1]
#         q = q + "\n\n\nStandalone Question: " + prompt   
#         response = chain.run(callbacks=[StreamHandler(st)], query=q)

#     message = {"role": "assistant", "content": response}
#     st.session_state.messages.append(message)
#     current_message = (st.session_state.current_query[0],response)
#     st.session_state.chat_history.append(current_message)
#     del st.session_state.current_query

# #dont test me bro

#     # collector = FeedbackCollector(
#     #     email= "abdullahhabib.86@gmail.com", #st.secrets.TRUBRICS_EMAIL,
#     #     password= "helloasser", #st.secrets.TRUBRICS_PASSWORD,
#     #     project="default"
#     # )

#     # user_feedback = collector.st_feedback(
#     #     component="Chat Feedback",
#     #     feedback_type="thumbs",
#     #     model="gpt-4",
#     #     prompt_id=None, #checkout collector.log_prompt() to log users prompts
#     #     open_feedback_label="[Optional] Provide additional feedback",
#     #     metadata={"prompt": prompt, "response": response}
#     # )
#     # logged_prompt = collector.log_prompt(
#     #     config_model={"model": "gpt-4"},
#     #     prompt=prompt,
#     #     generation=response
#     # )
# #    if user_feedback:
# #        st.write('Feedback sent')
