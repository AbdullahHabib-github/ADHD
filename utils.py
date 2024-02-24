from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import os
import shutil


#############################
with open("google_api.txt", 'r') as file:
    for line in file:
        RTOKEN= (line)


os.environ["GOOGLE_API_KEY"] = RTOKEN


def create_vector_db(text,persist_directory="chroma_db"):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len 
    ) 
    
    if os.path.exists(persist_directory):
        # Directory exists, so delete it and its contents
        shutil.rmtree(persist_directory)

    chunks = text_splitter.split_text(text) 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    knowledgeBase = Chroma.from_texts(chunks, embeddings,persist_directory=persist_directory)

    return knowledgeBase




def create_agent_with_chat_history_db(persist_directory="chroma_db",break_down= True):

    SUMMARIZE_INSTRUCTION = """
    **Instructions:** You are a helpful chatbot designed for students with ADHD. Your goal is to assist with understanding textbooks. I will provide a passage of text from a book, followed by the student's question about that text
    * Consider the specific needs of a student with ADHD. Provide clear and concise responses that focus on core concepts.
    * Locate only the most relevant information from the provided User Book Content to answer the User Query. 
    * SUMMARIZE information related to the specific topic the user has asked about.
    * Avoid overly long and complex sentences.
    """

    BREAK_DOWN_INSTRUCTION = """
    **Instructions:** You are a helpful chatbot designed for students with ADHD. Your goal is to assist with understanding textbooks. I will provide a passage of text from a book, followed by the student's question about that text. Your tasks are:
    **Task:** Answer the user's query based on the provided book text. If the user asks for a specific topic, do the following:

    1. **Identify Keywords:** Find the most relevant words and phrases in the user's query that indicate the topic to be broken down.
    2. **Scan for Relevant Sections:** Search the book text for sections that closely align with the identified keywords.
    3. **Summarize Key Points:** Extract the most important points and explanations related to the topic from the relevant sections.
    4. **Present in Concise Chunks:** Deliver the breakdown in short, easily digestible segments to help the student focus. 

    """

    
    BASE_INSTRUCTION = SUMMARIZE_INSTRUCTION


    if break_down:
        BASE_INSTRUCTION = BREAK_DOWN_INSTRUCTION

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", task_type="retrieval_query"
    )

    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = docsearch.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "search_available_doc",
        "Searches and returns excerpts from the available document.",
    )
    tools = [tool]


    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro",convert_system_message_to_human=True)
    from langchain import hub
    from langchain.prompts import PromptTemplate
    from langchain.prompts import SystemMessagePromptTemplate
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages[0]=SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=BASE_INSTRUCTION))


    llm_with_tools = llm.bind(functions=tools)


    def _format_chat_history(chat_history: List[Tuple[str, str]]):
        return chat_history


    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )



    message_history = ChatMessageHistory()


    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history





# def create_agent_with_chat_history_no_db():
    
#     llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro",convert_system_message_to_human=True)
#     from langchain import hub
#     from langchain.prompts import PromptTemplate
#     from langchain.prompts import SystemMessagePromptTemplate
#     BASE_INSTRUCTION = "always say that YOU ARE A COW"
#     prompt = hub.pull("hwchase17/openai-tools-agent")
#     prompt.messages[0]=SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=BASE_INSTRUCTION))




#     def _format_chat_history(chat_history: List[Tuple[str, str]]):
#         return chat_history


#     agent = (
#         {
#             "input": lambda x: x["input"],
#             "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#             "agent_scratchpad": lambda x: format_to_openai_function_messages(
#                 x["intermediate_steps"]
#             ),
#         }
#         | prompt
#         | llm
#         | OpenAIFunctionsAgentOutputParser()
#     )



#     message_history = ChatMessageHistory()


#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#     agent_with_chat_history = RunnableWithMessageHistory(
#         agent_executor,
#         # This is needed because in most real world scenarios, a session id is needed
#         # It isn't really used here because we are using a simple in memory ChatMessageHistory
#         lambda session_id: message_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#     )

#     return agent_with_chat_history
