import streamlit as st 

from utils import setup

def generate_response(prompt_input,userid):

    result = agent_with_chat_history.invoke(
        {"input": prompt_input},
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        config={"configurable": {"session_id": userid}})
    
    
    return result["output"]

        

# app title 
# st.set_page_config(page_title="Chat with PDF")
st.title("📝 Chat with your Books")

if "state_id" not in st.session_state:
    import random
    num = random.random()
    num*=100000
    num=int(num)%10000
    st.session_state["state_id"] = str(num)
user_id = st.session_state["state_id"] 
uploaded_file = st.file_uploader("Upload a your book", type=("pdf"), label_visibility='hidden')#, on_change=setup)

# Save the uploaded file to the session state
if uploaded_file is not None and 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = uploaded_file
    setup()  # Call the setup function


if uploaded_file is not None: 
    
    # load the chain
    agent_with_chat_history = st.session_state.agent_with_chat_history

    # store llm responses 
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": f"I just read the **{uploaded_file.name[:-4]}**. Ask me anything about it."}]

    # display chat messages
    for message in st.session_state.messages:
        avatar=None 
        if message["role"] == "assistant":
            avatar = '🤖'
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])



    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.session_state.current_query.append(prompt)
        with st.chat_message("user"):
            st.write(prompt)

    # generate response if last message is not from assistant
        with st.chat_message("assistant", avatar='🤖'):  
            response = generate_response(prompt_input=prompt, userid=user_id)
            st.write(response)
        
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)    
        st.session_state.agent_with_chat_history = agent_with_chat_history
