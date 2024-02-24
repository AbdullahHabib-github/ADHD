from flask import Flask, request
from storage_functions import (upload_folder_to_cloud_storage_client,download_folder_from_cloud_storage_client)
from utils import create_vector_db, create_agent_with_chat_history_db
from PyPDF2 import PdfReader
import os

app = Flask(__name__)

agent_with_chat_history = None

def generate_response(prompt_input,userid):
    result = agent_with_chat_history.invoke(
        {"input": prompt_input},
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        config={"configurable": {"session_id": userid}})
    
    
    return result["output"]


@app.route('/upload_book', methods=['POST'])
def upload_pdf():
    print("lol")
    if 'pdf_file' not in request.files:
        return 400  # Error handling
    file = request.files['pdf_file']
    filename = file.filename  
    print(filename,"uploaded" )
    # username = request.username
    # bookname = request.bookname
    # persist_directory = username+"_"+bookname.replace(' ', '_')

    persist_directory = "chroma_db"

    if not os.path.exists("Books"):
        os.makedirs("Books")
    file_path = os.path.join("Books", filename)
    file.save(file_path)

    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    if os.path.exists(file_path):
      os.remove(file_path)
    try:
        docsearch = create_vector_db(text,persist_directory=persist_directory)
        upload_folder_to_cloud_storage_client(bucket_name = "booksdb", source_folder_path = persist_directory, destination_folder_path = persist_directory)
        return 200
    except:
        return 400

from pydantic import BaseModel
class ChatRequest(BaseModel):
    thread_id: str
    prompt: str
    
@app.route('/chatbot', methods=['POST'])
def chat_me():
    request_data = request.json  # Get the JSON data from the request
    chat_request = ChatRequest(**request_data)  # Parse JSON data into ChatRequest object
    thread_id = chat_request.thread_id
    input_text = chat_request.prompt
    print(thread_id)
    response = generate_response(input_text,thread_id)
    return {
  "choices": [{
    "message": {
      "role": "assistant",
      "content": response,
    },
  }],
}


@app.route('/chatbot', methods=['GET'])
def chat_me_get():
    
    # username = request.username
    # bookname = request.bookname
    # persist_directory = username+"_"+bookname.replace(' ', '_')
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
      download_folder_from_cloud_storage_client(bucket_name="booksdb", prefix=persist_directory,destination_base_dir="")

    import uuid
    thread_id = str(uuid.uuid4())
    global agent_with_chat_history 
    agent_with_chat_history = create_agent_with_chat_history_db(persist_directory=persist_directory)

    response = generate_response("Start",thread_id)

   
    response =  {
  "choices": [{
    "message": {
      "role": "assistant",
      "content": response,
    },
  }],
}
    return {"Response": response, "Thread_id" : thread_id}



if __name__ == '__main__':
    app.run(debug=True)  # Use debug mode during development
