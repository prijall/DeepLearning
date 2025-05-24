from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_KEY"]

llm=HuggingFaceEndpoint(
      repo_id="HuggingFaceH4/zephyr-7b-beta",
      huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
      task='task-generation')


model = ChatHuggingFace(llm=llm)

chat_template=ChatPromptTemplate.from_messages([
    ('system', 'you are very sweet agent'),
    MessagesPlaceholder(variable_name='chat_history'), 
    ('human', '{query}')
])

chat_history=[]

#@ Loading chat history:
with open('chat_history.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith("HumanMessage(content="):
            content = line[len('HumanMessage(content="'):-2]  # remove prefix & trailing ")
            chat_history.append(HumanMessage(content=content))
        elif line.startswith("AIMessage(content="):
            content = line[len('AIMessage(content="'):-2]
            chat_history.append(AIMessage(content=content))


#@ Creating a prompt:
prompt=chat_template.invoke({'chat_history':chat_history, 'query': 'Where is my refund'})
response=llm.invoke(prompt)
print(response)