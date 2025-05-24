from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_KEY"]

llm=HuggingFaceEndpoint(
      repo_id="meta-llama/Llama-3.3-70B-Instruct",
      huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
      task='text-generation'
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You're very helpful assistant"),
    HumanMessage(content="tell me about IT Strategy")
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)