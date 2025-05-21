from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#@ Loading API Key from Environment
load_dotenv()

#accessing model:
model=ChatOpenAI(model='o1-mini', temperature=0.5)
result=model.invoke("Give me top 5 highlight of chelsea's premier league matches")
print(result.content)
