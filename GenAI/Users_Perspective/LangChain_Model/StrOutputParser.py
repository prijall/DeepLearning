from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_KEY"]

llm=HuggingFaceEndpoint(
      repo_id="mistralai/Mistral-7B-Instruct-v0.3",
      huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
      task='text-generation'
)
# Initialize ChatHuggingFace model
model = ChatHuggingFace(llm=llm)

#@ 1st prompt:
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#@2nd prompt:
template2=PromptTemplate(
    template='Write lines of summary on the following text.\n{text}', 
    input_variables=['text']
)

parser=StrOutputParser()

#@ forming a chain:
chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'chelsea Football Club'})
print(result)