from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_KEY"]

llm=HuggingFaceEndpoint(
      repo_id="meta-llama/Llama-3.3-70B-Instruct",
      huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
      task='text-generation'
)
# Initialize ChatHuggingFace model
model = ChatHuggingFace(llm=llm) 

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))