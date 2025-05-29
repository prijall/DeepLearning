from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
import os
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_KEY"]

llm=HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal["positive", "negative"]=Field(description='"Give the sentiment')

parser2=PydanticOutputParser(pydantic_object=Feedback)

#@ Prompt1:
prompt1=PromptTemplate(
    template='classify the following text into either positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain= prompt1 | model | parser2

prompt2=PromptTemplate(
    template='write appropriate respose to this pos feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='write an appropriate response for this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain=RunnableBranch(
    (lambda x: x.sentiment=='positive', prompt2 | model | parser),
    (lambda x: x.sentiment=='negative', prompt3 | model | parser), 
    RunnableLambda(lambda x: "couldnot find")
)

chain=classifier_chain | branch_chain

print(chain.invoke({'feedback':'This phone is beautiful'}))

chain.get_graph().print_ascii()