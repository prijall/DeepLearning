from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_KEY"]

llm=HuggingFaceEndpoint(
      repo_id="meta-llama/Llama-3.3-70B-Instruct",
      huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
      task='text-generation'
)
# Initialize ChatHuggingFace model
model = ChatHuggingFace(llm=llm)  # Replace "gpt2" with your preferred HF model

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
