from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from all_keys import groq_api_key
import requests
from bs4 import BeautifulSoup
import os

load_dotenv()
groq_api_key = groq_api_key

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=groq_api_key,
)

prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing the 
    following keys: company , location , role, experience, skills and description.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):    
    """
)

prompt_qa = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### USER'S QUESTION:
    {question}
    ### INSTRUCTION:
    Based on the provided scraped text, generate a precise and informative response to the user's question.
    If the answer is not found in the text, respond with "I could not find relevant information."
    ### ANSWER:
    """
)

@app.get("/fetch-job/")
def fetch_job(url: str):
    try:
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke({"page_data": page_data})
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res.content)
        return {"status": "success", "data": json_res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/ask-question/")
def ask_question(url: str, question: str):
    try:
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content
        chain_qa = prompt_qa | llm
        answer = chain_qa.invoke({"page_data": page_data, "question": question})
        return {"status": "success", "answer": answer.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/fetch-image/")
def fetch_image(url: str):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image else ""
        return {"status": "success", "image": image_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}
