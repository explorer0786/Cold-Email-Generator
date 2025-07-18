import os
import onnxruntime

os.environ["TRANSFORMERS_NO_TF"] = "1"

from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

import uuid, json
import pandas as pd
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ─── 0. Load LLM ────────────────────────────
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# ─── 1. Scrape & Extract Job Description ────
loader = WebBaseLoader(
    "https://www.google.com/about/careers/applications/jobs/results/"
    "117034903765164742-senior-software-engineer-android-partner"
)
page_data = loader.load()[0].page_content
# Force model to start with JSON:
parser = JsonOutputParser()
prompt_extract = PromptTemplate(
    input_variables=["page_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
{page_data}

You are given the text of a careers page.
Extract each job posting and return an ARRAY of JSON objects with:
"role", "experience", "skills", "description". 

{format_instructions}
"""
)

chain_extract = prompt_extract | llm | parser     # guaranteed parsed JSON
jobs = chain_extract.invoke({"page_data": page_data})
first_job = jobs[0]

# ─── 2. Build / Query Portfolio Vector Store ─
df = pd.read_csv("my_portfolio.csv")
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])
# Query portfolio by skills
links = collection.query(query_texts=first_job['skills'], n_results=2).get('metadatas', [])
link_list = [item["links"] for item in links[0]] if links else ["https://example.com/default-portfolio"]
link_list_str = "\n".join(f"- {link}" for link in link_list)

# ─── 3. Generate Professional Cold Email ────
prompt_email = PromptTemplate.from_template("""
### JOB DESCRIPTION:
{job_description}

You are Hardik, Business Development Executive at Jain's Technology — an AI and software consulting firm.

Write a professional, personalized cold email pitching Jain's Technology for this role.
Mention these portfolio items: {link_list}

Only output the email! No preamble, no explanation, no code, no markdown.
""")

chain_email = prompt_email | llm
mail_msg = chain_email.invoke({
    "job_description": json.dumps(first_job, indent=2),
    "link_list": link_list_str
})

# PRINT OUTPUT (with safeguards)
print("\n📩 Generated Cold Email:\n")
print(getattr(mail_msg, 'content', mail_msg))   # Will print .content if exists, else just the object

# Optional: For debugging, un-comment to see object structure
# print("\nDEBUG:", dir(mail_msg), repr(mail_msg))