import os
import onnxruntime

os.environ["TRANSFORMERS_NO_TF"] = "1"

from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")
import streamlit as st
import os
import uuid
import json
import pandas as pd
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

st.title("📧 AI Cold Email Generator")
st.markdown("Produce a professional, personalized cold email for any job posting.")

groq_api_key = st.text_input("GROQ API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

job_url = st.text_input("Paste the job posting URL", value="https://www.google.com/about/careers/applications/jobs/results/117034903765164742-senior-software-engineer-android-partner")
uploaded_file = st.file_uploader("Upload your portfolio .csv (with 'Techstack' and 'Links')", type=['csv'])
run_button = st.button("✨ Generate Email")

if run_button:
    if not groq_api_key or not uploaded_file or not job_url:
        st.error("Please provide the job URL, your GROQ API key, and upload your portfolio CSV.")
    else:
        try:
            # 1. Load LLM
            llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile"
            )

            # 2. Scrape & Extract Job Description
            loader = WebBaseLoader(job_url)
            page_data = loader.load()[0].page_content

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
""")
            chain_extract = prompt_extract | llm | parser
            jobs = chain_extract.invoke({"page_data": page_data})
            first_job = jobs[0]

            # 3. Portfolio Vector Store
            df = pd.read_csv(uploaded_file)
            # Use a unique temp directory for Chroma
            chroma_path = f".chroma_tmp_{uuid.uuid4().hex}"
            client = chromadb.PersistentClient(chroma_path)
            collection = client.get_or_create_collection(name="portfolio")
            if not collection.count():
                for _, row in df.iterrows():
                    collection.add(
                        documents=[str(row["Techstack"])],
                        metadatas={"links": row["Links"]},
                        ids=[str(uuid.uuid4())]
                    )
            links = collection.query(query_texts=first_job['skills'], n_results=2).get('metadatas', [])
            link_list = [item["links"] for item in links[0]] if links else ["https://example.com/default-portfolio"]
            link_list_str = "\n".join(f"- {link}" for link in link_list)

            # 4. Generate Email
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
            email_str = getattr(mail_msg, 'content', mail_msg)

            st.markdown("### 📩 Generated Cold Email")
            st.write(email_str)
        except Exception as e:
            st.error(f"Error: {e}")
