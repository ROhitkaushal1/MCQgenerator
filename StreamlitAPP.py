import os
import sys
sys.path.append("src")
import json
import PyPDF2
import traceback
import pandas as pd
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import evaluation_chain
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback


# Load expected RESPONSE_JSON structure
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Title
st.title("MCQs Creator Application with LangChain 🧠📝")
# --- User Input Form ---
with st.form("user input"):
    uploaded_file = st.file_uploader("Upload PDF or Text")
    mcq_count = st.number_input("No of MCQ's", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")
    button = st.form_submit_button("Create MCQs")

# --- On Submit ---
if button and uploaded_file and mcq_count and subject and tone:
    with st.spinner("Generating MCQs..."):
        try:
            # Step 1: Extract text from uploaded file
            text = read_file(uploaded_file)

            # Step 2: Call LangChain Evaluation Chain
            with get_openai_callback() as cb:
                response = evaluation_chain.invoke({
                    "text": text,
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON)
                })

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Something went wrong while generating MCQs.")
        else:
        # Step 3: Show token usage and cost
            st.write(f"🧾 **Total Tokens**: {cb.total_tokens}")
            st.write(f"📩 **Prompt Tokens**: {cb.prompt_tokens}")
            st.write(f"📤 **Completion Tokens**: {cb.completion_tokens}")
            st.write(f"💰 **Total Cost (USD)**: ${cb.total_cost:.6f}")
            if isinstance(response, dict):
                quiz = response.get("quiz", None)
            quiz = response.get("quiz", None)
            if quiz is not None:
                table_data = get_table_data(quiz)
                
                if table_data is not None:
                    df = pd.DataFrame(table_data)
                    df.index = df.index + 1
                    st.table(df)

                    # Display the review in a text box as well
                    st.text_area(label="Review", value=response.get("review", ""), height=150)
                else:
                    st.error("Error in the table data")
            else:
                st.write(response)
                st.subheader("Raw Quiz Output")
                st.code(response["quiz"])