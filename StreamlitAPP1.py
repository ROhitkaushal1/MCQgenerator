import os
import sys
import json
import PyPDF2
import traceback
import pandas as pd
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import evaluation_chain
from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
import ast  # For safely parsing strings into Python objects

sys.path.append("src")
# Load expected RESPONSE_JSON structure
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# App Title
st.title("MCQs Creator Application with LangChain 🧠📝")

# --- User Input Form ---
with st.form("user input"):
    uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt"])
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
            st.error("❌ Something went wrong while generating MCQs.")
        else:
            # Step 3: Show token usage and cost
            st.write(f"🧾 **Total Tokens**: {cb.total_tokens}")
            st.write(f"📩 **Prompt Tokens**: {cb.prompt_tokens}")
            st.write(f"📤 **Completion Tokens**: {cb.completion_tokens}")
            st.write(f"💰 **Total Cost (USD)**: ${cb.total_cost:.6f}")

            try:
                # Parse response if it's a string
                if isinstance(response, str):
                    response = json.loads(response)
                

                quiz = response.get("quiz", None)

                # Handle case where quiz is a stringified dict
                if isinstance(quiz, str):
                    quiz = ast.literal_eval(quiz)
                    quiz = quiz.get('quiz')
    
                # Get questions
                quiz_data = quiz.get("questions", []) if isinstance(quiz, dict) else []
              


                if quiz_data:
                    table_data = get_table_data(quiz_data)
                    if isinstance(table_data, list) and table_data:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)

                        # Display the review
                        st.text_area(label="Review", value=response.get("review", ""), height=150)
                    else:
                        st.error("⚠️ Could not format table data properly.")
                else:
                    # st.error("⚠️ No quiz questions found.")
                    st.write("🔍 Full quiz content:")
                    st.json(quiz)

            except Exception as e:
                st.error("⚠️ Error processing quiz data.")
                traceback.print_exception(type(e), e, e.__traceback__)
                st.write("🔍 Raw response:")
                st.json(response)