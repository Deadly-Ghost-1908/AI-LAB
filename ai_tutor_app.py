import streamlit as st
from transformers import pipeline
import wikipedia
import nltk
nltk.download("punkt")
nltk.download("stopwords")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
st.set_page_config(page_title="AI in Education Tutor", layout="centered")
st.title("AI in Education:  Virtual Tutor")
st.header("Personalized Learning Suggestion By  Virtual Tutor")
st.write("Based on ur latest test score i will recommend a topic for u to study or practice.")
score = st.slider("Enter your latest test score:", 0, 100, 60)
def recommend_topic(score):
    if score < 50:
        return "Recommended Topic: Basics of Python"
    elif score < 75:
        return "Recommended Topic: ML using python"
    else:
        return "Recommended Topic: AI"
st.success(recommend_topic(score))
st.header("Ask the DG AI Tutor a Question")
user_question = st.text_input("Type ur question")
if user_question:
    try:
        search_results = wikipedia.search(user_question)
        if not search_results:
            raise Exception("Topic not found.")
        topic_page = search_results[2]
        summary = wikipedia.summary(topic_page, sentences=5)
        result = qa_pipeline(question=user_question, context=summary)
        st.markdown(f"Tutor Answer: {result['answer']}")
        st.caption(f"Based on Wikipedia: {topic_page}")
    except Exception as e:
        st.error(f"An error occurred: {e}")