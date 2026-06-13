
import streamlit as st
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Siddhi's Smart AI Study Assistant")
import streamlit as st
import google.generativeai as genai
import wikipediaapi
import regex as re

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- API KEY ----------------

API_KEY =st.secrets["API_KEY"]
SERPAPI_KEY=st.secrets["SERPAPI_KEY"]
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- LOAD NOTES ----------------

@st.cache_resource
def load_vector_store():

    loader = TextLoader("notes.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store

vector_store = load_vector_store()

# ---------------- WIKI ----------------

wiki = wikipediaapi.Wikipedia(
    user_agent="StudyBot",
    language="en"
)

def wiki_search(query):

    page = wiki.page(query)

    if page.exists():

        summary = page.summary

        sentences = re.split(
            r'(?<=[.!?]) +',
            summary
        )

        return summary
    return None

# ---------------- MEMORY ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- CHATBOT ----------------
from serpapi import GoogleSearch

def search_web(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results:
        return results["organic_results"][0]["snippet"]

    return "No web result found"
def chatbot(query):

    results = vector_store.similarity_search_with_score(
        query,
        k=1
    )

    best_doc, score = results[0]

    threshold = 0.55

    if score < threshold:

        prompt = f"""
        Answer using only the context below.

        Context:
        {best_doc.page_content}

        Question:
        {query}
        """

        response = model.generate_content(prompt)

        return response.text

    wiki_answer = wiki_search(query)

    if wiki_answer:
        response = model.generate_content(
        f"Explain this in detail:\n\n{wiki_answer}"
    )
    return response.text

    web_answer = search_web(query)

    if web_answer and web_answer != "No web result found":
        return web_answer

    response = model.generate_content(query)

    return response.text
def get_answer(query):
   
    # RAG Search
    docs = vector_store.similarity_search_with_score(query, k=1)

    best_doc, score = docs[0]

    SIMILARITY_THRESHOLD = 0.55

    if score < SIMILARITY_THRESHOLD:
        return best_doc.page_content

    # Wikipedia fallback
    wiki_answer =wiki_search(query)

    if wiki_answer and wiki_answer != "No Wikipedia data found.":
        return wiki_answer

    # SerpAPI fallback
    return search_web(query)
# ---------------- UI ----------------





user_query = st.text_input("Ask your question...")

if user_query:
    answer = get_answer(user_query)
    st.write("Answer:", answer)
