# streamlit run main.py --server.port 8502

import streamlit as st
from langchain.schema import Document

import re
import os
import json
from typing import Literal, List
from typing_extensions import TypedDict

# --- asyncio patch for Streamlit ---
import nest_asyncio
nest_asyncio.apply()
# ------------------------------------
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader

import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

# --- Load Environment Variables and Configure API Key ---
load_dotenv()
gemini_key = os.getenv("GOOGLE_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Explicitly configure the Gemini API key
if gemini_key:
    genai.configure(api_key=gemini_key)
else:
    # Display an error in the app and stop execution if the key is missing.
    st.error("Gemini API key not found. Please set the 'gemini' key in your .env file.")
    st.stop()

# # --- Helper Functions (Unchanged) ---
# def load_docs_from_json(file_path):
#     loaded_docs = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             loaded_docs.append(Document(**data))
#     return loaded_docs
def load_docs_from_json(file_path):
    # Load the entire JSON array at once
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate that data is a list
    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of objects")

    # Convert each dict to a LangChain Document
    loaded_docs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a JSON object")
        
        loaded_docs.append(
            Document(
                page_content=item.get("page_content", ""),
                metadata={
                    "document_title": item.get("document_title"),
                    "document_type": item.get("document_type"),
                    "category": item.get("category"),
                    "page_number": item.get("page_number"),
                }
            )
        )

    return loaded_docs




def clean_arabic_numbers(text):
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    return text


@st.cache_resource
def initialize_components():
    """Initializes all the major components for the RAG system."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=gemini_key)
    documents = load_docs_from_json("docs.json")

    vector_db = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db_new",
    )
    dense_retriever = vector_db.as_retriever(search_kwargs={'k': 10})
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME,
                                 api_key=gemini_key,
                                 temperature=0,
                                 max_output_tokens=2048,
                                 timeout=None,
                                 max_retries=3)
    return retriever, llm

retriever, llm = initialize_components()

# --- Pydantic Models, Prompts, and Chains (Unchanged) ---
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(..., description="Route to web search or a vectorstore.")
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

system_router = "You are an expert at routing a user question to a vectorstore or web search. The vectorstore contains documents related to income tax laws (قانون الضريبة على الدخل). Use the vectorstore for questions on these topics. Otherwise, use web-search."
route_prompt = ChatPromptTemplate.from_messages([("system", system_router), ("human", "{question}")])
structured_llm_router = llm.with_structured_output(RouteQuery)
question_router = route_prompt | structured_llm_router

rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

system_grader = "You are a grader assessing if an answer resolves a question. Give a binary score 'yes' or 'no'."
answer_prompt = ChatPromptTemplate.from_messages([("system", system_grader), ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")])
structured_llm_grader = llm.with_structured_output(GradeAnswer)
answer_grader = answer_prompt | structured_llm_grader

system_rewriter = "You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval. Provide only the new question."
re_write_prompt = ChatPromptTemplate.from_messages([("system", system_rewriter), ("human", "Initial question: \n\n {question} \n Formulate an improved question.")])
question_rewriter = re_write_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

# --- Graph State and Nodes (Unchanged) ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

def retrieve_node(state):
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

def generate_node(state):
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}

def transform_query_node(state):
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}

def web_search_node(state):
    docs = web_search_tool.invoke({"query": state["question"]})
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = [Document(page_content=web_results)]
    return {"documents": web_results_doc, "question": state["question"]}

# --- Conditional Edges (Unchanged) ---
def route_question_edge(state):
    source = question_router.invoke({"question": state["question"]})
    return "web_search" if source.datasource == 'web_search' else "vectorstore"

def grade_generation_edge(state):
    score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
    return "useful" if score.binary_score == "yes" else "not useful"

# --- Build and Compile Graph (Cached) ---
@st.cache_resource
def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.set_conditional_entry_point(route_question_edge, {"web_search": "web_search", "vectorstore": "retrieve"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge('retrieve', "generate")
    workflow.add_edge('transform_query', 'retrieve')
    workflow.add_conditional_edges('generate', grade_generation_edge, {"useful": END, "not useful": "transform_query"})
    return workflow.compile()

graph_app = create_graph()

import streamlit as st
import time

st.markdown("""
<style>
/* Make main content RTL */
section[data-testid="stMain"] {
    direction: rtl;
    text-align: right;
    font-family: 'Tahoma', 'Segoe UI', 'Arial', sans-serif;
}


/* Chat messages: content should be RTL */
div[data-testid="stChatMessageContent"] {
    direction: rtl;
    text-align: right;
}


/* Optional: Keep sidebar LTR (for English buttons like "Clear") */
section[data-testid="stSidebar"] {
    direction: ltr;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)


# === 2. Header & Intro (now naturally RTL due to CSS) ===
st.title("Travel assistant chatbot")
st.logo("depi.png")
st.markdown(
  "Ask all your questions regarding Egyptian tourism",
    unsafe_allow_html=True
)

# === 3. Initialize chat history ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === 4. Display chat history ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # No need for manual dir='rtl' — CSS handles it!
        st.markdown(message["content"])


# === 5. User input & response ===
if user_question := st.chat_input("Write your answer here. "):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = graph_app.invoke({"question": user_question})
                final_generation = response.get("generation", "Sorry, couldn't answer.")

                # Stream the response word by word (with RTL preserved by CSS)
                placeholder = st.empty()
                streamed_text = ""
                for chunk in final_generation.split(" "):
                    streamed_text += chunk + " "
                    placeholder.markdown(streamed_text)
                    time.sleep(0.08)

                # Save final response
                st.session_state.messages.append({"role": "assistant", "content": final_generation})

            except Exception as e:
                error_msg = f"Erorr: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})