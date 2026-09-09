import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd

from config import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_vector_db(embed_func):
    try:
        return Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embed_func)
    except Exception as e:
        st.error(
            f"Error loading ChromaDB. Make sure '{config.CHROMA_PERSIST_DIRECTORY}' exists and is populated. Error: {e}")
        st.stop()  # Stop the app if DB cannot be loaded

@st.cache_resource(show_spinner=False)
def get_chat_model():
    return ChatGroq(model=config.CHAT_MODEL_NAME,
                    temperature=0.0,
                    max_tokens=400)

embeddings = get_embedding_model()
vectordb = get_vector_db(embeddings)
model = get_chat_model()


def call_model(state: MessagesState):
    system_prompt = (
        "You are a question-answering assistant for the provided documents. "
        "Answer the question ONLY using the retrieved context. "
        "Do not use outside knowledge. "
        "If the answer cannot be found in the context, say: "
        "'I don't know based on the provided documents.' "
        "Keep the answer concise, using a maximum of three sentences."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def get_langgraph_app():
    workflow = StateGraph(state_schema=MessagesState)

    workflow.add_node('model', call_model)
    workflow.add_edge(START, 'model')

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


app = get_langgraph_app()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "streamlit_chat_session"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a any question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                docs = vectordb.similarity_search_with_score(prompt, k=3)

                _docs = pd.DataFrame(
                    [
                        [prompt,
                         doc.metadata.get("page"),
                         doc.page_content,
                         doc.metadata.get("source"),
                         score
                         ]
                        for doc, score in docs

                    ],
                    columns=["Question", "Page_No", "Content", "Source", "Score"]
                )

                context = "\n\n --- \n\n".join(_docs["Content"])

                human_msg = HumanMessage(content=f"Context : {context} \n\n Question : {prompt}")
                result = app.invoke({"messages": [human_msg]},
                                    config={"configurable": {"thread_id": st.session_state.thread_id}})

                ai_response = result["messages"][-1].content

                source_doc = _docs["Source"][0] if not _docs.empty and "Source" in _docs else "N/A"

                # Get unique top 3-page numbers and convert to string
                top_page_no = _docs["Page_No"].drop_duplicates().head().astype(str).tolist()
                page_no = ",".join(top_page_no) if top_page_no else "N/A"

                final_response = f"{ai_response} \n\n **Source Document** : {source_doc} \n **Reference Page No** : {page_no}"

                st.markdown(final_response)

                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I encountered an error. Please try again."})
