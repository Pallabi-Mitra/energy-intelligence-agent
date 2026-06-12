import os
import streamlit as st
import urllib3
urllib3.disable_warnings()
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="GridCore Energy Assistant",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ GridCore Energy Intelligence Assistant")
st.caption("Ask anything about energy markets, electricity generation, and grid operations.")

groq_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password",
    placeholder="gsk_..."
)

@st.cache_resource
def build_rag_chain(api_key):
    os.environ["GROQ_API_KEY"] = api_key

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    hardcoded_text = """
    Electricity in the US is generated from fossil fuels, nuclear energy, and renewable sources.
    Fossil fuels include coal, natural gas, and petroleum.
    Renewable sources include wind, solar, hydro, geothermal, and biomass.
    Most electricity is generated using steam turbines.
    Transmission lines carry high voltage electricity over long distances.
    Substations use transformers to step down voltage for distribution.
    Distribution lines deliver lower voltage electricity to homes and businesses.
    Local electric utilities operate the distribution network.
    Cold snaps can reduce generator capacity due to equipment stress and fuel supply issues.
    Unplanned outages occur when generators fail unexpectedly.
    Day ahead markets allow electricity to be bought and sold one day before delivery.
    Grid operators balance supply and demand in real time to maintain reliability.
    """

    all_docs = [Document(page_content=hardcoded_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
    You are an expert energy market assistant at GridCore Systems.
    Answer the question using only the context provided below.
    If the answer is not in the context, say "I don't have enough information on that."
    Be concise and technical.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not groq_key:
    st.info("Enter your Groq API key in the sidebar to start.")
else:
    with st.spinner("Loading knowledge base..."):
        chain = build_rag_chain(groq_key)

    if question := st.chat_input("Ask about energy markets..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )