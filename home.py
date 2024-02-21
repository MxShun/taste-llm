import streamlit as st
import tempfile
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.readers.file.docs_reader import PDFReader

st.title(st.secrets.TITLE)

index = st.session_state.get("index")

if index is None:
    with st.spinner(text="準備中..."):
        documents = PDFReader().load_data(file=Path("./pdf/sample_min.pdf"))

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(
            documents=documents, service_context=service_context
        )
        st.session_state["index"] = index


question = st.text_input(label="質問")

if question:
    with st.spinner(text="考え中..."):
        query_engine = index.as_query_engine()
        answer = query_engine.query(question)
        st.write(answer.response)
        st.info(answer.source_nodes)
