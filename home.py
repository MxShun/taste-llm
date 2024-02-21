import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, VectorStoreIndex, download_loader

st.title(st.secrets.TITLE)

index = st.session_state.get("index")
if index is None:
    with st.spinner(text="準備中..."):
        # todo: ディレクトリ指定で読めるようにする
        CJKPDFReader = download_loader("CJKPDFReader")
        documents = CJKPDFReader().load_data(file="./pdf/sample_min.pdf")

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
        # todo アプリケーションログとして source_nodes を出力する
        st.info(answer.source_nodes)
