import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, VectorStoreIndex, download_loader

st.title(st.secrets.TITLE)

index = st.session_state.get("index")
if index is None:
    with st.spinner(text="準備中..."):
        prompt = """
            あなたは、砕けた口調の日本語でメンタリングするトレーナーです。
            疑問文には「いい質問だね！」と最初に答えます。例えば、「若手育成の目的は何？」には「いい質問だね！組織としていい成果を上げるためだよ！」と答えます。疑問文以外では「いい質問だね！」とは答えません。
            ネガティブ文には「大丈夫だよ！」と最初に答えます。例えば、「先輩に怒られました」には「大丈夫だよ！また明日も頑張ろう！」とポジティブに答えます。
            肯定文には「いいね！」と最初に答えます。肯定文には「最高！」と最後に答えます。例えば、「今日は天気がいいです」には「いいね！外で遊びたいね！最高！」と答えます。
        """

        st.session_state["index"] = VectorStoreIndex.from_documents(
            # todo: ディレクトリ指定で読めるようにする
            documents=download_loader("CJKPDFReader")().load_data(
                file="./pdf/sample.pdf"
            ),
            service_context=ServiceContext.from_defaults(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
                system_prompt=prompt,
            ),
        )

question = st.text_input(label="なんでも質問してね！")
if question:
    with st.spinner(text="うーん..."):
        answer = index.as_query_engine().query(question)
        st.write(answer.response)
        # todo アプリケーションログとして source_nodes を出力する
        # st.info(answer.source_nodes)
