import streamlit as st

from rag import PDF_PATHS, ask_question, describe_corpus, get_vectorstore


st.set_page_config(
    page_title="RAG Chat UI",
    page_icon=":material/article:",
    layout="wide",
)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask a question about the indexed document and I'll answer from the retrieved context.",
            }
        ]


def render_sidebar() -> int:
    with st.sidebar:
        st.title("RAG Controls")
        st.caption("Streamlit UI for your document question-answering pipeline.")

        st.markdown(f"**Corpus**: `{len(PDF_PATHS)} PDFs`")
        st.caption(describe_corpus())
        k_value = st.slider("Retrieved chunks", min_value=1, max_value=8, value=3)

        if st.button("Warm up vector store", use_container_width=True):
            with st.spinner("Loading embeddings and vector store..."):
                get_vectorstore()
            st.success("Vector store is ready.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Ask a fresh question whenever you're ready.",
                }
            ]

    return k_value


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                with st.expander("Retrieved context"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(f"**Chunk {index}**")
                        st.write(source)


def main() -> None:
    initialize_state()

    st.title("Document RAG Chat")
    st.caption("Ask questions against your Chroma-backed PDF knowledge base.")

    k_value = render_sidebar()
    render_chat_history()

    prompt = st.chat_input("Ask something about the document...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = ask_question(prompt, k=k_value)
        st.markdown(result["answer"])

        sources = [doc.page_content for doc in result["context"]]
        with st.expander("Retrieved context"):
            for index, source in enumerate(sources, start=1):
                st.markdown(f"**Chunk {index}**")
                st.write(source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
        }
    )
    


if __name__ == "__main__":
    main()
