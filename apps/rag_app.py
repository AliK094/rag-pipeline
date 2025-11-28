import textwrap

import streamlit as st

from rag_pipeline.query.answer_engine import RAGAnswerEngine

@st.cache_resource(show_spinner=True)
def get_answer_engine() -> RAGAnswerEngine:
    return RAGAnswerEngine(
        index_dir="data/indexes/chroma/hotpot",
        collection_name="hotpot_corpus",
        top_k=5,
        model_name_embed="BAAI/bge-base-en-v1.5",
        device_embed="mps",
        batch_size_embed=128,
        llm_model="llama3.1:8b",
        max_new_tokens=512,
        temperature=0.2,
    )


def main() -> None:
    st.set_page_config(
        page_title="RAG Hotpot QA Explorer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 RAG Hotpot QA Explorer")

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=5, step=1)
    show_metadata = st.sidebar.checkbox("Show metadata", value=True)
    show_raw_content = st.sidebar.checkbox("Show full chunk content", value=False)

    st.write("Ask a question, see the LLM answer, and inspect which chunks were used.")

    query = st.text_input(
        "Query",
        value="Was Artie Ziff a good prom date for Marge?",
        placeholder="Ask a question about the Hotpot corpus...",
    )

    answer_tab, debug_tab = st.tabs(["💬 RAG Answer", "🧠 Retriever Debug"])

    engine = get_answer_engine()

    # ---------------- TAB 1: RAG ANSWER (LLM) ----------------
    with answer_tab:
        st.subheader("LLM Answer (RAG)")

        ask_clicked = st.button("Ask LLM", type="primary", key="ask_llm_button")

        if ask_clicked and query.strip():
            with st.spinner("Running retrieval + Ollama..."):
                rag_result = engine.answer(query, k=top_k)

            answer = rag_result.get("answer", "").strip()
            contexts = rag_result.get("contexts", [])

            if answer:
                st.markdown("### 🧠 Answer")
                st.markdown(answer)
            else:
                st.warning("The LLM did not return an answer.")

            if contexts:
                st.markdown("### 📚 Supporting Contexts")
                for ctx in contexts:
                    st.markdown("---")
                    rank = ctx.get("rank")
                    title = ctx.get("title") or "(no title)"
                    doc_id = ctx.get("doc_id")
                    source_path = ctx.get("source_path")
                    doc_type = ctx.get("doc_type")
                    chunk_id = ctx.get("chunk_id")
                    content = ctx.get("content") or ""

                    header_bits = []
                    if rank is not None:
                        header_bits.append(f"Rank {rank}")
                    if doc_type:
                        header_bits.append(f"`{doc_type}`")
                    if header_bits:
                        st.markdown("**" + "  ·  ".join(header_bits) + "**")

                    st.markdown(f"**Title:** {title}")
                    if doc_id is not None:
                        st.markdown(f"**Doc ID:** `{doc_id}`")
                    if chunk_id is not None:
                        st.markdown(f"**Chunk ID:** `{chunk_id}`")

                    if show_metadata and source_path:
                        st.caption(f"📁 `{source_path}`")

                    if not show_raw_content:
                        content = textwrap.shorten(content, width=400, placeholder=" ...")

                    with st.expander("Show content", expanded=show_raw_content):
                        st.write(content)
            else:
                st.info("No supporting contexts returned.")

        elif ask_clicked and not query.strip():
            st.warning("Please enter a query before asking the LLM.")

    # ---------------- TAB 2: RETRIEVER DEBUG ----------------
    with debug_tab:
        st.subheader("Retriever Debug View")

        search_clicked = st.button("Search Retriever Only", key="search_retriever_button")

        if search_clicked and query.strip():
            with st.spinner("Searching index..."):
                results = engine.retrieve_with_metadata(query, k=top_k)

            if not results:
                st.warning("No results found.")
            else:
                for res in results:
                    st.markdown("---")
                    rank = res["rank"]
                    title = res.get("title") or "(no title)"
                    doc_id = res.get("doc_id")
                    source_path = res.get("source_path")
                    doc_type = res.get("doc_type")
                    chunk_id = res.get("chunk_id")

                    st.markdown(f"**Rank {rank}**  ·  `{doc_type or 'unknown'}`")
                    st.markdown(f"**Title:** {title}")

                    if doc_id is not None:
                        st.markdown(f"**Doc ID:** `{doc_id}`")
                    if chunk_id is not None:
                        st.markdown(f"**Chunk ID:** `{chunk_id}`")

                    if show_metadata and source_path:
                        st.caption(f"📁 `{source_path}`")

                    content = res["content"] or ""
                    if not show_raw_content:
                        content = textwrap.shorten(content, width=400, placeholder=" ...")

                    st.markdown("**Content:**")
                    st.write(content)

        elif search_clicked and not query.strip():
            st.warning("Please enter a query before searching.")


if __name__ == "__main__":
    main()
