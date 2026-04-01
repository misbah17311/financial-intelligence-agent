# Streamlit chat interface for the Financial Intelligence Agent
# run with: streamlit run src/ui/app.py

import streamlit as st
import sys
import os
import time

# make sure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.graph import run_query
from src.config import LLM_PROVIDER, LLM_MODEL, DATASET_DESCRIPTION

# --- page config ---
st.set_page_config(
    page_title="Financial Intelligence Agent",
    page_icon="📊",
    layout="wide",
)

# --- sidebar ---
with st.sidebar:
    st.title("📊 Financial Intelligence Agent")
    st.markdown("---")
    st.markdown("### What I can answer")
    st.markdown("""
    **Structured data (SQL):**
    - Company revenue, profit, market cap
    - Sector comparisons and trends
    - Aggregations by year/quarter
    
    **Unstructured data (News):**
    - Analyst opinions and sentiment
    - Market trend explanations
    - Company-specific news and events
    
    **Complex queries:**
    - Multi-step reasoning
    - Compare X vs Y
    - Trends over time + why
    """)

    st.markdown("---")
    st.markdown("### Example queries")
    examples = [
        "Compare average revenue of Tech vs Healthcare sectors from 2020 to 2024",
        "What are the top 5 companies by revenue in the Technology sector?",
        "What is the market saying about NVIDIA's growth prospects?",
        "How has Apple's net income changed quarter over quarter in 2023?",
        "Which sector had the worst financial performance in 2022 and why?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill_query"] = ex

    st.markdown("---")
    st.caption(f"LLM: {LLM_PROVIDER} / {LLM_MODEL}")

# --- main chat area ---
st.title("Ask me about financial markets")

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            with st.expander("Details"):
                st.json(msg["metadata"])

# check if we have a prefilled query from the sidebar
prefill = st.session_state.pop("prefill_query", None)
user_input = st.chat_input("Ask a question about financial data...") or prefill

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (planning → retrieving → analyzing → reviewing)"):
            start = time.time()
            try:
                result = run_query(user_input)
                elapsed = time.time() - start

                answer = result["answer"]
                confidence = result["confidence"]

                # display the answer
                st.markdown(answer)

                # show metadata in a collapsible section
                metadata = {
                    "confidence": confidence,
                    "retries": result.get("retries", 0),
                    "time_seconds": round(elapsed, 2),
                    "plan": result.get("plan", "")[:500],
                    "critic_verdict": result.get("critique", "")[:300],
                }
                with st.expander("Pipeline details"):
                    st.json(metadata)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                })

            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
