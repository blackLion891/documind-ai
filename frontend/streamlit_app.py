
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind AI")
st.caption("Your intelligent document question-answering assistant")

st.divider()

# Backend connection
st.sidebar.header("Backend Status")

try:
    health_response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=5
    )

    if health_response.status_code == 200:
        st.sidebar.success("Backend connected")
    else:
        st.sidebar.error("Backend is unavailable")

except requests.RequestException:
    st.sidebar.error("Cannot connect to FastAPI")

# Document upload
st.header("📤 Upload Documents")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX, or TXT file",
    type=["pdf", "docx", "txt"]
)

if st.button("Upload Document", type="primary"):
    if uploaded_file is None:
        st.warning("Please select a document first.")
    else:
        try:
            with st.spinner("Uploading and processing document..."):
                response = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=300
                )

            if response.status_code == 200:
                st.success("Document uploaded successfully!")
                st.json(response.json())

            else:
                st.error(f"Upload failed: {response.text}")

        except requests.RequestException as error:
            st.error(f"Connection error: {error}")

st.divider()

# Uploaded document listing
st.header("📂 Uploaded Documents")

if st.button("Refresh Documents"):
    try:
        response = requests.get(
            f"{API_BASE_URL}/documents/",
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()

            st.write(
                f"Total documents: {data['total_documents']}"
            )

            if data["documents"]:
                st.dataframe(
                    data["documents"],
                    use_container_width=True
                )
            else:
                st.info("No documents uploaded yet.")

        else:
            st.error("Unable to retrieve documents.")

    except requests.RequestException as error:
        st.error(f"Connection error: {error}")
st.divider()

# Document Question Answering
st.header("💬 Chat with Your Documents")

st.caption("Ask questions based on your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(
                        f"**File:** {source.get('file_name', 'Unknown')}"
                    )
                    st.write(
                        f"**Page:** {source.get('page_number', 'Unknown')}"
                    )
                    st.write(
                        f"**Relevance Score:** {source.get('score', 'N/A')}"
                    )
                    st.divider()

# Accept user question
question = st.chat_input("Ask a question about your documents...")

if question:
    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat/ask",
                    json={
                        "question": question,
                        "top_k": 5
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer received."
                    )

                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.write(
                                    f"**File:** {source.get('file_name', 'Unknown')}"
                                )
                                st.write(
                                    f"**Page:** {source.get('page_number', 'Unknown')}"
                                )
                                st.write(
                                    f"**Relevance Score:** {source.get('score', 'N/A')}"
                                )
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:
                    error_message = (
                        f"Request failed: {response.text}"
                    )
                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

            except requests.RequestException as error:
                error_message = f"Connection error: {error}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Clear chat history
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()