from dotenv import load_dotenv
import streamlit as st
from LegalChatBot import LegalGraphChatBot
from DocumentQAGraph import DocumentQATool
import time
import os

load_dotenv()

st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖️",
    layout="wide",
)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found!")
    st.info("""
    **For local development:** Add `OPENAI_API_KEY` to your `.env` file
    
    **For Streamlit Cloud:** Go to Settings → Secrets and add:
    ```
    OPENAI_API_KEY = "your-api-key-here"
    ```
    """)
    st.stop()

# Simple styling
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Simple header */
    .main-header {
        background-color: #1e3a8a;
        color: white;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
        border-radius: 5px;
    }
    
    /* Basic container */
    .content-box {
        background-color: #f9fafb;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    
    /* Simple button styling */
    .stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    /* Error styling */
    .stAlert {
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
<div class="main-header">
    <h1>⚖️ LegalEase AI</h1>
    <p>AI-powered Legal Assistant</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("Select Tool")

    tool_option = st.radio(
        "Choose a tool:",
        ["NyayGPT", "Ask Document"],
        label_visibility="collapsed",
    )

    st.divider()

    st.info(
        "**NyayGPT**: Get legal guidance and consultation\n\n**Ask Document**: Upload and analyze PDF documents"
    )

# Initialize tool selection
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = tool_option
else:
    st.session_state.selected_tool = tool_option


def display_streaming_response(generator):
    """Display streaming response"""
    response_text = ""
    response_container = st.empty()

    try:
        for chunk in generator:
            if chunk:
                response_text += chunk
                response_container.markdown(response_text + "▋")
                time.sleep(0.03)
    except Exception as e:
        response_text = f"Error: {str(e)}"
        response_container.error(response_text)

    response_container.markdown(response_text)
    return response_text


# Main content
if st.session_state.selected_tool == "Chatbot":
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = LegalGraphChatBot()
                st.session_state.chatbot_state = st.session_state.chatbot.init_state()
                st.session_state.chatbot_history = []
                st.success("✓ Chatbot initialized successfully!")
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("""
            **To fix this:**
            1. Run `python embed_docs.py` locally to create the FAISS index
            2. Commit the `faiss_index_legal` folder to your repository
            3. If the folder is too large (>100MB), use Git LFS or cloud storage
            """)
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to initialize chatbot: {str(e)}")
            with st.expander("Show full error"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()

    st.subheader("Legal Chatbot")
    st.write(
        "Ask questions about Indian law, legal procedures, rights, and regulations."
    )

    st.divider()

    # Display chat history
    for user_msg, bot_msg in st.session_state.chatbot_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    # Chat input
    query = st.chat_input("Enter your legal question...")

    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)

        # Display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing..."):
                    generator = st.session_state.chatbot.invoke_streaming(
                        st.session_state.chatbot_state, query
                    )
                    response = display_streaming_response(generator)

                st.session_state.chatbot_history.append((query, response))
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show details"):
                    import traceback
                    st.code(traceback.format_exc())

elif st.session_state.selected_tool == "Document Summarizer":
    # Initialize document bot
    if "docbot" not in st.session_state:
        try:
            with st.spinner("Initializing document analyzer..."):
                st.session_state.docbot = DocumentQATool()
                st.session_state.docbot_state = st.session_state.docbot.init_state()
                st.session_state.docbot_history = []
                st.success("✓ Document analyzer initialized successfully!")
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("""
            **To fix this:**
            1. Run `python embed_docs.py` locally to create the FAISS index
            2. Commit the `faiss_index_legal` folder to your repository
            3. If the folder is too large (>100MB), use Git LFS or cloud storage
            """)
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to initialize document analyzer: {str(e)}")
            with st.expander("Show full error"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()

    st.subheader("Document Summarizer")
    st.write("Upload PDF documents for AI-powered analysis and summarization.")

    st.divider()

    # File upload
    st.write("**Upload Document**")
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload legal documents, contracts, or any PDF",
    )

    if pdf_file:
        if st.button("Process Document"):
            try:
                with st.spinner("Processing document..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress.progress(i + 1)

                    # Save and process
                    temp_path = f"temp_{int(time.time())}.pdf"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.read())

                    success = st.session_state.docbot.upload_pdf_and_embed(
                        temp_path, st.session_state.docbot_state
                    )

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    if success:
                        st.success("✓ Document processed successfully!")
                    else:
                        st.error("❌ Failed to process document.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show details"):
                    import traceback
                    st.code(traceback.format_exc())

    st.divider()

    # Display chat history
    for user_msg, bot_msg in st.session_state.docbot_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    # Chat input
    query = st.chat_input("Ask questions about the document...")

    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)

        # Display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Processing..."):
                    generator = st.session_state.docbot.invoke_streaming(
                        st.session_state.docbot_state, query
                    )
                    response = display_streaming_response(generator)

                st.session_state.docbot_history.append((query, response))
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show details"):
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("LegalEase AI - Your Legal Assistant")

