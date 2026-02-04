import streamlit as st
import os
import markdown
from xhtml2pdf import pisa
from io import BytesIO
from main import app as graph_app  # Import the compiled graph

# --- PDF Generation Utility ---
def convert_markdown_to_pdf(markdown_content):
    """
    Converts Markdown text to PDF bytes.
    """
    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_content)
    
    # Add some basic styling for the PDF
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; font-size: 12px; }}
            h1 {{ color: #333; font-size: 24px; }}
            h2 {{ color: #444; font-size: 20px; }}
            h3 {{ color: #555; font-size: 16px; }}
            p {{ line-height: 1.5; }}
            code {{ background-color: #f4f4f4; padding: 2px; }}
            pre {{ background-color: #f4f4f4; padding: 10px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(styled_html, dest=pdf_buffer)
    
    if pisa_status.err:
        return None
    return pdf_buffer.getvalue()

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Deep Research Agent", page_icon="🕵️‍♂️", layout="wide")

# Custom CSS for a cleaner chat look
st.markdown("""
<style>
    .stChatMessage {
        background-color: transparent; 
        border: none;
    }
    .stChatMessage .stMarkdown {
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stChatMessageContent"] {
        background-color: #001f3f;
        border-radius: 10px;
        padding: 10px;
        color: #ffffff;
    }
    div[data-testid="stChatMessageContent"] p {
        margin-bottom: 0.5rem;
    }
    /* Dark mode adjustments would go here if needed */
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ Deep Research Agent")
st.caption("Powered by Gemini 2.5 Flash, LangGraph, & DuckDuckGo")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Get it from Google AI Studio")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("1. **Planner**: Breaks topic into 3 queries.")
    st.markdown("2. **Researcher**: Searches & scrapes web content.")
    st.markdown("3. **Writer**: Compiles a final report.")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "final_report" not in st.session_state:
    st.session_state.final_report = None

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Enter your research topic..."):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please enter your Google API Key in the sidebar.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        status_container = st.status("Initializing Agent...", expanded=True)
        final_report_text = ""
        
        try:
            initial_state = {"topic": prompt, "plan": [], "current_query_index": 0, "summaries": []}
            
            # Stream the graph execution to show progress
            # We use stream(mode="updates") to see which node finished and what it produced
            for event in graph_app.stream(initial_state):
                
                # 'event' is a dictionary like {'planner': {...}} or {'researcher': {...}}
                for node_name, state_update in event.items():
                    
                    if node_name == "planner":
                        plan = state_update.get("plan", [])
                        status_container.write(f"✅ **Plan Created**: Generated {len(plan)} search queries.")
                        status_container.write(f"_{plan}_")
                    
                    elif node_name == "researcher":
                        # The researcher node outputs the *new* index, so query just finished was index-1
                        idx = state_update.get("current_query_index", 1) - 1
                        summary_len = len(state_update.get("summaries", [""])[0])
                        status_container.write(f"🔍 **Research Step**: Finished Query {idx+1}. (Scraped & Summarized {summary_len} chars)")
                    
                    elif node_name == "writer":
                        final_report_text = state_update.get("final_report", "")
                        status_container.update(label="Research Complete!", state="complete", expanded=False)
            
            # Show the final report
            st.markdown("### 📝 Final Report")
            st.markdown(final_report_text)
            
            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": final_report_text})
            st.session_state.final_report = final_report_text

        except Exception as e:
            status_container.update(label="Error Occurred", state="error")
            st.error(f"An error occurred: {str(e)}")

# --- Download Button (Outside the chat loop) ---
if st.session_state.final_report:
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        pdf_bytes = convert_markdown_to_pdf(st.session_state.final_report)
        if pdf_bytes:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="research_report.pdf",
                mime="application/pdf",
                key="download-pdf"
            )
