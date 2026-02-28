import streamlit as st
import os
import markdown
import nh3  # FIX SEC-02: HTML sanitization before PDF rendering
from xhtml2pdf import pisa
from io import BytesIO
from main import app as graph_app  # Import the compiled graph

# --- PDF Generation Utility ---
def convert_markdown_to_pdf(markdown_content):
    """
    Converts Markdown text to PDF bytes.
    """
    # FIX SEC-02: sanitize HTML produced from Gemini output before passing to PDF renderer.
    # markdown.markdown() does NOT strip injected HTML tags; nh3.clean() removes disallowed tags.
    raw_html = markdown.markdown(markdown_content)
    _ALLOWED_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "strong", "em", "code",
        "pre", "blockquote", "br", "hr",
        "table", "thead", "tbody", "tr", "th", "td",
    }
    html_content = nh3.clean(raw_html, tags=_ALLOWED_TAGS)
    
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

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "pdf_source_report" not in st.session_state:
    st.session_state.pdf_source_report = None

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Enter your research topic..."):
    if not api_key and not os.environ.get("GOOGLE_API_KEY"):
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
            initial_state = {
                "topic": prompt,
                "api_key": api_key,
                "plan": [],
                "current_query_index": 0,
                "summaries": [],
                "final_report": ""
            }
            
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
                        # FIX BUG-03: summaries may be an empty list; guard [0] access
                        summaries = state_update.get("summaries", [])
                        summary_len = len(summaries[0]) if summaries else 0
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
            st.session_state.pdf_bytes = None
            st.session_state.pdf_source_report = None

        except Exception as e:
            status_container.update(label="Error Occurred", state="error")
            print(f"[app-error] {e}")
            st.error("An internal error occurred while generating the report.")

# --- Download Button (Outside the chat loop) ---
if st.session_state.final_report:
    st.divider()
    col1, _ = st.columns([1, 4])  # FIX DEAD-02: col2 was created but never used
    with col1:
        if st.session_state.pdf_source_report != st.session_state.final_report:
            st.session_state.pdf_bytes = convert_markdown_to_pdf(st.session_state.final_report)
            st.session_state.pdf_source_report = st.session_state.final_report
        pdf_bytes = st.session_state.pdf_bytes
        if pdf_bytes:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="research_report.pdf",
                mime="application/pdf",
                key="download-pdf"
            )
