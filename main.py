import os
import operator
import trafilatura
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- Configuration ---
# Using the requested Gemini 2.5 Flash model
MODEL_NAME = "gemini-2.5-flash"

# --- State Definition ---
class AgentState(TypedDict):
    """
    The state of our Deep Research Agent.
    """
    topic: str
    plan: List[str]
    current_query_index: int
    # Annotated[...] allows us to just return new summaries and have them appended to the list
    summaries: Annotated[List[str], operator.add]
    final_report: str

# --- LLM Initialization ---
# Ensure GOOGLE_API_KEY is available in os.environ before running
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

# --- Nodes ---

def planner_node(state: AgentState):
    """
    Planner Node: Breaks the user's topic into 3 distinct search queries.
    """
    topic = state["topic"]
    print(f"\n--- [PLANNER] Generating search queries for: '{topic}' ---")
    
    system_instruction = (
        "You are a research planner. Break down the user's topic into 3 distinct, "
        "search-optimized queries. Return ONLY the 3 queries, one per line. "
        "Do not include numbering or bullet points."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_instruction), 
        HumanMessage(content=topic)
    ])
    
    # Parse the response into a clean list of queries
    raw_plan = response.content.strip().split('\n')
    plan = [line.strip() for line in raw_plan if line.strip()][:3]
    
    print(f"Plan generated: {plan}")
    
    # Initialize state for the research loop
    return {
        "plan": plan, 
        "current_query_index": 0, 
        "summaries": []
    }

def research_node(state: AgentState):
    """
    Research Node:
    - Takes the current query.
    - Searches DuckDuckGo.
    - Scrapes content.
    - Summarizes with Gemini.
    """
    plan = state["plan"]
    index = state["current_query_index"]
    query = plan[index]
    
    print(f"\n--- [RESEARCHER] Processing Query {index + 1}/{len(plan)}: '{query}' ---")
    
    # 1. Search DuckDuckGo
    print("  -> Searching DuckDuckGo...")
    search_results = []
    try:
        with DDGS() as ddgs:
            # Get top 3 results
            results_gen = ddgs.text(query, max_results=3)
            if results_gen:
                search_results = list(results_gen)
    except Exception as e:
        print(f"  [Error] Search failed: {e}")

    # 2. Loop through URLs and Scrape
    scraped_texts = []
    for result in search_results:
        url = result.get('href')
        title = result.get('title', 'No Title')
        print(f"  -> Scraping: {title} ({url})")
        
        try:
            # Trafilatura fetch and extract
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    # Append source metadata to text for the LLM
                    scraped_texts.append(f"SOURCE: {url}\nCONTENT:\n{text[:8000]}") # Truncate to avoid context overflow
                else:
                    print("     [Skipped] No main text found.")
            else:
                print("     [Skipped] Failed to fetch URL.")
        except Exception as e:
            print(f"     [Error] Scraping failed for {url}: {e}")

    # 3. Summarize Findings
    combined_text = "\n\n".join(scraped_texts)
    summary = ""
    
    if combined_text:
        print("  -> Summarizing findings with Gemini...")
        summary_prompt = (
            f"You are a research assistant. Analyze the following scraped text for the query: '{query}'. "
            f"Provide a concise, fact-heavy summary of the key information found. "
            f"Ignore irrelevant navigation or boilerplate text.\n\n"
            f"{combined_text}"
        )
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content
    else:
        summary = f"No detailed information could be scraped for the query: {query}"
        print("  -> No content scraped. Skipping summary.")

    # Return update to state
    # We append the new summary to 'summaries' and increment the index
    return {
        "summaries": [summary], 
        "current_query_index": index + 1
    }

def writer_node(state: AgentState):
    """
    Writer Node: Takes all summaries and writes the final report.
    """
    print("\n--- [WRITER] Composing Final Report ---")
    topic = state["topic"]
    summaries = state["summaries"]
    
    # Combine all summaries
    research_context = "\n\n---\n\n".join(summaries)
    
    prompt = (
        f"You are a professional technical writer. The user asked for a report on: '{topic}'.\n"
        f"Below are the summaries from the research phase:\n\n"
        f"{research_context}\n\n"
        f"Write a comprehensive, well-structured Markdown report based ONLY on the above findings. "
        f"Include a Title, Introduction, Key Findings (structured appropriately), and Conclusion."
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"final_report": response.content}

# --- Manager Logic (Conditional Edge) ---

def manager_logic(state: AgentState):
    """
    Checks if there are more queries to process.
    """
    if state["current_query_index"] < len(state["plan"]):
        return "continue"
    else:
        return "finish"

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", research_node)
workflow.add_node("writer", writer_node)

# Set Entry Point
workflow.set_entry_point("planner")

# Add Edges
workflow.add_edge("planner", "researcher")

# Conditional Edge for the Loop
workflow.add_conditional_edges(
    "researcher",
    manager_logic,
    {
        "continue": "researcher",  # Loop back to research
        "finish": "writer"         # Move to writing
    }
)

# End Edge
workflow.add_edge("writer", END)

# Compile the graph
app = workflow.compile()

# --- Main Execution Block ---

if __name__ == "__main__":
    print("### Deep Research Agent (Gemini 2.5 + LangGraph) ###")
    
    # Ensure API Key is set
    if not os.environ.get("GOOGLE_API_KEY"):
        key = input("Please enter your GOOGLE_API_KEY: ").strip()
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        else:
            print("Error: GOOGLE_API_KEY is required.")
            exit(1)

    # Get User Input
    user_topic = input("\nEnter the research topic: ")
    
    if user_topic:
        initial_state = {"topic": user_topic}
        
        try:
            # Run the graph
            result = app.invoke(initial_state)
            
            # Output Result
            print("\n" + "="*50)
            print("FINAL REPORT")
            print("="*50 + "\n")
            print(result["final_report"])
            
            # Save to file
            filename = "final_report.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result["final_report"])
            print(f"\n[Saved report to {filename}]")
            
        except Exception as e:
            print(f"An error occurred during execution: {e}")
