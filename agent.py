from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rendered_html, download_file, post_request, run_code, add_dependencies
from typing import TypedDict, Annotated, List, Any, Optional
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
import os
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
USE_AIPIPE_ONLY = os.getenv("USE_AIPIPE_ONLY", "false").lower() == "true"
RECURSION_LIMIT = 5000

# List of fallback models to try (in order) when Gemini hits quota
# Ordered from most capable to least capable
AIPIPE_FALLBACK_MODELS = [
    # Latest and most powerful models
    "anthropic/claude-sonnet-4.5",  # Claude Sonnet 4.5 - most advanced reasoning
    "openai/gpt-5.1",  # GPT-5.1 - latest OpenAI model
    "google/gemini-3.0",  # Gemini 3.0 - latest Google model
    
    # Latest Gemini experimental models
    "google/gemini-exp-1206:free",  # Latest experimental Gemini
    "google/gemini-2.0-flash-thinking-exp-1219:free",  # Gemini with thinking mode
    "google/gemini-2.5-flash",  # Your proven model
    "google/gemini-2.0-flash-exp:free",
    
    # Strong GPT models
    "openai/gpt-4o-mini",  # GPT-4o mini - very capable
    "openai/gpt-4o-mini-2024-07-18",
    "openai/gpt-3.5-turbo",
    
    # Claude models
    "anthropic/claude-3.5-sonnet",  # Claude 3.5 Sonnet
    
    # Llama models
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    
    # Other capable models
    "qwen/qwen-2-7b-instruct:free",
    "openai/gpt-4.1-nano",  # Smallest fallback
]

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]

# -------------------------------------------------
# SMART LLM WITH MULTI-MODEL FALLBACK
# -------------------------------------------------
class FallbackLLM:
    """
    Smart LLM wrapper with multiple fallback models.
    1. Tries direct Gemini API first (fast and free)
    2. Falls back to AiPipe models when Gemini quota is exceeded
    """
    
    def __init__(self):
        self.using_fallback = USE_AIPIPE_ONLY
        self.current_fallback_index = 0
        
        # Initialize direct Gemini as primary (unless USE_AIPIPE_ONLY is true)
        self.primary_llm = None if USE_AIPIPE_ONLY else self._init_gemini()
        
        # Initialize AiPipe models as fallback
        self.fallback_llms = self._init_aipipe_models()
        
        if USE_AIPIPE_ONLY:
            print("⚙️ Configured to use AiPipe only (skipping direct Gemini)")
        else:
            if self.primary_llm:
                print("🟢 Using direct Gemini API as primary")
            print(f"🟡 AiPipe fallback ready with {len(self.fallback_llms)} model(s)")
        
        if not self.primary_llm and not self.fallback_llms:
            raise Exception("No LLM available. Please check your API keys.")
        
        print("✅ LLM initialization complete")
    
    def _init_gemini(self) -> Optional[BaseChatModel]:
        """Initialize Google Gemini DIRECT API with rate limiting"""
        try:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=5/60,  # 5 requests per minute
                check_every_n_seconds=1,  
                max_bucket_size=5  
            )
            
            # Using Gemini 2.5 Flash directly - the model that worked for you!
            llm = init_chat_model(
                model_provider="google_genai",
                model="gemini-2.5-flash",
                rate_limiter=rate_limiter
            ).bind_tools(TOOLS)
            
            print("🟢 Direct Gemini 2.5 Flash initialized successfully")
            return llm
            
        except Exception as e:
            print(f"⚠️ Failed to initialize direct Gemini: {e}")
            return None
    
    def _init_aipipe_models(self) -> List[BaseChatModel]:
        """Initialize multiple AiPipe models as fallbacks"""
        if not AIPIPE_TOKEN:
            print("⚠️ AIPIPE_TOKEN not found in .env - fallback won't be available")
            return []
        
        llms = []
        for model in AIPIPE_FALLBACK_MODELS:
            try:
                llm = ChatOpenAI(
                    model=model,
                    api_key=AIPIPE_TOKEN,
                    base_url="https://aipipe.org/openrouter/v1",
                    default_headers={
                        "accept": "*/*",
                        "accept-language": "en-US,en;q=0.9",
                    },
                    temperature=0.7,
                    max_tokens=4000,
                    max_retries=2,
                ).bind_tools(TOOLS)
                
                llms.append(llm)
                print(f"   ↳ AiPipe '{model}' ready")
                
            except Exception as e:
                print(f"   ⚠️ Failed to init AiPipe '{model}': {e}")
        
        return llms
    
    def __call__(self, input_data, config=None, **kwargs):
        """
        Make the class callable for use with RunnableLambda.
        Tries direct Gemini first, then AiPipe models on quota errors.
        """
        # If already using fallback, skip Gemini
        if self.using_fallback and self.fallback_llms:
            return self._invoke_with_fallback(input_data, config, **kwargs)
        
        # Try direct Gemini API first
        if self.primary_llm:
            try:
                print("🟢 Using direct Gemini API")
                return self.primary_llm.invoke(input_data, config=config, **kwargs)
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for quota/rate limit errors
                quota_indicators = [
                    "429", "quota", "resourceexhausted", 
                    "rate limit", "rate_limit", "ratelimit",
                    "quota_metric", "violations"
                ]
                
                is_quota_error = any(indicator in error_str for indicator in quota_indicators)
                
                if is_quota_error:
                    print(f"⚠️ Direct Gemini quota exceeded! Switching to AiPipe fallback...")
                    self.using_fallback = True
                    return self._invoke_with_fallback(input_data, config, **kwargs)
                else:
                    print(f"❌ Gemini error (not quota): {str(e)[:200]}")
                    raise e
        
        # If direct Gemini not available, use AiPipe fallback
        if self.fallback_llms:
            print("🟡 Direct Gemini unavailable, using AiPipe")
            return self._invoke_with_fallback(input_data, config, **kwargs)
        
        raise Exception("No LLM available")
    
    def _invoke_with_fallback(self, input_data, config=None, **kwargs):
        """Try AiPipe fallback models in sequence until one works"""
        if not self.fallback_llms:
            raise Exception("AiPipe fallback not available. Please set AIPIPE_TOKEN in .env")
        
        attempts = 0
        max_attempts = len(self.fallback_llms) * 2  # Try each model twice
        last_error = None
        
        while attempts < max_attempts:
            current_model = self.fallback_llms[self.current_fallback_index]
            model_name = AIPIPE_FALLBACK_MODELS[self.current_fallback_index]
            
            try:
                print(f"🟡 AiPipe: {model_name}")
                result = current_model.invoke(input_data, config=config, **kwargs)
                # Success!
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                attempts += 1
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate" in error_str or "limit" in error_str:
                    print(f"⚠️ {model_name} rate limited, next...")
                    # Move to next model
                    self.current_fallback_index = (self.current_fallback_index + 1) % len(self.fallback_llms)
                    
                    # Progressive delay
                    if attempts < max_attempts:
                        delay = min(2 * (attempts // len(self.fallback_llms) + 1), 10)
                        time.sleep(delay)
                else:
                    # Not a rate limit error
                    print(f"❌ {model_name} error: {str(e)[:200]}")
                    raise e
        
        # All models failed
        error_msg = f"All {len(self.fallback_llms)} AiPipe models rate-limited/unavailable."
        if last_error:
            error_msg += f" Last: {str(last_error)[:200]}"
        raise Exception(error_msg)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent. You solve challenges step-by-step.

WORKFLOW:
1. Fetch the page using get_rendered_html(url)
2. Analyze the HTML to find:
   - The quiz question/task
   - Required parameters (email, secret, etc.)
   - The submission endpoint URL
   - The expected format (JSON structure)
3. Solve the task
4. Submit using post_request(url, payload)
5. Check response for next URL or END signal

CRITICAL RULES:
- DO NOT fetch the same URL repeatedly - fetch once, analyze, then act
- ALWAYS read the full HTML before deciding what to do
- NEVER invent URLs or endpoints - extract them from the page
- Include email={EMAIL} and secret={SECRET} when required
- If response has "url" field → fetch that URL next
- If response has NO "url" field → return exactly "END"

AVAILABLE TOOLS:
- get_rendered_html(url) - Fetch and render a webpage
- post_request(url, payload, headers) - Submit answers via POST
- download_file(url, filename) - Download files
- run_code(code, language) - Execute code
- add_dependencies(packages) - Install Python packages

REMEMBER:
- Each page is different - read instructions carefully
- Don't repeat actions - if you fetched a page, analyze it and move on
- Time limit is 3 minutes per task
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

# Create smart LLM with fallback and wrap it in RunnableLambda
smart_llm = FallbackLLM()
llm_runnable = RunnableLambda(smart_llm)
llm_with_prompt = prompt | llm_runnable


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"
    
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content[0].get("text").strip() == "END":
        return END
    return "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)

app = graph.compile()


# -------------------------------------------------
# RUN AGENT
# -------------------------------------------------
def run_agent(url: str) -> str:
    app.invoke(
        {"messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print("Tasks completed successfully")
