import os
import asyncio
import uuid
import logging
import re
import sqlite3
from dotenv import load_dotenv
import time
import sys
from io import StringIO
import contextlib

# --- Core ADK Imports ---
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService

# --- GenAI Imports ---
import google.generativeai as genai
from google.genai import types

# --- Configuration & Setup ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")
genai.configure(api_key=api_key)

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash-lite" 
DB_NAME = "zappos_support.db"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- Policy Knowledge Base ---
POLICY_KNOWLEDGE_BASE = """
### Customer Support Policies

* **24/7 Availability:** The Zappos Customer Loyalty Team is available 24/7, with the exception of several major U.S. holidays.
* **Service Philosophy:**
    * The primary goal is to make the customer happy.
    * Support agents do not use scripts or focus on minimizing call times.
    * Agents are empowered to do what they believe is the right thing for the customer and the company.

### Shipping Policies

* **Free Shipping:** Zappos offers free standard shipping on all orders shipped within the United States, including U.S. territories and military APO/FPO addresses.
* **Shipping Speeds:**
    * **Standard Shipping:** Typically arrives in 3-5 business days.
    * **Expedited Shipping:** 1-3 business days. This is a free benefit for Zappos VIP members.
* **Shipping Locations:**
    * Shipments can be sent to PO Boxes and Military APO/FPO addresses (these may take additional time).
    * Zappos does not ship to international locations (other than U.S. territories and APO/FPO addresses).
* **Restrictions:** Expedited shipping is not available for all locations (e.g., outside the continental U.S., PO Boxes) or for all items (e.g., items considered hazardous materials).

### Return Policies

* **Free Returns:** Returns are free for all orders shipped within the United States, including U.S. territories and military APO/FPO addresses.
* **Return Window:**
    * Items can be returned for a full refund to the original form of payment if the return is dropped off and scanned by the carrier within **60 days** of the purchase date.
    * Items may be returned for store credit for up to **one year** from the purchase date.
* **Item Condition:**
    * Merchandise must be returned in new, unworn, and unwashed condition.
    * Items must be in their original packaging with all original tags, including any seals and security tags, still attached.
    * Footwear must be returned with the original shoe box in its original condition, without postal labels or tape affixed to it.
* **Non-Returnable Items:** "Final Sale" items are not eligible for return.
* **How to Return:**
    * Customers with a Zappos account can start a return from their order history.
    * Customers who checked out as a guest can use the order number from their confirmation email.
    * Zappos provides options for returns, including using a UPS QR code (no printer needed) or printing a return label.
* **Refunds:**
    * Zappos offers "Fast Refunds," where the refund process is initiated as soon as the return package is scanned by the carrier.
    * It may take some financial institutions up to 10 business days (or longer for services like After Pay or PayPal) to process the refund.
* **Gift Returns:** Gifts can be returned by contacting the Customer Loyalty Team. The refund is typically issued as a Zappos gift card to the person returning the gift.

### Other Key Policies

* **Conditions of Use:** Governs the use of the Zappos website and services. Zappos reserves the right to refuse service, terminate accounts, or cancel orders at its discretion.
* **Privacy Policy:** Zappos collects and uses customer information to process orders, respond to requests, and for marketing. They have security measures in place but do not guarantee 100% security of personal information.
* **Store Credit:**
    * Store credit is non-transferable and can only be used on the account it was issued to.
    * It does not expire and does not have any service fees.
* **Interest-Based Ads:** Zappos uses customer interactions on its site to display personalized or targeted ads for products and services. They state they do not share information that on its own identifies you (like name or email) with advertisers for this purpose.
"""

# --- Database Initialization ---
def init_database():
    """Creates the zappos_support.db file and history table if it doesn't exist."""
    log.info(f"Initializing database: {DB_NAME}")
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            summary TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email ON history (email);")
        
        conn.commit()
    except sqlite3.Error as e:
        log.error(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()

# --- Custom Tools ---
def generate_ticket_id() -> str:
    """Generates a new, unique ticket ID."""
    log.info("[Tool Call: generate_ticket_id]")
    return f"TICKET_{uuid.uuid4().hex[:8].upper()}"

def tokenize_pii(text_to_tokenize: str) -> str:
    """Finds PII (emails, phones) in a string and replaces it with a token."""
    log.info(f"[Tool Call: tokenize_pii] Tokenizing: {text_to_tokenize}")
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    
    text = re.sub(email_pattern, "[REDACTED_EMAIL]", text_to_tokenize)
    text = re.sub(phone_pattern, "[REDACTED_PHONE]", text)
    return text

def get_policy_info(topic: str) -> str:
    """Retrieves a specific policy section from the knowledge base (RAG)."""
    log.info(f"[Tool Call: get_policy_info] Searching for topic: {topic}")
    topic_lower = topic.lower()
    
    if "return" in topic_lower or "damaged" in topic_lower or "defective" in topic_lower or "refund" in topic_lower:
        return_match = re.search(r"### Return Policies\n(.*?)(?=\n###|$)", POLICY_KNOWLEDGE_BASE, re.DOTALL | re.IGNORECASE)
        policy_text = return_match.group(1).strip() if return_match else ""
        return policy_text if policy_text.strip() else "Error: Could not find return policies."

    elif "shipping" in topic_lower or "lost" in topic_lower or "missing" in topic_lower:
        match = re.search(r"### Shipping Policies\n(.*?)(?=\n###|$)", POLICY_KNOWLEDGE_BASE, re.DOTALL | re.IGNORECASE)
        if match: 
            return match.group(1).strip()

    elif "order" in topic_lower or "forgot" in topic_lower:
        return "If you've forgotten your order number, please provide the email address associated with your purchase and the approximate date of the order. Our team can locate your order details using this information. Note that for guest checkouts, the order number is sent in the confirmation email."

    log.warning(f"[Tool Call: get_policy_info] No matching section found for '{topic}'")
    return f"Error: No policy information found for the topic '{topic}'."

def get_customer_history(email: str) -> str:
    """Retrieves the past support history for a customer from the SQLite database."""
    log.info(f"[Tool Call: get_customer_history] Looking up history for: {email}")
    if not email or email == "Unknown":
        return "No customer email provided. Cannot retrieve history."
    
    email_lower = email.lower()
    history = []
    
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("SELECT summary FROM history WHERE email = ? ORDER BY timestamp ASC", (email_lower,))
        rows = cursor.fetchall()
        
        if not rows:
            return "No past history found for this customer."
        
        history = [row[0] for row in rows]
        
    except sqlite3.Error as e:
        log.error(f"Database read error: {e}")
        return f"Error retrieving history: {e}"
    finally:
        if conn:
            conn.close()
            
    return "Found past customer history:\n" + "\n".join(history)

def save_customer_history(email: str, ticket_id: str, summary: str) -> str:
    """Saves a summary of the current issue to the SQLite database."""
    log.info(f"[Tool Call: save_customer_history] Saving history for: {email}")
    if not email or email == "Unknown":
        return "No customer email provided. Cannot save history."
    
    email_lower = email.lower()
    record = f"Issue {ticket_id} (Current): {summary}"
    
    try:
        for attempt in range(5):
            try:
                conn = sqlite3.connect(DB_NAME, timeout=10)
                cursor = conn.cursor()
                
                cursor.execute("INSERT INTO history (email, summary, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (email_lower, record))
                conn.commit()
                
                log.info(f"Database updated for {email_lower}.")
                return f"Successfully saved new history record for {email_lower}."
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    time.sleep(0.5)
                    continue
                else:
                    raise
            finally:
                if conn:
                    conn.close()
        raise sqlite3.OperationalError("Database locked after retries")
    except sqlite3.Error as e:
        log.error(f"Database write error: {e}")
        return f"Error saving history: {e}"

# --- Coordinator Agent Definition ---
def create_coordinator_agent() -> LlmAgent:
    """Creates the main Coordinator Agent."""
    log.info("Creating Unified Coordinator Agent...")
    
    return LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="UnifiedSupportAgent",
        description="A single agent that handles all customer interactions.",
        instruction="""
        You are a stateless, unified customer support agent for Zappos.
        
        **CRITICAL RULE:** Process each user message from scratch. Use function calls for tools when needed.

        **PLAN:**

        1. Analyze Intent of the latest message:
           - Chit-chat: purely greetings or casual talk without any question or request (e.g., "hello", "hi", "how are you", "thank you").
           - Support request: questions or requests about policies, returns, shipping, orders, or any support-related topic (e.g., "I want to return my item", "What is the return policy?", "I forgot my order number").

        2. If chit-chat:
           - Respond briefly and friendly. No tools.

        3. If support request:
           - Use tools to gather info and respond accordingly.
           - Always call tools in sequence as per the support plan.
           - Only output the final refined response after all steps.

        **SUPPORT PLAN:**
        Use function calls for:
        1. generate_ticket_id
        2. get_customer_history with email from message.
        3. get_policy_info with identified topic.
        4. tokenize_pii on message and history.
        5. Then, draft response based on info.
        6. Critique and refine.
        7. save_customer_history with summary.
        8. Output only the refined response.

        If policy not found, respond empathetically and suggest contacting support with more details.

        Always be empathetic, helpful, and concise.

        Remember, your final message should contain only the refined response to the user. Do not include any reasoning, steps, or mentions of tools in your final output.
        """,
        tools=[
            FunctionTool(func=get_policy_info),
            FunctionTool(func=get_customer_history),
            FunctionTool(func=save_customer_history),
            FunctionTool(func=tokenize_pii),
            FunctionTool(func=generate_ticket_id)
        ]
    )

# --- Create Runner ---
def create_runner():
    root_agent = create_coordinator_agent()
    session_service = InMemorySessionService()
    
    runner = Runner(
        agent=root_agent,
        app_name="CustomerSupportPipeline",
        session_service=session_service,
    )
    return runner

async def get_agent_response(user_input):
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    
    runner = create_runner()
    output = StringIO()
    with contextlib.redirect_stdout(output):
        events = await runner.run_debug(user_input)
    
    logging.getLogger().setLevel(original_level)
    
    # Extract final response
    final_text = ""
    for event in reversed(events or []):
        try:
            if hasattr(event, 'content') and event.content and event.content.role == 'model':
                for part in (event.content.parts or []):
                    if hasattr(part, 'text') and part.text:
                        final_text = part.text.strip()
                        if final_text.startswith('"""') and final_text.endswith('"""'):
                            final_text = final_text[3:-3].strip()
                        elif final_text.startswith('"') and final_text.endswith('"'):
                            final_text = final_text[1:-1].strip()
                        elif final_text.startswith("'") and final_text.endswith("'"):
                            final_text = final_text[1:-1].strip()
                        break
                if final_text:
                    break
        except AttributeError:
            continue
    
    return final_text if final_text else "I'm sorry, I couldn't process your request. How can I assist you?"


def main():
    init_database()
    
    print("Welcome! Please enter your information to begin")
    
    name = input("Your Name: ").strip()
    
    while True:
        email = input("Your Email: ").strip()
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        if re.match(email_pattern, email):
            break
        print("Please enter a valid email address")
    
    print("\nChat started. Type 'quit' to exit.")
    
    first = True
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == 'quit':
            break
        
        if first:
            enhanced_prompt = f"My name is {name} and my email is {email}. {prompt}"
            first = False
        else:
            enhanced_prompt = f"My email is {email}. {prompt}"
        
        try:
            response_text = asyncio.run(get_agent_response(enhanced_prompt))
            print(f"Assistant: {response_text}")
        except Exception as e:
            print(f"Assistant: Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    main()