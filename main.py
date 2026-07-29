import os
import json
from typing import List, Dict, Optional, Annotated, Sequence
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

# -------------------------------------------------------------------
# Data Schemas
# -------------------------------------------------------------------
class AccountInfo(BaseModel):
    name: str
    address: str

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    user_message: str
    accounts: List[AccountInfo] = []
    active_account_index: Optional[int] = -1

class IntentStructuredOutput(BaseModel):
    action: str = Field(
        description="Categorize into: 'create_account', 'list_accounts', 'add_funds', 'check_balance', 'send_tokens', 'chat'"
    )
    target_address: Optional[str] = Field(
        default=None, 
        description="Resolved EVM address for balance checks, funding, or sender selection."
    )
    recipient_address: Optional[str] = Field(
        default=None, 
        description="Recipient EVM address if transferring tokens."
    )
    amount: Optional[float] = Field(
        default=None, 
        description="Amount to transfer if specified."
    )
    asset_type: Optional[str] = Field(
        default="ETH", 
        description="Asset type (ETH or ERC20)."
    )
    message: str = Field(
        description="Conversational response or clarifying prompt to the user."
    )

class ChatResponse(BaseModel):
    action: str
    message: str
    target_address: Optional[str] = None
    recipient_address: Optional[str] = None
    amount: Optional[float] = None
    asset_type: Optional[str] = None

# -------------------------------------------------------------------
# LangGraph Setup
# -------------------------------------------------------------------
class AgentState(Dict):
    messages: List[BaseMessage]
    accounts: List[AccountInfo]
    user_message: str
    intent_result: Optional[IntentStructuredOutput]

def process_intent_node(state: AgentState) -> AgentState:
    accounts_str = json.dumps([acc.dict() for acc in state["accounts"]]) if state["accounts"] else "No accounts created yet."
    
    system_instructions = f"""
    You are an intelligent Web3 assistant managing user vault accounts on Ethereum Sepolia.
    
    Current User Vault Accounts:
    {accounts_str}
    
    Analyze the user's message in context of the ongoing conversation and determine the action:
    1. 'create_account': User wants to create/generate a new account.
    2. 'list_accounts': User wants to see all listed vault accounts.
    3. 'add_funds': User wants to add funds/deposit to an account.
       - If the user specifies an account (e.g., 'account 3', '2nd account', or an explicit address), map it to the exact address from the vault list and populate `target_address`.
       - If NO specific account or address is provided, set `target_address` to null and ask the user in `message` to choose which account to fund.
    4. 'check_balance': User asks for the balance of a specific account/wallet (e.g., "balance of account 2", "check 0x...").
       - Resolve and set `target_address` to the corresponding address.
    5. 'send_tokens': User wants to send/transfer ETH or ERC20 tokens (e.g., "send 0.1 eth from account 2 to 0x...").
       - Map the sender to `target_address`, recipient to `recipient_address`, and set `amount` & `asset_type`.
    6. 'chat': General conversation or questions.
    """

    messages = [SystemMessage(content=system_instructions)] + state["messages"]
    
    structured_llm = llm.with_structured_output(IntentStructuredOutput)
    response: IntentStructuredOutput = structured_llm.invoke(messages)
    
    state["intent_result"] = response
    return state

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("process_intent", process_intent_node)
builder.add_edge(START, "process_intent")
builder.add_edge("process_intent", END)

# In-memory checkpointer for conversational memory
memory_saver = MemorySaver()
graph = builder.compile(checkpointer=memory_saver)

# -------------------------------------------------------------------
# FastAPI Endpoint
# -------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    
    # Run graph execution
    inputs = {
        "messages": [HumanMessage(content=request.user_message)],
        "accounts": request.accounts,
        "user_message": request.user_message
    }
    
    result = graph.invoke(inputs, config=config)
    intent: IntentStructuredOutput = result["intent_result"]
    
    return ChatResponse(
        action=intent.action,
        message=intent.message,
        target_address=intent.target_address,
        recipient_address=intent.recipient_address,
        amount=intent.amount,
        asset_type=intent.asset_type
    )