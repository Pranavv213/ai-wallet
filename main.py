import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

class IntentResponse(BaseModel):
    action: str  # "create_account" | "import_private_key" | "list_accounts" | "add_funds" | "chat"
    message: str

class ChatRequest(BaseModel):
    user_message: str

@app.post("/api/chat", response_model=IntentResponse)
async def chat_endpoint(request: ChatRequest):
    system_prompt = """
    You are an intent-classification assistant for a multi-account Web3 wallet app.
    Analyze the user's message and categorize it into one of these actions:
    
    1. 'create_account': User wants to generate/create an account.
    2. 'import_private_key': User wants to import/paste a key or seed phrase.
    3. 'list_accounts': User wants to view, list, or switch active accounts.
    4. 'add_funds': User wants to fund, send crypto to, or add funds to their selected account.
    5. 'chat': General questions or conversations.
    """

    prompt = f"{system_prompt}\n\nUser Message: {request.user_message}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IntentResponse,
            ),
        )

        result_data = json.loads(response.text)
        return IntentResponse(**result_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))