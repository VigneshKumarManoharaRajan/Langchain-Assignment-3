"""
Travel Agent Chatbot using LangChain
=====================================
Features:
  - Streaming token-by-token output via OpenAI + LangChain
  - Manual conversation memory stored in a Python list
  - Auto-summarization every 5 conversation turns to reduce context size
  - Strict travel-only topic enforcement via system prompt
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage


# ──────────────────────────────────────────────
# 1.  LLM  (streaming=True enables token-by-token output)
# ──────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True,                  # Enables streaming responses
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)


# ──────────────────────────────────────────────
# 2.  Prompt Templates
# ──────────────────────────────────────────────

# Main chat prompt — enforces travel-only scope
TRAVEL_SYSTEM_PROMPT = """You are an expert travel agent assistant with deep knowledge of:
- Flight bookings, airlines, and airports
- Hotel recommendations and accommodations
- Travel destinations around the world
- Trip itinerary planning
- Travel tips, visas, and travel insurance
- Local culture, food, and attractions

IMPORTANT RULES:
1. You ONLY answer travel-related questions.
2. If the user asks about ANYTHING unrelated to travel, respond EXACTLY with:
   "I can't help with it"
3. Do not add explanations when refusing — just that exact phrase.
4. Be friendly, concise, and helpful for all travel topics.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", TRAVEL_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),   # injected conversation history
    ("human", "{user_input}"),
])

# Summarization prompt — condenses history into a short paragraph
SUMMARIZE_SYSTEM_PROMPT = """You are a helpful assistant that summarizes conversations.
Summarize the following travel conversation concisely, preserving key details like
destinations mentioned, preferences, and any decisions made."""

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZE_SYSTEM_PROMPT),
    ("human", "Please summarize this conversation:\n\n{conversation_text}"),
])

# Non-streaming LLM for summarization (we don't need to stream the summary)
summarize_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    streaming=False,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)


# ──────────────────────────────────────────────
# 3.  Manual Memory  (plain Python list)
# ──────────────────────────────────────────────
# Each entry is a LangChain message object:
#   HumanMessage(content="...")  or  AIMessage(content="...")
# A SystemMessage summary marker may be prepended after compression.

conversation_history: list = []   # ← the manual memory store
turn_counter: int = 0             # counts completed human↔AI turns


# ──────────────────────────────────────────────
# 4.  Summarization Logic
# ──────────────────────────────────────────────

def summarize_history(history: list) -> str:
    """
    Converts the current conversation_history list into a plain-text
    transcript and asks the LLM to summarize it.
    Returns the summary string.
    """
    # Build a readable transcript from message objects
    lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            lines.append(f"[Summary context]: {msg.content}")

    conversation_text = "\n".join(lines)

    # Call LLM for summarization (non-streaming)
    chain = summarize_prompt | summarize_llm
    result = chain.invoke({"conversation_text": conversation_text})
    return result.content


def compress_history_with_summary(history: list) -> list:
    """
    Summarizes the existing history and replaces it with a single
    SystemMessage containing the summary.  This keeps the context
    window small while preserving conversational context.
    """
    print("\n[Memory] Summarizing conversation history to reduce context size...")
    summary = summarize_history(history)
    print(f"[Memory] Summary created: {summary[:80]}...\n")

    # Replace all previous messages with one summary SystemMessage
    return [SystemMessage(content=f"Previous conversation summary: {summary}")]


# ──────────────────────────────────────────────
# 5.  Streaming Chat Function
# ──────────────────────────────────────────────

def chat(user_input: str) -> str:
    """
    Sends user_input to the LLM with full conversation history,
    streams the response token-by-token to stdout, appends both
    messages to memory, and triggers summarization every 5 turns.

    Returns the complete assistant response as a string.
    """
    global conversation_history, turn_counter

    # Build the prompt with current history
    chain = chat_prompt | llm
    messages_for_prompt = {
        "history": conversation_history,   # inject manual memory list
        "user_input": user_input,
    }

    # ── Streaming: iterate over chunks and print each token immediately ──
    print("\nTravel Agent: ", end="", flush=True)
    full_response = ""

    for chunk in chain.stream(messages_for_prompt):
        token = chunk.content          # each chunk holds one or more tokens
        print(token, end="", flush=True)
        full_response += token

    print()  # newline after streaming completes

    # ── Update manual memory ──
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=full_response))
    turn_counter += 1

    # ── Summarize every 5 turns to keep context manageable ──
    if turn_counter % 5 == 0:
        conversation_history = compress_history_with_summary(conversation_history)

    return full_response


# ──────────────────────────────────────────────
# 6.  Main REPL Loop
# ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ✈️   Travel Agent Chatbot (type 'quit' to exit)")
    print("=" * 55)
    print("Ask me anything about flights, hotels, or destinations!\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Safe travels! ✈️")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Travel Agent: Goodbye! Safe travels! ✈️")
            break

        chat(user_input)

        # Show memory stats for transparency
        print(f"  [Memory: {len(conversation_history)} messages stored, "
              f"turn {turn_counter}]")


if __name__ == "__main__":
    main()
