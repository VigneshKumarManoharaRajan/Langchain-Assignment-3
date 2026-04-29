# ✈️ Travel Agent Chatbot

A conversational travel assistant built with **LangChain** and **OpenAI GPT-4o-mini** that answers travel-related questions with streamed responses, manual memory management, and automatic chat summarization.

---

## Features

| Feature | Details |
|---|---|
| 🌊 Streaming output | Responses print token-by-token in real time |
| 🧠 Manual memory | Full conversation history stored in a Python list |
| 📝 Auto-summarization | Every 5 turns, history is compressed into a summary |
| 🚫 Topic enforcement | Non-travel questions are rejected with a fixed message |
| 🔗 LangChain prompts | Structured via `ChatPromptTemplate` + `MessagesPlaceholder` |

---

## Project Structure

```
travel_agent_chatbot.py   ← single file, all logic included
README.md
```

---

## Requirements

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Installation

```bash
# 1. Clone or download the project
# 2. Install dependencies
pip install langchain langchain-openai openai

# 3. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."   # macOS/Linux
set OPENAI_API_KEY=sk-...        # Windows CMD
```

---

## Usage

```bash
python travel_agent_chatbot.py
```

```
=======================================================
  ✈️   Travel Agent Chatbot (type 'quit' to exit)
=======================================================
Ask me anything about flights, hotels, or destinations!

You: Best time to visit Japan?

Travel Agent: The best time to visit Japan is during spring (March–May)
for cherry blossoms or autumn (September–November) for fall foliage...
  [Memory: 2 messages stored, turn 1]

You: What's 2+2?

Travel Agent: I can't help with it
  [Memory: 4 messages stored, turn 2]
```

Type `quit`, `exit`, or `bye` to end the session.

---

## How It Works

### 1. Streaming
`ChatOpenAI` is initialized with `streaming=True`. The `chain.stream()` iterator yields chunks as the model generates them, and each token is printed immediately with `flush=True` — producing a live typing effect.

```python
for chunk in chain.stream(messages_for_prompt):
    print(chunk.content, end="", flush=True)
```

### 2. Manual Memory
Conversation history is stored in a plain Python list of LangChain message objects. After each turn, both the user message and assistant reply are appended and injected into the next prompt via `MessagesPlaceholder`.

```python
conversation_history.append(HumanMessage(content=user_input))
conversation_history.append(AIMessage(content=full_response))
```

### 3. Auto-Summarization (every 5 turns)
When `turn_counter` hits a multiple of 5, the full history is converted to a text transcript, sent to the LLM for summarization, and then **replaced** with a single `SystemMessage` containing the summary. This keeps the context window small for long conversations.

```
Turn 5 reached →
  [10 messages] → summarize → [1 SystemMessage with summary]
```

### 4. Topic Enforcement
The system prompt strictly instructs the model to reply with `"I can't help with it"` for any non-travel question — no explanations, no alternatives.

---

## Configuration

You can tweak the following constants directly in the script:

| Variable | Default | Description |
|---|---|---|
| `model` | `gpt-4o-mini` | OpenAI model to use |
| `temperature` (chat) | `0.7` | Response creativity |
| `temperature` (summary) | `0.3` | Summary consistency |
| `turn_counter % 5` | `5` | Summarization frequency |

---

## Dependencies

```
langchain
langchain-openai
openai
```

---

## License

MIT — free to use and modify.
