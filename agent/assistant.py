from typing import List, Dict

from openai import OpenAI

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, OPENAI_API_KEY
from retrieval.retriever import Retriever


class PortfolioAgent:

    def __init__(self, retriever: Retriever = None):
        self.client    = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = retriever or Retriever()
        self.history:  List[Dict] = []

    def chat(self, user_message: str) -> str:
        knowledge = self.retriever.search_knowledge(user_message)
        code      = self.retriever.search_code(user_message)

        system = f"""You are a helpful assistant for Aria Mostajeran's portfolio website.
Aria is an MSc Data Science & AI graduate from TU/e (Eindhoven, Netherlands),
actively looking for ML / Data Science / AI Engineering roles.

Answer the user's question using ONLY the context below.
Be specific — use real names, numbers, and technologies from the context.
For code questions, show the actual code snippet from the CODE CONTEXT.
If the answer is not in the context, say "I don't have that information."
Keep answers concise (2-5 sentences) unless showing code.

KNOWLEDGE CONTEXT:
{knowledge}

CODE CONTEXT:
{code}"""

        messages = [{"role": "system", "content": system}] + self.history
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model       = LLM_MODEL,
            messages    = messages,
            temperature = LLM_TEMPERATURE,
            max_tokens  = LLM_MAX_TOKENS,
        ).choices[0].message.content

        self.history.append({"role": "user",     "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.history = []

    def status(self) -> Dict:
        return {
            **self.retriever.status(),
            "history_turns": len([m for m in self.history if m["role"] == "user"]),
            "llm_model":     LLM_MODEL,
        }
