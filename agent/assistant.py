"""
PortfolioAgent — ReAct agent that answers questions about Aria's portfolio.

ReAct loop (Reason + Act):
  1. LLM receives question + tool descriptions
  2. LLM responds with TOOL/INPUT to call a tool
  3. Tool result injected back into messages
  4. LLM either calls another tool or writes final answer
  5. Final answer stored in history for multi-turn memory

Three tools:
  search_knowledge — semantic search over CV + project docs
  get_cv_section   — direct CV section lookup by name
  search_code      — semantic search over source code
"""

import re
from typing import List, Dict, Tuple, Optional

from openai import OpenAI

from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, OPENAI_API_KEY
)
from retrieval.retriever import Retriever


class PortfolioAgent:

    def __init__(self, retriever: Retriever = None):
        self.client    = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = retriever or Retriever()
        self.history:  List[Dict] = []
        self.tools     = self._register_tools()
        self.system_prompt = self._build_system_prompt()

    # ── Tool registry ─────────────────────────────────────────────────────────

    def _register_tools(self) -> Dict:
        """
        Maps tool names to their function + description.
        The description is what the LLM reads to decide which tool to use —
        write it clearly and specifically, it matters a lot.
        """
        return {
            "search_knowledge": {
                "fn": self.retriever.search_knowledge,
                "description": (
                    "Search Aria's portfolio knowledge base — CV, project descriptions, "
                    "skills, experience, education, awards. Use for any factual question "
                    "about Aria's background, projects, or qualifications."
                ),
                "input": "A natural language search query"
            },
            "get_cv_section": {
                "fn": self.retriever.get_cv_section,
                "description": (
                    "Retrieve a specific CV section by name. More precise than "
                    "search_knowledge for known sections: Skills, Experience, "
                    "Education, Awards, Profile, Availability, Certifications."
                ),
                "input": "Section name, e.g. 'Skills' or 'Experience'"
            },
            "search_code": {
                "fn": self.retriever.search_code,
                "description": (
                    "Search Aria's actual source code from his GitHub repos. Use when "
                    "asked how something is implemented, to see a specific function, "
                    "or for technical implementation details."
                ),
                "input": "A search query describing what code to find"
            }
        }

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """
        The system prompt sets the agent's behaviour for the whole session.
        Tool descriptions are injected here — the LLM reads them to decide
        which tool fits each question.

        Key instructions:
        - ALWAYS call a tool first (prevents hallucination)
        - Only state facts from tool results (grounding)
        - Refuse clearly when answer isn't in context
        """
        tool_list = "\n".join([
            f"- {name}: {info['description']} | Input: {info['input']}"
            for name, info in self.tools.items()
        ])

        return f"""You are a helpful assistant for Aria Mostajeran's portfolio website.
Aria is an MSc Data Science & AI graduate from TU/e (Eindhoven, Netherlands),
actively looking for ML / Data Science / AI Engineering roles.

You have access to these tools to look up information about Aria:
{tool_list}

To call a tool, respond with EXACTLY this format and nothing else:
TOOL: tool_name
INPUT: your input here

After receiving the tool result:
- Call another tool if you need more information
- Write your final answer if you have enough

Rules:
- ALWAYS call a tool before answering — never answer from memory
- Only state facts that appear in tool results
- If the answer is not in any tool result, say "I don't have that information"
- Be specific: use real names, numbers, technologies from the results
- For code questions, show the actual code from search_code results
- Keep answers concise (2-5 sentences) unless detail is clearly needed"""

    # ── Core chat loop ────────────────────────────────────────────────────────

    def chat(self, user_message: str, verbose: bool = False) -> str:
        """
        Process one user message through the ReAct loop.
        Appends to self.history so context carries across turns.

        verbose=True prints each iteration — useful for debugging
        which tools are being called and what they return.

        History note:
        self.history stores only final user/assistant exchanges.
        Tool calls within a turn are stored in the local `messages` list
        but not persisted to history — the LLM sees them within the turn
        but they don't accumulate across turns.
        """
        self.history.append({"role": "user", "content": user_message})

        # Full message list for this turn: system + conversation history
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.history

        for iteration in range(1, 5):  # max 4 tool calls per turn

            response = self.client.chat.completions.create(
                model       = LLM_MODEL,
                messages    = messages,
                temperature = LLM_TEMPERATURE,
                max_tokens  = LLM_MAX_TOKENS
            ).choices[0].message.content

            if verbose:
                print(f"\n[iter {iteration}] LLM output:\n{response[:400]}")

            tool_name, tool_input = self._parse_tool_call(response)

            if tool_name:
                tool_result = self._call_tool(tool_name, tool_input)

                if verbose:
                    print(f"[iter {iteration}] Tool '{tool_name}' →\n{tool_result[:300]}...")

                # Append tool exchange to this turn's messages
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result:\n{tool_result}\n\n"
                        f"Now answer the original question using this information."
                    )
                })

            else:
                # No tool call detected — this is the final answer
                self.history.append({"role": "assistant", "content": response})
                return response

        # Safety fallback if max iterations hit without a final answer
        fallback = "I wasn't able to find a complete answer. Please try rephrasing."
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_tool_call(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract tool name and input from LLM response using regex.
        Returns (None, None) if no tool call found — signals final answer.
        """
        tool_match  = re.search(r'TOOL:\s*(\w+)',            response)
        input_match = re.search(r'INPUT:\s*(.+?)(?:\n|$)',   response, re.DOTALL)

        if tool_match and input_match:
            return tool_match.group(1).strip(), input_match.group(1).strip()
        return None, None

    def _call_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in self.tools:
            return (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {list(self.tools.keys())}"
            )
        return self.tools[tool_name]["fn"](tool_input)

    # ── Session management ────────────────────────────────────────────────────

    def reset(self):
        """Clear conversation history for a fresh session."""
        self.history = []

    def status(self) -> Dict:
        """Return agent + knowledge base status for diagnostics."""
        return {
            **self.retriever.status(),
            "history_turns": len([m for m in self.history if m["role"] == "user"]),
            "llm_model":     LLM_MODEL,
        }