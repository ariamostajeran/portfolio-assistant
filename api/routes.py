"""
Flask API routes for the portfolio assistant.

Endpoints:
  GET  /projects/assistant   — serves the chat UI page
  POST /api/chat             — send a message, get an answer
  POST /api/chat/reset       — clear conversation history
  GET  /api/chat/status      — knowledge base health check

Session management:
  Each browser session gets a UUID → one PortfolioAgent instance.
  Agents stored in a module-level dict (_agents).
  In production this would use Redis — for a portfolio site a dict is fine.
"""

import os
import uuid
from flask import Blueprint, request, jsonify, render_template, session

from agent.assistant import PortfolioAgent

bp = Blueprint("assistant", __name__)

# Module-level dict: session_id → PortfolioAgent
# One agent per visitor session so conversation history is isolated
_agents: dict = {}


def _get_agent(session_id: str) -> PortfolioAgent:
    """
    Get or create an agent for this session.
    Creating a new agent initialises the Retriever (loads embedding model,
    connects to Chroma) — this happens once per session, not per message.
    """
    if session_id not in _agents:
        _agents[session_id] = PortfolioAgent()
    return _agents[session_id]


# ── Routes ────────────────────────────────────────────────────────────────────

@bp.route("/projects/assistant")
def assistant_page():
    """Serve the chat UI. Sets a session cookie if not already set."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("assistant.html")


@bp.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle one chat message.

    Request:  POST /api/chat
              Content-Type: application/json
              { "message": "Does Aria know Docker?" }

    Response: { "answer": "Yes, Aria has...", "session_id": "uuid" }

    Errors:   400 if message is empty
              500 if agent throws (with error message)
    """
    data    = request.get_json()
    message = (data or {}).get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]
    agent      = _get_agent(session_id)

    try:
        answer = agent.chat(message)
        return jsonify({
            "answer":     answer,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/chat/reset", methods=["POST"])
def reset_chat():
    """
    Clear conversation history for the current session.
    Called when the user clicks "New conversation".
    Does not delete the agent — just wipes its history.
    """
    session_id = session.get("session_id")
    if session_id and session_id in _agents:
        _agents[session_id].reset()
    return jsonify({"status": "reset"})


@bp.route("/api/chat/status")
def status():
    """
    Return knowledge base status.
    Useful for debugging — shows chunk counts for both collections.
    Hit this in the browser to confirm ingestion worked.

    Example response:
    {
      "knowledge_chunks": 14,
      "code_chunks": 31,
      "embedding_model": "BAAI/bge-small-en-v1.5",
      "knowledge_top_k": 3,
      "code_top_k": 3,
      "llm_model": "gpt-3.5-turbo",
      "history_turns": 0
    }
    """
    try:
        # Create a fresh agent just for status — no session needed
        agent = PortfolioAgent()
        return jsonify(agent.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500