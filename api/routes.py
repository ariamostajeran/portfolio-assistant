"""
Flask API routes for the portfolio assistant.

Endpoints:
  GET  /projects/assistant   — serves the chat UI page
  POST /api/chat             — send a message, get an answer
  POST /api/chat/reset       — clear conversation history
  GET  /api/chat/status      — knowledge base health check
"""

import os
import uuid
from flask import Blueprint, request, jsonify, render_template, session

from retrieval.retriever import Retriever
from agent.assistant import PortfolioAgent

bp = Blueprint("assistant", __name__)

# Single shared Retriever — loaded once at startup
_retriever = Retriever()


def _get_agent() -> PortfolioAgent:
    return PortfolioAgent(retriever=_retriever)


# ── Routes ────────────────────────────────────────────────────────────────────

@bp.route("/projects/assistant")
def assistant_page():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("assistant.html")


@bp.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = (data or {}).get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]

    try:
        agent  = _get_agent()
        answer = agent.chat(message)
        return jsonify({"answer": answer, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/chat/reset", methods=["POST"])
def reset_chat():
    return jsonify({"status": "reset"})


@bp.route("/api/chat/status")
def status():
    try:
        return jsonify(_retriever.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
