import os
from flask import Blueprint, request, jsonify, render_template

from retrieval.retriever import Retriever
from agent.assistant import PortfolioAgent

bp = Blueprint("assistant", __name__)

# Single shared Retriever — loaded once at startup, read-only after that
_retriever = Retriever()


# ── Routes ────────────────────────────────────────────────────────────────────

@bp.route("/projects/assistant")
def assistant_page():
    return render_template("assistant.html")


@bp.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = (data or {}).get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        agent  = PortfolioAgent(retriever=_retriever)
        answer = agent.chat(message)
        return jsonify({"answer": answer})
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
