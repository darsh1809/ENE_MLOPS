"""
wsgi.py
-------
Flask application factory – entry point for Gunicorn / local dev.

Run locally:
    python wsgi.py

Run with Gunicorn (production):
    gunicorn --bind 0.0.0.0:5001 --workers 2 "wsgi:create_app()"
"""

import logging
import os

from flask import Flask, jsonify

from app.routes import api_bp


def create_app() -> Flask:
    """Application factory."""
    flask_app = Flask(__name__)

    # ── Logging ────────────────────────────────────────────────────────────
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, log_level, logging.INFO),
    )

    # ── Register blueprints ────────────────────────────────────────────────
    flask_app.register_blueprint(api_bp)

    # ── Global error handlers ──────────────────────────────────────────────
    @flask_app.errorhandler(404)
    def not_found(_err):
        return jsonify({"error": "Endpoint not found."}), 404

    @flask_app.errorhandler(405)
    def method_not_allowed(_err):
        return jsonify({"error": "Method not allowed."}), 405

    @flask_app.errorhandler(500)
    def internal_error(_err):
        return jsonify({"error": "Internal server error."}), 500

    return flask_app


# ── Local dev entry-point ──────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app  = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)
