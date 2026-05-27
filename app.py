import os
from flask import Flask
from api.routes import bp as assistant_bp

def create_app():
    app = Flask(__name__, template_folder="templates")
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
    app.register_blueprint(assistant_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)