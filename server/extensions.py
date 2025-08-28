import os

from flask import Flask
from flask_socketio import SocketIO

# --- Global socketio Object ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__, template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))
app.config['SECRET_KEY'] = 'secret-emotion-key!'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# --- Global Settings Configuration ---
LLM_API_KEY = None
DEFAULT_LLM_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_LOCAL_LLM_MODEL_NAME = "tiiuae/falcon-7b-instruct"

# --- Shared Application State ---
APP_STATE = {
    "local_model_ready": False
}