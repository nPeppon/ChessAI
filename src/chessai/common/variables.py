import os

PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
LOG_PATH = os.path.join(PROJECT_ROOT, r"logs")
DATA_PATH = os.path.join(PROJECT_ROOT, r"data")
TEMP_PATH = os.path.join(DATA_PATH, r"temp")

# Ensure the directories exist
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)
