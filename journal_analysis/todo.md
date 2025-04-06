# add config loading to build_database.py
import json

def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)

config = load_config()