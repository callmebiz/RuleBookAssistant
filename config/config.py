# config/config.py

import os
import json

def load_environment():
    keys = json.load(open("config/keys.json"))
    os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = keys["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "RuleBookAssistant"