import os
from dotenv import load_dotenv

load_dotenv()  
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "customer-support-agent"
)
