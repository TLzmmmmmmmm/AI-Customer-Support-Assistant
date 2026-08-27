import os

from dotenv import load_dotenv


load_dotenv()


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is not set")


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

DEEPSEEK_TIMEOUT_SECONDS = 30.0
DEEPSEEK_MAX_RETRIES = 0

LLM_APP_MAX_RETRIES = 1
LLM_RETRY_DELAY_SECONDS = 0.5