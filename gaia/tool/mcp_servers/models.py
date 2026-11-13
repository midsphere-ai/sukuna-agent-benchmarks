from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(verbose=True)
import os
import httpx

class ModelManager():
    def __init__(self):
        self.registered_models = {}

    def init_models(self):
        self._register_gpt55()

    def _register_gpt55(self):
        api_key = os.getenv("GRADIENT_API_KEY", "")
        base_url = os.getenv("GRADIENT_BASE_URL", "https://inference.do-ai.run/v1")
        model_id = "openai-gpt-5.5"
        model = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=8192,
            timeout=120,
            http_client=httpx.Client(verify=False),
            http_async_client=httpx.AsyncClient(verify=False),
        )
        self.registered_models[model_id] = model
