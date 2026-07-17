from deepeval.models.base_model import DeepEvalBaseLLM
from common.clients.litellm import completion_with_fallback
import asyncio

class LiteLLMDeepEvalWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini/gemini-3.5-flash"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        async def call():
            resp = await completion_with_fallback(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        return asyncio.run(call())

    async def a_generate(self, prompt: str) -> str:
        resp = await completion_with_fallback(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    def get_model_name(self):
        return self.model_name
