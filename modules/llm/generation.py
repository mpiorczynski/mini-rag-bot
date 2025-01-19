import requests

MODEL_NAME = "speakleash/Bielik-11B-v2.3-Instruct"
LLM_API_URL = "http://localhost:8084/v1/chat/completions"


def generate_response(
    prompt: list[dict[str, str]],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    **generation_kwargs,
) -> str:
    response = requests.post(
        LLM_API_URL,
        json={
            "model": model_name,
            "messages": prompt,
            **generation_kwargs,
        },
    )
    response.raise_for_status()
    return response.json()
