import os
import re
from typing import Iterable, Optional

import requests


DEFAULT_PROMPT_MODEL = os.getenv("HF_PROMPT_MODEL", "meta-llama/Llama-3.1-8B-Instruct:fastest")
DEFAULT_LLM_PROVIDER = os.getenv("HF_LLM_PROVIDER", "fastest")
CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


def get_api_token() -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HUGGINGFACE_API_TOKEN is missing. Add it in backend/.env or set it in your terminal."
        )

    return token


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_styles(styles: Iterable[str]) -> list[str]:
    cleaned_list = []

    for style in styles or []:
        clean_style = clean_text(str(style)).lower()
        if clean_style and clean_style not in cleaned_list:
            cleaned_list.append(clean_style)

    return cleaned_list


def create_style_text(styles: list[str], color: Optional[str]) -> str:
    parts = list(styles)

    if color:
        parts.append(f"dominant color {color}")

    return ", ".join(parts)


def trim_prompt_text(prompt_text: str) -> str:
    clean_prompt = clean_text(prompt_text)
    clean_prompt = clean_prompt.replace("[INST]", "").replace("[/INST]", "").strip(" \"'")

    words = clean_prompt.split()
    if len(words) > 75:
        clean_prompt = " ".join(words[:75])

    return clean_prompt


def create_llama_message(context: str, style_text: str) -> str:
    return f"""You are helping with an AI icon generator.

Write one short image generation prompt for a logo or icon.

Context: {context}
Styles: {style_text or "clean, modern"}

Rules:
- Start with the main subject
- Include the requested styles
- Mention centered composition
- Mention professional quality
- Keep it under 75 words
- Output only the prompt text
"""


def get_router_model_name() -> str:
    if ":" in DEFAULT_PROMPT_MODEL:
        return DEFAULT_PROMPT_MODEL

    return f"{DEFAULT_PROMPT_MODEL}:{DEFAULT_LLM_PROVIDER}"


def raise_api_error(response: requests.Response):
    try:
        error_data = response.json()
    except Exception:
        error_data = response.text

    raise RuntimeError(
        f"Hugging Face Llama API request failed ({response.status_code}): {error_data}"
    )


def ask_llama_directly(context: str, styles: list[str], color: Optional[str]) -> str:
    response = requests.post(
        CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {get_api_token()}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_router_model_name(),
            "messages": [
                {
                    "role": "user",
                    "content": create_llama_message(clean_text(context), create_style_text(styles, color)),
                }
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        },
        timeout=60,
    )

    if not response.ok:
        raise_api_error(response)

    data = response.json()
    prompt_text = data["choices"][0]["message"]["content"]
    return trim_prompt_text(prompt_text)


def ask_llama_with_langchain(context: str, styles: list[str], color: Optional[str]) -> str:
    raw_model_name = DEFAULT_PROMPT_MODEL.split(":", 1)[0]

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import HuggingFaceHub

    prompt_template = PromptTemplate(
        input_variables=["context", "style_text"],
        template="""
You are helping with an AI icon generator.

Write one short image generation prompt for a logo or icon.

Context: {context}
Styles: {style_text}

Rules:
- Start with the main subject
- Include the requested styles
- Mention centered composition
- Mention professional quality
- Keep it under 75 words
- Output only the prompt text
""".strip(),
    )

    model = HuggingFaceHub(
        repo_id=raw_model_name,
        huggingfacehub_api_token=get_api_token(),
        model_kwargs={"temperature": 0.7, "max_new_tokens": 200},
    )

    chain = LLMChain(llm=model, prompt=prompt_template)
    prompt_text = chain.run(
        {
            "context": clean_text(context),
            "style_text": create_style_text(styles, color),
        }
    )

    return trim_prompt_text(prompt_text)


def get_prompt_engine_mode() -> str:
    try:
        get_api_token()
    except RuntimeError:
        return "llama-api-not-configured"

    if ":" in DEFAULT_PROMPT_MODEL:
        return "llama-api-direct"

    try:
        from langchain.chains import LLMChain  # noqa: F401
        from langchain.prompts import PromptTemplate  # noqa: F401
        from langchain_community.llms import HuggingFaceHub  # noqa: F401
        return "langchain-llama"
    except Exception:
        return "llama-api-direct"


def engineer_prompt(context, styles, color=None):
    cleaned_styles = clean_styles(styles)
    mode = get_prompt_engine_mode()

    if mode == "llama-api-not-configured":
        raise RuntimeError(
            "HUGGINGFACE_API_TOKEN is missing. Add it in backend/.env or set it in your terminal."
        )

    if mode == "langchain-llama":
        try:
            prompt_text = ask_llama_with_langchain(context, cleaned_styles, color)
            if prompt_text:
                return prompt_text
        except Exception as error:
            print(f"LangChain prompt step failed, using direct Hugging Face API instead: {error}")

    prompt_text = ask_llama_directly(context, cleaned_styles, color)

    if not prompt_text:
        raise RuntimeError("The prompt model returned an empty result.")

    return prompt_text


if __name__ == "__main__":
    sample_prompt = engineer_prompt(
        context="coffee shop logo",
        styles=["vector", "minimalist", "modern"],
        color="#8B4513",
    )
    print(sample_prompt)
