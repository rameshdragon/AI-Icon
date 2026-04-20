import base64
import hashlib
import io
import os

import requests
from PIL import Image


DEFAULT_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium-diffusers")
DEFAULT_IMAGE_PROVIDER = os.getenv("HF_IMAGE_PROVIDER", "hf-inference")
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, duplicate, text, watermark, signature, cropped"
)


def get_api_token() -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HUGGINGFACE_API_TOKEN is missing. Add it in backend/.env or set it in your terminal."
        )

    return token


def create_seed(prompt: str, image_number: int) -> int:
    seed_text = f"{prompt}-{image_number}"
    seed_hash = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(seed_hash[:8], 16)


class ImageGenerator:
    def __init__(self):
        self.mode = "stable-diffusion-api-not-configured"
        self.load_error = None
        self.model_id = DEFAULT_IMAGE_MODEL

    def prepare_environment(self):
        """This project uses hosted models, so nothing is needed here."""

    def get_router_url(self) -> str:
        return f"https://router.huggingface.co/{DEFAULT_IMAGE_PROVIDER}/models/{self.model_id}"

    def convert_image_to_base64(self, raw_image_bytes) -> str:
        image = Image.open(io.BytesIO(raw_image_bytes))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def raise_api_error(self, response: requests.Response):
        try:
            error_data = response.json()
        except Exception:
            error_data = response.text

        raise RuntimeError(
            f"Hugging Face image API request failed ({response.status_code}): {error_data}"
        )

    def request_single_image(self, prompt: str, image_number: int) -> str:
        response = requests.post(
            self.get_router_url(),
            headers={"Authorization": f"Bearer {get_api_token()}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 30,
                    "seed": create_seed(prompt, image_number),
                },
            },
            timeout=180,
        )

        if not response.ok:
            self.raise_api_error(response)

        return self.convert_image_to_base64(response.content)

    def generate_images(self, prompt: str, num_images: int = 4) -> list[str]:
        self.mode = "stable-diffusion-api"
        image_list = []

        print(f"Generating {num_images} images with hosted Stable Diffusion for prompt: {prompt}")

        for image_number in range(num_images):
            try:
                image_base64 = self.request_single_image(prompt, image_number)
                image_list.append(image_base64)
            except Exception as error:
                self.mode = "stable-diffusion-api-error"
                self.load_error = str(error)
                raise RuntimeError(str(error)) from error

        return image_list


generator = ImageGenerator()


def generate_icons(prompt, count=4):
    return generator.generate_images(prompt, count)


if __name__ == "__main__":
    sample_images = generate_icons(
        "coffee cup icon, vector art, minimalist, professional",
        count=1,
    )
    print(f"Generated {len(sample_images)} image(s)")
