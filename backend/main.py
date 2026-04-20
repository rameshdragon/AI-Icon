import json
import os
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from image_generator import generate_icons, generator
from prompt_engineer import engineer_prompt, get_prompt_engine_mode


BASE_DIR = Path(__file__).resolve().parent
SCRAPER_FILE = BASE_DIR / "scraper.js"

STYLE_OPTIONS = {
    "icon_types": [
        "flat",
        "3d",
        "vector",
        "hand-drawn",
        "minimalist",
        "geometric",
        "abstract",
        "isometric",
    ],
    "finishes": [
        "matte",
        "glossy",
        "metallic",
        "shiny",
    ],
    "styles": [
        "modern",
        "retro",
        "cartoon",
        "realistic",
        "gradient",
        "monochrome",
    ],
}


def load_local_env():
    """Load .env values from backend/.env or the project root .env file."""
    possible_files = [BASE_DIR / ".env", BASE_DIR.parent / ".env"]

    for env_file in possible_files:
        if not env_file.exists():
            continue

        lines = env_file.read_text(encoding="utf-8").splitlines()
        for raw_line in lines:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line.replace("export ", "", 1)

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if key and value:
                os.environ.setdefault(key, value)


load_local_env()


app = FastAPI(title="AI Icon Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ManualGenerateRequest(BaseModel):
    prompt: str
    styles: List[str]
    color: Optional[str] = None


class URLGenerateRequest(BaseModel):
    url: HttpUrl
    styles: List[str]
    color: Optional[str] = None


class GenerateResponse(BaseModel):
    optimized_prompt: str
    images: List[str]


class PageContentParser(HTMLParser):
    """Small HTML parser used when Puppeteer is not available."""

    def __init__(self):
        super().__init__()
        self.title_text = []
        self.heading_text = []
        self.page_text = []
        self.description = ""
        self.keywords = ""
        self.inside_title = False
        self.inside_h1 = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs}

        if tag == "title":
            self.inside_title = True
        elif tag == "h1":
            self.inside_h1 = True
        elif tag == "meta":
            meta_name = (attributes.get("name") or attributes.get("property") or "").lower()
            meta_content = (attributes.get("content") or "").strip()

            if meta_name in {"description", "og:description"} and meta_content and not self.description:
                self.description = meta_content

            if meta_name == "keywords" and meta_content and not self.keywords:
                self.keywords = meta_content

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag == "title":
            self.inside_title = False
        elif tag == "h1":
            self.inside_h1 = False

    def handle_data(self, data):
        clean_text = " ".join(data.split())

        if not clean_text:
            return

        if self.inside_title:
            self.title_text.append(clean_text)

        if self.inside_h1:
            self.heading_text.append(clean_text)

        current_page_text = " ".join(self.page_text)
        if len(current_page_text) < 2000:
            self.page_text.append(clean_text)

    def to_dict(self):
        return {
            "title": " ".join(self.title_text).strip(),
            "description": self.description,
            "keywords": self.keywords,
            "h1": " ".join(self.heading_text).strip(),
            "text": " ".join(self.page_text).strip()[:2000],
        }


def run_puppeteer_scraper(url: str):
    """Use the Node scraper first because it captures rendered websites better."""
    node_path = shutil.which("node")
    if not node_path or not SCRAPER_FILE.exists():
        raise RuntimeError("Puppeteer scraper is not available")

    result = subprocess.run(
        [node_path, str(SCRAPER_FILE), url],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=60,
    )

    stdout_text = (result.stdout or "").strip()
    stderr_text = (result.stderr or "").strip()

    if result.returncode != 0:
        if stderr_text:
            try:
                error_data = json.loads(stderr_text)
                raise RuntimeError(error_data.get("error", stderr_text))
            except json.JSONDecodeError:
                raise RuntimeError(stderr_text)

        raise RuntimeError("Puppeteer scraper failed")

    if not stdout_text:
        raise RuntimeError("Puppeteer scraper returned empty output")

    return json.loads(stdout_text)


def run_python_scraper(url: str):
    """Simple fallback scraper for basic websites."""
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AIIconGenerator/1.0; +http://localhost)"},
    )

    with urlopen(request, timeout=20) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="replace")

    parser = PageContentParser()
    parser.feed(html)
    return parser.to_dict()


def get_website_data(url: str):
    try:
        return run_puppeteer_scraper(url)
    except Exception as error:
        print(f"Puppeteer scraper failed, using Python fallback: {error}")
        return run_python_scraper(url)


def build_context_text(page_data):
    """Turn scraped website data into one clean string for the prompt model."""
    parts = []

    if page_data.get("title"):
        parts.append(f"Title: {page_data['title']}")

    if page_data.get("description"):
        parts.append(f"Description: {page_data['description']}")

    if page_data.get("keywords"):
        parts.append(f"Keywords: {page_data['keywords']}")

    if page_data.get("h1"):
        parts.append(f"Heading: {page_data['h1']}")

    if page_data.get("text"):
        parts.append(f"Body: {page_data['text'][:400]}")

    return " | ".join(parts)


@app.on_event("startup")
async def startup_event():
    generator.prepare_environment()
    print("Starting AI Icon Generator API...")
    print(f"Prompt engine mode: {get_prompt_engine_mode()}")
    print("Image generation uses the hosted Stable Diffusion API.")


@app.get("/")
def read_root():
    return {
        "message": "AI Icon Generator API",
        "version": "1.0.0",
        "endpoints": {
            "manual_generate": "/api/generate/manual",
            "url_generate": "/api/generate/url",
            "styles": "/api/styles",
            "health": "/health",
        },
    }


@app.get("/api/styles")
def get_styles():
    return STYLE_OPTIONS


@app.post("/api/generate/manual", response_model=GenerateResponse)
async def generate_from_manual(request: ManualGenerateRequest):
    try:
        print(f"Manual generation request: {request.prompt}")

        optimized_prompt = engineer_prompt(
            context=request.prompt,
            styles=request.styles,
            color=request.color,
        )

        generated_images = generate_icons(optimized_prompt, count=4)

        return GenerateResponse(
            optimized_prompt=optimized_prompt,
            images=generated_images,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/generate/url", response_model=GenerateResponse)
async def generate_from_url(request: URLGenerateRequest):
    try:
        print(f"URL generation request: {request.url}")

        page_data = get_website_data(str(request.url))
        context_text = build_context_text(page_data)

        optimized_prompt = engineer_prompt(
            context=context_text,
            styles=request.styles,
            color=request.color,
        )

        generated_images = generate_icons(optimized_prompt, count=4)

        return GenerateResponse(
            optimized_prompt=optimized_prompt,
            images=generated_images,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Website scraping timed out")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "prompt_engine_mode": get_prompt_engine_mode(),
        "image_mode": generator.mode,
        "image_model": generator.model_id,
        "image_api_configured": bool(os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")),
        "last_model_error": generator.load_error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
