# AI Icon Generator

AI-powered icon and logo generation system using Llama 2 for prompt optimization and Stable Diffusion XL for image generation. Built with a React frontend and FastAPI backend. Both models are used through hosted Hugging Face APIs, so the app does not download local model weights when you run it.

## What It Does

User provides a text prompt or website URL plus style preferences -> Llama 2 optimizes the prompt -> Stable Diffusion XL generates 4 unique icons -> User downloads the ones they like.

## Tech Stack

**Frontend:** React, CSS, Axios  
**Backend:** FastAPI, LangChain, Hugging Face  
**Models:** Llama 2 (7B), Stable Diffusion XL  
**Scraper:** Puppeteer (Node.js)

## How to Run

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set your Hugging Face API token:
```powershell
$env:HUGGINGFACE_API_TOKEN="your_token_here"
```
Use a token that has permission to call Hugging Face Inference Providers.

3. Install Node.js dependencies for the scraper:
```bash
npm install
```

4. Start the FastAPI server:
```bash
python main.py
```

Backend runs on `http://localhost:8000`

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the React dev server:
```powershell
npm.cmd start
```

Frontend runs on `http://localhost:3000`

## How It Works

### Manual Mode
1. User enters a prompt like `coffee shop logo`
2. User selects style guardrails
3. User picks a dominant color
4. Llama 2 optimizes the prompt via LangChain and the hosted Hugging Face API
5. Stable Diffusion XL generates 4 images through the hosted Hugging Face API
6. Images are returned as base64 to the frontend

### URL Mode
1. User pastes a website URL
2. Puppeteer scrapes title, description, and page content
3. Context is extracted and sent to Llama 2 with style guardrails
4. Llama 2 creates an optimized prompt
5. Stable Diffusion XL generates 4 images through the hosted Hugging Face API
6. User downloads the preferred icon

## API Endpoints

- `GET /api/styles` - Returns available style options
- `POST /api/generate/manual` - Generate from text prompt
- `POST /api/generate/url` - Generate from website URL
- `GET /health` - Health check

## Project Structure

```text
ai_icon_generator/
|-- backend/
|   |-- main.py
|   |-- prompt_engineer.py
|   |-- image_generator.py
|   |-- scraper.js
|   |-- requirements.txt
|   `-- package.json
`-- frontend/
    |-- src/
    |   |-- App.js
    |   |-- App.css
    |   |-- index.js
    |   `-- index.css
    |-- public/
    |   `-- index.html
    `-- package.json
```

## Requirements

- Python 3.8+
- Node.js 16+
- Hugging Face API token
- Token permission to call Hugging Face Inference Providers
- Internet access for hosted model inference

## Note

This project uses hosted Hugging Face inference for both Llama 2 and Stable Diffusion XL. Running the app should not download local model weights or require multi-GB model storage on your machine.
