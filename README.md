---
title: LLM Analysis Quiz Solver
emoji: 🃏
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# LLM Analysis - Autonomous Quiz Solver Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121.3+-green.svg)](https://fastapi.tiangolo.com/)

An intelligent, autonomous agent built with LangGraph and LangChain that solves data-related quizzes involving web scraping, data processing, OCR, audio transcription, image encoding, and various analysis tasks. The system uses Google's Gemini 2.5 Flash model to orchestrate tool usage and make decisions autonomously.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Tools & Capabilities](#tools--capabilities)
- [Advanced Features](#advanced-features)
- [Docker Deployment](#docker-deployment)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🔍 Overview

This project was developed for the TDS (Tools in Data Science) course project, where the objective is to build an application that can autonomously solve multi-step quiz tasks involving:

- **Data sourcing**: Scraping websites, calling APIs, downloading files (images, PDFs, audio)
- **Data preparation**: Cleaning text, processing images with OCR, transcribing audio
- **Data analysis**: Filtering, aggregating, statistical analysis, code execution
- **Data visualization**: Generating charts, narratives, and presentations
- **Image processing**: Converting images to base64 for submissions
- **Audio processing**: Transcribing audio files to text

The system receives quiz URLs via a REST API, navigates through multiple quiz pages, solves each task using LLM-powered reasoning and specialized tools, and submits answers back to the evaluation server. It includes intelligent retry mechanisms, timeout handling, and malformed JSON recovery.

## 🗃️ Architecture

The project uses a **LangGraph state machine** architecture with the following components:

```
┌─────────────┐
│   FastAPI   │  ← Receives POST requests with quiz URLs
│   Server    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Agent     │  ← LangGraph orchestrator with Gemini 2.5 Flash
│   (LLM)     │     + Malformed JSON handler
└──────┬──────┘     + Timeout manager
       │            + Message trimmer (60K tokens)
       │
       ├────────────┬────────────┬─────────────┬──────────────┬─────────────┐
       ▼            ▼            ▼             ▼              ▼             ▼
   [Scraper]   [Downloader]  [Code Exec]  [POST Req]   [OCR Image]  [Transcribe]
                                                         [Encode B64]  [Add Deps]
```

### Key Components:

1. **FastAPI Server** (`main.py`): Handles incoming POST requests, validates secrets, triggers agent in background
2. **LangGraph Agent** (`agent.py`): State machine that coordinates tool usage, handles timeouts, recovers from errors
3. **Tools Package**: Modular tools for different capabilities (8 specialized tools)
4. **Shared Store** (`shared_store.py`): Global state for base64 storage and URL timing
5. **LLM**: Google Gemini 2.5 Flash with rate limiting (4 requests per minute)

## ✨ Features

- ✅ **Autonomous multi-step problem solving**: Chains together multiple quiz pages
- ✅ **Dynamic JavaScript rendering**: Uses Playwright for client-side rendered pages
- ✅ **Code generation & execution**: Writes and runs Python code for data tasks
- ✅ **OCR capabilities**: Extracts text from images using Tesseract
- ✅ **Audio transcription**: Converts MP3/WAV files to text using Google Speech API
- ✅ **Image encoding**: Converts images to base64 with memory-efficient storage
- ✅ **Flexible data handling**: Downloads files, processes PDFs, CSVs, images, audio
- ✅ **Self-installing dependencies**: Automatically adds required Python packages
- ✅ **Intelligent retry mechanism**: Retries failed attempts with 4-attempt limit
- ✅ **Timeout management**: 180-second per-quiz timeout with auto-skip on failure
- ✅ **Malformed JSON recovery**: Automatically detects and recovers from invalid LLM outputs
- ✅ **Message trimming**: Maintains conversation history under 60K tokens
- ✅ **Docker containerization**: Ready for deployment on HuggingFace Spaces or cloud platforms
- ✅ **Rate limiting**: Respects API quotas with in-memory rate limiter (4 req/min)

## 📁 Project Structure

```
llm-analysis-tds-project/
├── agent.py                    # LangGraph state machine & orchestration
│                                 - Agent node with timeout handling
│                                 - Malformed JSON recovery node
│                                 - Smart routing logic
│                                 - Message trimming (60K tokens)
├── main.py                     # FastAPI server with /solve endpoint
│                                 - Background task execution
│                                 - Health check endpoint
│                                 - State cleanup between requests
├── shared_store.py             # Global state management
│                                 - BASE64_STORE for image encodings
│                                 - url_time for timeout tracking
├── tools/
│   ├── __init__.py
│   ├── web_scraper.py          # Playwright-based HTML renderer
│   ├── run_code.py             # Python code executor with uv
│   ├── download_file.py        # File downloader (PDF, CSV, images, audio)
│   ├── send_request.py         # HTTP POST tool with retry logic
│   ├── add_dependencies.py     # Dynamic package installer
│   ├── image_content_extracter.py  # OCR using Tesseract
│   ├── audio_transcribing.py   # Audio to text transcription
│   └── encode_image_to_base64.py   # Base64 encoder with placeholder system
├── pyproject.toml              # Project dependencies & configuration
├── Dockerfile                  # Container image with Playwright
├── .env                        # Environment variables (not in repo)
└── README.md
```

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Git
- Tesseract OCR (for image text extraction)
- FFmpeg (for audio processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ashwinambatwariitm/llm-analysis-tds-project.git
cd llm-analysis-tds-project
```

### Step 2: Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg
```

#### macOS
```bash
brew install tesseract ffmpeg
```

#### Windows
Download and install:
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [FFmpeg](https://ffmpeg.org/download.html)

### Step 3: Install Python Dependencies

#### Option A: Using `uv` (Recommended)

Ensure you have uv installed, then sync the project:

```bash
# Install uv if you haven't already  
pip install uv

# Sync dependencies  
uv sync
uv run playwright install chromium
```

Start the FastAPI server:
```bash
uv run main.py
```
The server will start at `http://0.0.0.0:7860`.

#### Option B: Using `pip`

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e .

# Install Playwright browsers
playwright install chromium
```

#### Option C: Using `uv` and requirements.txt

```bash
# Install uv if you haven't already  
pip install uv

# Make sure you're in the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Your credentials from the course registration
EMAIL=your.email@example.com
SECRET=your_secret_string

# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy it to your `.env` file

## 🚀 Usage

### Local Development

Start the FastAPI server:

```bash
# If using uv
uv run main.py

Or

# If using standard Python
python main.py
```

The server will start on `http://0.0.0.0:7860`

### Testing the Endpoint

Send a POST request to test your setup:

```bash
curl -X POST http://localhost:7860/solve \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your.email@example.com",
    "secret": "your_secret_string",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

Expected response:

```json
{
  "status": "ok"
}
```

The agent will run in the background and solve the quiz chain autonomously.

## 🌐 API Endpoints

### `POST /solve`

Receives quiz tasks and triggers the autonomous agent.

**Request Body:**

```json
{
    "email": "your.email@example.com",
    "secret": "your_secret_string",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

**Responses:**

| Status Code | Description                    |
| ----------- | ------------------------------ |
| `200`       | Secret verified, agent started |
| `400`       | Invalid JSON payload           |
| `403`       | Invalid secret                 |

### `GET /healthz`

Health check endpoint for monitoring and Hugging Face Spaces compatibility.

**Response:**

```json
{
  "status": "ok",
  "uptime_seconds": 3600
}
```

## 🛠️ Tools & Capabilities

The agent has access to 8 specialized tools:

### 1. **Web Scraper** (`get_rendered_html`)

- Uses Playwright to render JavaScript-heavy pages
- Waits for network idle before extracting content
- Returns fully rendered HTML + list of image URLs
- Automatically truncates HTML > 300KB to prevent memory issues
- **Use case**: Loading quiz pages, extracting instructions

### 2. **File Downloader** (`download_file`)

- Downloads files (PDFs, CSVs, images, audio, etc.) from direct URLs
- Saves files to `LLMFiles/` directory
- Returns the saved filename for further processing
- **Use case**: Getting data files, images for OCR, audio for transcription

### 3. **Code Executor** (`run_code`)

- Executes arbitrary Python code in isolated subprocess
- Uses `uv run` for dependency management
- Returns stdout, stderr, and exit code
- Automatically truncates output > 10KB
- **Use case**: Data processing, analysis, calculations, algorithm challenges

### 4. **POST Request** (`post_request`)

- Sends JSON payloads to quiz submission endpoints
- Includes automatic BASE64_KEY placeholder replacement
- Intelligent retry logic with 4-attempt limit
- Timeout enforcement (180s per quiz, 90s retry cooldown)
- Extracts next URL from responses and updates state
- **Use case**: Submitting answers, navigating to next quiz

### 5. **Dependency Installer** (`add_dependencies`)

- Dynamically installs Python packages as needed
- Uses `uv add` for fast package resolution
- Enables the agent to adapt to different task requirements
- **Use case**: Installing numpy, pandas, matplotlib, etc. on-demand

### 6. **OCR Image Tool** (`ocr_image_tool`)

- Extracts text from images using Tesseract OCR
- Supports multiple input formats: bytes, file paths, base64, PIL images
- Configurable language support (default: English)
- **Use case**: Reading text from screenshots, images, captchas

### 7. **Audio Transcription** (`transcribe_audio`)

- Converts MP3/WAV audio files to text
- Automatic MP3 to WAV conversion using pydub
- Uses Google Speech Recognition API (free tier)
- Cleans up temporary files after processing
- **Use case**: Transcribing voice questions, audio challenges

### 8. **Image to Base64 Encoder** (`encode_image_to_base64`)

- Converts images to base64 encoding
- **Memory-efficient design**: Stores base64 in separate `BASE64_STORE`
- Returns lightweight placeholder: `BASE64_KEY:<uuid>`
- Prevents conversation overflow (100KB+ base64 becomes 50-char placeholder)
- Automatically swapped by `post_request` before submission
- **Use case**: Preparing images for upload in quiz answers

## 🎯 Advanced Features

### Timeout Management

The system implements a sophisticated timeout mechanism:

1. **Per-Quiz Timeout** (180 seconds):
   - Tracked in `url_time` dictionary
   - If exceeded: agent submits wrong answer to skip and move on
   - Prevents infinite loops on unsolvable quizzes

2. **Retry Cooldown** (90 seconds):
   - After wrong answer, waits before retrying same quiz
   - Tracked using `offset` environment variable
   - Prevents rapid-fire incorrect submissions

3. **Retry Limit** (4 attempts):
   - Maximum 4 attempts per quiz
   - After limit: moves to next quiz regardless of correctness

### Malformed JSON Recovery

Handles LLM output errors gracefully:

```python
# Detects when LLM generates invalid JSON in tool calls
if finish_reason == "MALFORMED_FUNCTION_CALL":
    # Sends correction message to LLM
    # LLM sees error and retries with proper escaping
    # Prevents agent crashes
```

**Common issues caught**:
- Unescaped quotes in strings
- Missing commas in JSON
- Incorrect data types

### Message Trimming

Prevents memory overflow in long conversations:

```python
# Keeps only most recent 60,000 tokens
trimmed_messages = trim_messages(
    messages=state["messages"],
    max_tokens=60000,
    strategy="last",      # Keep recent, discard old
    include_system=True,  # Always preserve instructions
    start_on="human"      # Ensure valid conversation structure
)
```

**Benefits**:
- Handles 50+ quiz chains without memory issues
- Maintains conversation context
- Faster LLM inference
- Lower API costs

### Base64 Storage Optimization

Solves the "large binary data in conversation" problem:

**Problem**: Base64-encoded images can be 100,000+ characters
- Overwhelms LLM context
- Causes malformed tool calls
- Breaks routing logic

**Solution**: Placeholder system
1. Encode image → Store in `BASE64_STORE[uuid]`
2. Return placeholder: `BASE64_KEY:abc-123-def-456`
3. LLM uses placeholder (only 50 chars)
4. `post_request` swaps placeholder with real base64

**Result**: 2000x size reduction in conversation memory

### Smart Routing

The agent uses conditional routing based on LLM output:

```
Last message analysis:
├─ Malformed JSON? → handle_malformed → retry
├─ Has tool calls? → tools → execute → agent
├─ Says "END"? → END (terminate)
└─ Otherwise → agent (continue thinking)
```

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t llm-analysis-agent .
```

### Run the Container

```bash
docker run -p 7860:7860 \
  -e EMAIL="your.email@example.com" \
  -e SECRET="your_secret_string" \
  -e GOOGLE_API_KEY="your_api_key" \
  llm-analysis-agent
```

### Deploy to HuggingFace Spaces

1. Create a new Space with Docker SDK
2. Push this repository to your Space
3. Add secrets in Space settings:
   - `EMAIL`
   - `SECRET`
   - `GOOGLE_API_KEY`
4. The Space will automatically build and deploy

**Dockerfile includes**:
- Playwright with Chromium
- Tesseract OCR
- FFmpeg for audio processing
- All Python dependencies
- Health check endpoint for monitoring

## 🧠 How It Works

### 1. Request Reception

- FastAPI receives POST request with quiz URL
- Validates secret against environment variables
- Clears previous state (`url_time`, `BASE64_STORE`)
- Sets environment variables (`url`, `offset`)
- Returns 200 OK immediately
- Starts agent in background task (non-blocking)

### 2. Agent Initialization

- LangGraph creates state machine with 3 nodes: `agent`, `tools`, `handle_malformed`
- Initial state contains system prompt + quiz URL
- `url_time` records start timestamp for timeout tracking

### 3. Task Loop

The agent follows this intelligent loop:

```
┌─────────────────────────────────────────────┐
│ 1. Agent Node                               │
│    - Check timeout (180s / 90s cooldown)    │
│    - Trim messages to 60K tokens            │
│    - LLM analyzes state & plans tools       │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 2. Router                                   │
│    - Malformed? → handle_malformed          │
│    - Tool calls? → tools                    │
│    - END signal? → terminate                │
│    - Otherwise? → agent (more thinking)     │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 3. Tool Execution                           │
│    - Scrape page / download files           │
│    - Run OCR / transcribe audio             │
│    - Execute code / install dependencies    │
│    - Submit answer via POST                 │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 4. Response Analysis                        │
│    - Check if answer correct                │
│    - Extract next URL (if exists)           │
│    - Decide: retry / continue / end         │
│    - Update url_time for timeout tracking   │
└──────────────────┬──────────────────────────┘
                   │
                   └─→ Loop back to step 1
```

### 4. State Management

- All messages (user, assistant, tool calls, tool results) stored in state
- LLM uses full history to make informed decisions
- Recursion limit: 5000 steps (prevents infinite loops)
- Message trimming: Keeps only 60K tokens to prevent overflow

### 5. Error Handling

**Three layers of error recovery**:

1. **Malformed JSON**: Detected by finish_reason, sends correction, allows retry
2. **Timeout Management**: Forces wrong answer submission after 180s
3. **Retry Limits**: Maximum 4 attempts per quiz, then moves on

### 6. Completion

- Agent outputs "END" when no new URL in response
- Background task completes
- Logs indicate success or failure
- All state cleared for next request

## 🔧 Troubleshooting

### Common Issues

**Problem**: Agent gets stuck on a quiz
- **Cause**: Quiz might be unsolvable or require manual intervention
- **Solution**: Timeout mechanism (180s) will auto-submit wrong answer and move on

**Problem**: "Malformed JSON" errors persist
- **Cause**: LLM struggling with complex tool call formatting
- **Solution**: Recovery node sends correction; usually resolves in 1-2 retries

**Problem**: Memory overflow / context too long
- **Cause**: Long quiz chains (50+ quizzes)
- **Solution**: Message trimming automatically keeps only 60K tokens

**Problem**: Base64 images causing errors
- **Cause**: Large base64 strings in conversation
- **Solution**: Placeholder system (`BASE64_KEY`) handles this automatically

**Problem**: Rate limit errors from Gemini API
- **Cause**: Too many requests too quickly
- **Solution**: InMemoryRateLimiter restricts to 4 req/min with token bucket

**Problem**: Playwright/Chromium crashes
- **Cause**: Memory issues in container
- **Solution**: Ensure Docker container has sufficient memory (2GB+)

## 📊 System Limits

- **Max tokens per conversation**: 60,000 (auto-trimmed)
- **Max recursion depth**: 5,000 steps
- **Timeout per quiz**: 180 seconds
- **Retry cooldown**: 90 seconds
- **Max retries**: 4 attempts per quiz
- **Rate limit**: 4 Gemini API requests per minute
- **HTML size limit**: 300KB (auto-truncated)
- **Code output limit**: 10KB (auto-truncated)

## 📝 Key Design Decisions

1. **LangGraph State Machine**: Provides better control flow than sequential chains
2. **Background Task Processing**: Prevents HTTP timeouts during long quiz chains
3. **Tool Modularity**: Each tool independent, testable, debuggable separately
4. **Rate Limiting**: Prevents API quota exhaustion with token bucket algorithm
5. **Playwright for Scraping**: Handles JavaScript-rendered pages that requests cannot
6. **Base64 Placeholder System**: Separates large binary data from conversation
7. **Message Trimming**: Maintains performance and prevents memory overflow
8. **Malformed JSON Recovery**: Graceful error handling instead of crashes
9. **Timeout with Auto-Skip**: Ensures progress even on unsolvable quizzes
10. **uv for Package Management**: Fast dependency resolution and installation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author**: Ashvin Ambatwar  
**Course**: Tools in Data Science (TDS)  
**Institution**: IIT Madras  

**Repository**: [https://github.com/ashwinambatwariitm/llm-analysis-tds-project](https://github.com/ashwinambatwariitm/llm-analysis-tds-project)

For questions or issues, please open an issue on the GitHub repository.
