# DIC03-ContAInnum

AI Agents for drug discovery: Research diseases, analyze compounds, and plan synthesis routes.

## Setup

### Prerequisites
- **Ollama** (v0.12+) with Mistral model - [Download](https://ollama.ai)
- **Node.js** (v20+) - [Download](https://nodejs.org)
- **Python** 3.12+ with pip

### Quick Start (2 Terminals)

**Terminal 1: Backend**
```cmd
run_backend.cmd
```

**Terminal 2: Frontend**
```cmd
run_frontend.cmd
```

Visit: **http://localhost:5173**

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/researcher/research` - Research diseases
- `POST /api/synthesis/analyze` - Analyze compounds
- `POST /api/integrated/research-and-manufacture` - Full workflow

API Docs: http://localhost:8000/docs

## Project Structure

```
DIC03-ContAInnum/
├── backend/              # FastAPI server
│   ├── app.py           # Main application
│   └── agents/          # AI agents
├── frontend/            # React + Vite
│   └── src/
├── run_backend.cmd      # Start backend
├── run_frontend.cmd     # Start frontend
└── SETUP_WINDOWS.md     # Detailed setup guide
```

## Troubleshooting

- Backend won't start? Make sure Ollama is running: `ollama serve`
- Frontend won't start? Install dependencies: `cd frontend && npm install`
- Port in use? Change port or kill the process using it

See SETUP_WINDOWS.md for detailed troubleshooting.
