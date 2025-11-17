# DIC03-ContAInnum

AI Agents for drug discovery: Research diseases, analyze compounds, and plan synthesis routes.

## Setup

### Prerequisites
- **Ollama** (v0.12+) with `llama3.1` model - [Download](https://ollama.ai)
  - After installing: `ollama pull llama3.1`
  - Start server: `ollama serve`
- **Node.js** (v20+) - [Download](https://nodejs.org)
- **Python** 3.12+ with pip
- **GPU recommended** for ChemLLM (RTX 3050 or better)

### Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/AidanSYu/DIC03-ContAInnum.git
   cd DIC03-ContAInnum
   ```

2. **Backend Setup** (PowerShell)
   ```powershell
   # Create Python virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   .venv\Scripts\Activate.ps1
   
   # Install dependencies
   pip install -r backend/requirements.txt
   pip install transformers accelerate torch sentencepiece einops
   ```

3. **Frontend Setup** (PowerShell)
   ```powershell
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

You need **3 terminals** (PowerShell) running simultaneously:

**Terminal 1: Ollama Server** (must be running first)
```powershell
ollama serve
```

**Terminal 2: Backend Server**
```powershell
cd "C:\path\to\DIC03-ContAInnum"
.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --reload --port 8000
```
Backend will be at: **http://127.0.0.1:8000**

**Terminal 3: Frontend Dev Server**
```powershell
cd "C:\path\to\DIC03-ContAInnum\frontend"
npm run dev
```
Frontend will be at: **http://localhost:5173**

### Quick Start (Using Convenience Scripts)

Alternatively, use the provided batch files:

**Terminal 1: Backend**
```cmd
run_backend.cmd
```

**Terminal 2: Frontend**
```cmd
run_frontend.cmd
```

**Note:** You still need Ollama running separately (`ollama serve`)

Visit: **http://localhost:5173**

## Usage

### Workflow

1. **Enter a disease** (e.g., "Type 2 diabetes")
2. **Click "Find Pathways"** - The researcher agent proposes 3 treatment approaches
3. **Select a pathway** and click "Select & Analyze"
4. **Watch the progress bar** as the system:
   - Deep-analyzes the pathway using Ollama `llama3.1`
   - Suggests drug candidates
   - Runs retrosynthesis analysis using ChemLLM
   - Assesses manufacturability using ChemLLM
5. **View results** with detailed reports for each candidate compound

### Architecture

- **Researcher Agent**: Uses Ollama `llama3.1` for disease research and web scraping
- **Retrosynthesis Agent**: Uses ChemLLM (AI4Chem/ChemLLM-7B-Chat-1.5-DPO) for lab-scale synthesis planning
- **Manufacturability Agent**: Uses ChemLLM for commercial-scale production assessment

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/researcher/pathways` - Generate 3 treatment pathways for a disease
- `POST /api/researcher/deep_analyze/start` - Start deep analysis (returns task_id and ETA)
- `GET /api/researcher/deep_analyze/status/{task_id}` - Poll analysis status
- `POST /api/synthesis/analyze` - Analyze compound synthesis
- `POST /api/integrated/research-and-manufacture` - Full workflow (legacy)

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

### Common Issues

**Ollama not running**
- Start Ollama: `ollama serve`
- Ensure `llama3.1` model is installed: `ollama pull llama3.1`
- Error message will appear in the UI if researcher agent can't connect

**Backend won't start**
- Check Python virtual environment is activated
- Verify all dependencies installed: `pip install -r backend/requirements.txt`
- Check port 8000 is not in use

**Frontend won't start**
- Install Node.js dependencies: `cd frontend && npm install`
- Check port 5173 is not in use
- Clear Vite cache: `npm run dev -- --force`

**ChemLLM not loading**
- Requires GPU (RTX 3050 or better recommended)
- First load downloads ~14GB model (8 checkpoint shards)
- Model will be cached at `~/.cache/huggingface/`
- Missing dependencies: `pip install transformers accelerate torch sentencepiece einops`

**Out of memory errors**
- ChemLLM requires ~8GB+ VRAM
- Close other GPU applications
- Consider using CPU-only mode (slower) by modifying `retrosynthesis.py`

**Slow analysis**
- First model load takes 30-60 seconds
- Subsequent generations are faster (~10-30 seconds)
- Background jobs run asynchronously to avoid blocking UI

### Testing

Run integration tests:
```bash
python test_integration_simple.py
```

This validates:
- All agent imports
- LLM model loading
- Frontend structure
- API endpoint registration

See SETUP_WINDOWS.md for detailed troubleshooting.



For AIDAN ONLY:

cd 'c:\Users\aidan\OneDrive - Duke University\ContAInnum COED\DIC03-ContAInnum\backend' ; python app.py

cd 'c:\Users\aidan\OneDrive - Duke University\ContAInnum COED\DIC03-ContAInnum\frontend' ; npm run dev
