from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Drug Dev Agents API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}



class ResearchDiseaseRequest(BaseModel):
    disease: str

class ResearchDiseaseResponse(BaseModel):
    query: str
    summary: str
    sources: List[str]

from agents.researcher import ResearcherAgent
from agents.retrosynthesis import RetrosynthesisEngine
from agents.manufacturer import ManufacturabilityAgent
import threading
import time
import uuid

# In-memory task store for background deep-analysis jobs
_TASKS = {}

def _estimate_eta_seconds(pathway_text: str) -> int:
    # Realistic estimate for CPU-based ChemLLM (7B params)
    # Each LLM call takes ~30-60s on CPU
    # 1 researcher call + 2-3 candidates * 2 calls each = ~5-7 calls
    base = 180  # 3 minutes minimum
    extra = max(0, min(300, len(pathway_text) // 5))  # Add time for longer pathways
    return base + extra

def _run_deep_analysis_task(task_id: str, disease: str, pathway_text: str):
    try:
        _TASKS[task_id]['status'] = 'running'
        _TASKS[task_id]['started_at'] = time.time()

        researcher = ResearcherAgent()
        analysis = researcher.deep_analyze_pathway(disease, pathway_text)

        # If candidates provided, run retrosynthesis + manufacturability for each
        candidates = analysis.get('candidates', []) or []
        retro = RetrosynthesisEngine()
        mfg = ManufacturabilityAgent()
        compound_reports = []
        for cand in candidates:
            name = cand.get('name') or cand.get('label') or 'unnamed'
            smiles = cand.get('smiles')
            retro_res = retro.retrosynthesis_analysis(name, smiles=smiles)
            mfg_res = mfg.assess_scalability(name, smiles=smiles, researcher_context=analysis.get('deep_analysis', ''))
            compound_reports.append({
                'name': name,
                'smiles': smiles,
                'retrosynthesis': retro_res,
                'manufacturability': mfg_res
            })

        _TASKS[task_id]['status'] = 'done'
        _TASKS[task_id]['finished_at'] = time.time()
        _TASKS[task_id]['result'] = {
            'disease': disease,
            'deep_analysis': analysis.get('deep_analysis', ''),
            'candidates': compound_reports
        }
    except Exception as e:
        _TASKS[task_id]['status'] = 'failed'
        _TASKS[task_id]['error'] = str(e)
        _TASKS[task_id]['finished_at'] = time.time()

@app.post("/api/researcher/research", response_model=ResearchDiseaseResponse)
def researcher_research(req: ResearchDiseaseRequest):
    agent = ResearcherAgent()
    result = agent.research_disease(req.disease)
    return ResearchDiseaseResponse(**result)


class PathwaysRequest(BaseModel):
    disease: str


@app.post("/api/researcher/pathways")
def researcher_pathways(req: PathwaysRequest):
    agent = ResearcherAgent()
    pathways = agent.generate_pathways(req.disease)
    return {"disease": req.disease, "pathways": pathways}


class DeepAnalyzeRequest(BaseModel):
    disease: str
    pathway_text: str


@app.post("/api/researcher/deep_analyze")
def researcher_deep_analyze(req: DeepAnalyzeRequest):
    # For backwards compatibility, run synchronously (may take long).
    researcher = ResearcherAgent()
    analysis = researcher.deep_analyze_pathway(req.disease, req.pathway_text)
    candidates = analysis.get('candidates', []) or []
    retro = RetrosynthesisEngine()
    mfg = ManufacturabilityAgent()
    compound_reports = []
    for cand in candidates:
        name = cand.get('name') or cand.get('label') or 'unnamed'
        smiles = cand.get('smiles')
        retro_res = retro.retrosynthesis_analysis(name, smiles=smiles)
        mfg_res = mfg.assess_scalability(name, smiles=smiles, researcher_context=analysis.get('deep_analysis', ''))
        compound_reports.append({
            'name': name,
            'smiles': smiles,
            'retrosynthesis': retro_res,
            'manufacturability': mfg_res
        })

    return {
        'disease': req.disease,
        'deep_analysis': analysis.get('deep_analysis', ''),
        'candidates': compound_reports
    }


class DeepAnalyzeStartRequest(BaseModel):
    disease: str
    pathway_text: str


@app.post('/api/researcher/deep_analyze/start')
def researcher_deep_analyze_start(req: DeepAnalyzeStartRequest):
    """Start a background deep analysis job and return a task id and ETA."""
    eta = _estimate_eta_seconds(req.pathway_text)
    task_id = str(uuid.uuid4())
    _TASKS[task_id] = {
        'status': 'pending',
        'eta_seconds': eta,
        'created_at': time.time(),
        'started_at': None,
        'finished_at': None,
        'result': None,
        'error': None
    }

    # Launch background thread to perform heavy work
    thread = threading.Thread(target=_run_deep_analysis_task, args=(task_id, req.disease, req.pathway_text), daemon=True)
    thread.start()

    return {'task_id': task_id, 'eta_seconds': eta, 'status': 'started'}


@app.get('/api/researcher/deep_analyze/status/{task_id}')
def researcher_deep_analyze_status(task_id: str):
    task = _TASKS.get(task_id)
    if not task:
        return {'status': 'not_found'}
    # calculate remaining ETA (simple)
    if task['status'] == 'pending':
        remaining = task['eta_seconds']
    elif task['status'] == 'running' and task['started_at']:
        elapsed = time.time() - task['started_at']
        remaining = max(0, task['eta_seconds'] - int(elapsed))
    else:
        remaining = 0

    return {
        'status': task['status'],
        'eta_seconds': task['eta_seconds'],
        'eta_remaining': remaining,
        'result': task.get('result'),
        'error': task.get('error')
    }


class CompoundAnalysisRequest(BaseModel):
    compound_name: str
    smiles: Optional[str] = None
    researcher_context: Optional[str] = None

class CompoundAnalysisResponse(BaseModel):
    compound_name: str
    molecule_type: str
    smiles: Optional[str]
    synthesis_analysis: Dict[str, Any]
    manufacturability: Dict[str, Any]
    integrated_summary: str

@app.post("/api/synthesis/analyze", response_model=CompoundAnalysisResponse)
def analyze_compound(req: CompoundAnalysisRequest):
    """Analyze a single compound for synthesis routes and manufacturability."""
    retro = RetrosynthesisEngine()
    mfg = ManufacturabilityAgent()
    
    retro_result = retro.retrosynthesis_analysis(req.compound_name, smiles=req.smiles)
    mfg_result = mfg.assess_scalability(
        req.compound_name, 
        smiles=req.smiles,
        researcher_context=req.researcher_context
    )
    
    return CompoundAnalysisResponse(
        compound_name=req.compound_name,
        molecule_type="small_molecule",
        smiles=req.smiles,
        synthesis_analysis=retro_result,
        manufacturability=mfg_result,
        integrated_summary=f"Synthesis complexity: {retro_result.get('molecular_properties', {}).get('complexity_score', 'N/A')}/100. Manufacturability score: {mfg_result.get('scalability_score', 'N/A')}/100"
    )


class IntegratedResearchRequest(BaseModel):
    disease: str

class IntegratedResearchResponse(BaseModel):
    disease: str
    research_summary: str
    research_sources: List[str]
    compound_analyses: List[Dict[str, Any]]

@app.post("/api/integrated/research-and-manufacture", response_model=IntegratedResearchResponse)
def integrated_research_manufacture(req: IntegratedResearchRequest):
    """
    Integrated endpoint: Research disease treatments and analyze manufacturability.
    Combines ResearcherAgent, RetrosynthesisEngine, and ManufacturabilityAgent.
    """
    # Step 1: Research the disease
    researcher = ResearcherAgent()
    research_results = researcher.research_disease(req.disease)
    
    # Step 2: Generate pathways and analyze candidates
    pathways = researcher.generate_pathways(req.disease)
    
    # For simplicity, analyze first pathway
    if pathways and len(pathways) > 0:
        first_pathway = pathways[0].get('summary', '')
        deep_analysis = researcher.deep_analyze_pathway(req.disease, first_pathway)
        candidates = deep_analysis.get('candidates', [])
        
        retro = RetrosynthesisEngine()
        mfg = ManufacturabilityAgent()
        compound_analyses = []
        
        for cand in candidates[:3]:  # Limit to 3 for performance
            name = cand.get('name', 'Unknown')
            smiles = cand.get('smiles', '')
            
            retro_result = retro.retrosynthesis_analysis(name, smiles=smiles)
            mfg_result = mfg.assess_scalability(name, smiles=smiles)
            
            compound_analyses.append({
                'name': name,
                'smiles': smiles,
                'retrosynthesis': retro_result,
                'manufacturability': mfg_result
            })
    else:
        compound_analyses = []
    
    return IntegratedResearchResponse(
        disease=req.disease,
        research_summary=research_results["summary"],
        research_sources=research_results["sources"],
        compound_analyses=compound_analyses
    )


# Retrosynthesis-specific endpoints
class RetrosynthesisRequest(BaseModel):
    compound_name: str
    smiles: Optional[str] = None
    disease_context: Optional[str] = None

@app.post("/api/retrosynthesis/analyze")
def retrosynthesis_analyze(req: RetrosynthesisRequest):
    """Analyze retrosynthetic routes for a compound."""
    retro = RetrosynthesisEngine()
    result = retro.retrosynthesis_analysis(req.compound_name, smiles=req.smiles)
    return {
        "compound_name": req.compound_name,
        "smiles": req.smiles,
        "analysis": result
    }


# Manufacturing-specific endpoints
class ManufacturingRequest(BaseModel):
    compound_name: str
    smiles: Optional[str] = None
    compound_type: str = "small_molecule"
    synthesis_complexity: str = "moderate"
    disease_context: Optional[str] = None

@app.post("/api/manufacturing/analyze")
def manufacturing_analyze(req: ManufacturingRequest):
    """Analyze manufacturability and scalability for a compound."""
    mfg = ManufacturabilityAgent()
    result = mfg.assess_scalability(
        compound_name=req.compound_name,
        compound_type=req.compound_type,
        synthesis_complexity=req.synthesis_complexity,
        smiles=req.smiles,
        researcher_context=req.disease_context
    )
    return {
        "compound_name": req.compound_name,
        "smiles": req.smiles,
        "analysis": result
    }


# Research chatbot endpoint
class ChatRequest(BaseModel):
    message: str
    disease_context: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

@app.post("/api/research/chat")
def research_chat(req: ChatRequest):
    """Research chatbot for answering questions about diseases and treatments."""
    researcher = ResearcherAgent()
    
    # Build context from conversation history
    context = ""
    if req.conversation_history:
        context = "\n".join([
            f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
            for msg in req.conversation_history[-6:]  # Last 6 messages for context
        ])
    
    # Build the enhanced prompt with disease context and conversation history
    prompt = f"""You are a pharmaceutical research assistant specializing in drug discovery and development.

Disease Context: {req.disease_context or 'General pharmaceutical research'}

Previous Conversation:
{context if context else 'No previous conversation'}

Current Question: {req.message}

Provide a detailed, scientifically accurate response. Include:
- Relevant research findings
- Potential therapeutic approaches
- Key molecular targets or pathways
- Safety considerations where applicable

If the question is about finding research or literature, perform a web search using your knowledge.
"""
    
    # Check if this is a research/literature query
    research_keywords = ['find', 'research', 'papers', 'studies', 'literature', 'pathway', 'treatment', 'therapy']
    is_research_query = any(keyword in req.message.lower() for keyword in research_keywords)
    
    if is_research_query:
        # Use research_disease for literature queries
        search_query = f"{req.message} {req.disease_context or ''}"
        result = researcher.research_disease(search_query)
        return {
            "response": result.get("summary", "I couldn't find relevant information."),
            "sources": result.get("sources", [])
        }
    else:
        # Use LLM for direct questions
        try:
            response = researcher._local_llm(prompt, max_tokens=1024)
            return {
                "response": response,
                "sources": []
            }
        except Exception as e:
            return {
                "response": f"I'm having trouble connecting to the AI model. Please make sure Ollama is running with llama3.1. Error: {str(e)}",
                "sources": []
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
