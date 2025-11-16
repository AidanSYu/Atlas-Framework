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

from backend.agents.researcher import ResearcherAgent
from backend.agents.retrosynthesis import RetrosynthesisEngine
from backend.agents.manufacturer import ManufacturabilityAgent
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
