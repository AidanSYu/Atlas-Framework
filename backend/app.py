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

from .agents.researcher import ResearcherAgent
from .agents.synthesis_manufacturer import SynthesisManufacturerAgent

@app.post("/api/researcher/research", response_model=ResearchDiseaseResponse)
def researcher_research(req: ResearchDiseaseRequest):
    agent = ResearcherAgent()
    result = agent.research_disease(req.disease)
    return ResearchDiseaseResponse(**result)


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
    agent = SynthesisManufacturerAgent()
    result = agent.analyze_compound(
        compound_name=req.compound_name,
        smiles=req.smiles,
        researcher_context=req.researcher_context
    )
    return CompoundAnalysisResponse(**result)


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
    Combines ResearcherAgent and SynthesisManufacturerAgent.
    """
    # Step 1: Research the disease
    researcher = ResearcherAgent()
    research_results = researcher.research_disease(req.disease)
    
    # Step 2: Analyze compounds for manufacturability
    manufacturer = SynthesisManufacturerAgent()
    compound_analyses = manufacturer.batch_analyze_from_research(research_results)
    
    return IntegratedResearchResponse(
        disease=req.disease,
        research_summary=research_results["summary"],
        research_sources=research_results["sources"],
        compound_analyses=compound_analyses
    )
