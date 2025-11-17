import requests
from typing import List
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import json
from typing import Dict
import re

class ResearcherAgent:
    def _local_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        """Query the local Ollama server for research prompts using `llama3.2:1b`.

        Research tasks often require broad, web-focused models. This method
        uses a local Ollama HTTP server (model `llama3.2:1b`) to handle those
        prompts. If Ollama is not running, it raises a clear RuntimeError
        explaining how to start and pull the model.
        """

        try:
            url = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "num_predict": max_tokens,
            }
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "ConnectTimeout" in error_msg or "Max retries exceeded" in error_msg:
                raise RuntimeError(
                    "Ollama is not running or unreachable. To enable research LLMs:\n"
                    "1. Install Ollama: https://ollama.ai (or use your platform's installer)\n"
                    "2. Pull the model: `ollama pull llama3.2:1b`\n"
                    "3. Start the server: `ollama serve`"
                )
            elif "Read timed out" in error_msg:
                raise RuntimeError(
                    "Ollama request timed out. The model may be taking too long; try a smaller prompt or ensure the model is downloaded."
                )
            else:
                raise RuntimeError(f"Ollama HTTP error: {error_msg}")

    def _scrape_url(self, url: str) -> str:
        """Scrape text content from a URL and extract DOI if present."""
        try:
            r = requests.get(url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs)
            return text[:8000]
        except Exception:
            return ""

    def _extract_doi(self, text: str) -> List[str]:
        """Extract DOI references from text."""
        doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        return list(set(dois))[:5]  # Return up to 5 unique DOIs

    def _search_web(self, query: str, num: int = 5):
        """Search the web using DuckDuckGo."""
        try:
            results = list(DDGS().text(query, max_results=num))
            clean = []
            for r in results:
                url = r.get("href") or r.get("url") or r.get("link")
                if url:
                    clean.append({
                        "url": url, 
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "")
                    })
            return clean
        except Exception:
            return []

    def professional_chat(self, question: str, context: str = "") -> Dict[str, any]:
        """Professional research assistant chatbot with technical terminology.
        
        Responds to research questions with expert-level scientific insights.
        Can perform web searches for literature when relevant and include references.
        """
        # Check if this question would benefit from literature search
        search_keywords = ['research', 'studies', 'papers', 'literature', 'findings', 'evidence', 'clinical', 'trials', 'mechanism', 'pathway']
        should_search = any(keyword in question.lower() for keyword in search_keywords)
        
        references = []
        context_info = ""
        
        if should_search:
            # Perform quick literature search for context
            search_query = f"{question} {context if context else ''}"
            search_results = self._search_web(search_query, num=5)
            
            for idx, result in enumerate(search_results):
                url = result.get("url", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                if url:
                    references.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet[:200]
                    })
                    
                    # Try to scrape for DOI
                    content = self._scrape_url(url)
                    dois = self._extract_doi(content) if content else []
                    if dois:
                        references[-1]["doi"] = dois[0]
            
            # Build context from search results
            if references:
                context_info = "\n\nRecent Literature Context:\n" + "\n".join([
                    f"- {ref['title']}: {ref.get('snippet', '')}"
                    for ref in references[:3]
                ])
        
        prompt = f"""You are a pharmaceutical research assistant helping with chemistry, drug discovery, and pharmacology questions.

Context: {context if context else "General pharmaceutical research"}

Question: {question}
{context_info}

Answer the question directly and professionally. For chemistry questions, use technical terminology and chemical nomenclature (IUPAC, SMILES, SMARTS) when appropriate. For simple questions, be brief. For complex topics, provide detailed explanations."""

        response = self._local_llm(prompt, max_tokens=1500)
        
        return {
            "response": response,
            "references": references[:5] if references else []
        }

    def search_literature(self, research_question: str) -> Dict[str, any]:
        """Search for real scientific papers based on a research question.
        
        Returns papers with titles, URLs, snippets, and extracted DOIs.
        """
        try:
            print(f"[LITERATURE SEARCH] Query: {research_question}")
            
            # Enhance query for scientific papers
            enhanced_query = f"{research_question} site:pubmed.ncbi.nlm.nih.gov OR site:scholar.google.com OR site:sciencedirect.com OR site:nature.com OR site:science.org"
            
            print(f"[LITERATURE SEARCH] Enhanced query: {enhanced_query}")
            search_results = self._search_web(enhanced_query, num=10)
            print(f"[LITERATURE SEARCH] Found {len(search_results)} raw results")
            
            papers = []
            all_dois = []
            
            for idx, result in enumerate(search_results):
                url = result.get("url", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                print(f"[LITERATURE SEARCH] Processing paper {idx + 1}: {title[:50]}...")
                
                # Try to scrape full content for DOI extraction
                content = self._scrape_url(url)
                dois = self._extract_doi(content) if content else self._extract_doi(snippet)
                
                all_dois.extend(dois)
                
                # Create a concise description from the snippet (first sentence or up to 150 chars)
                description = snippet[:150] if snippet else "No description available"
                if len(snippet) > 150:
                    # Try to cut at sentence boundary
                    period_idx = description.rfind('. ')
                    if period_idx > 50:
                        description = description[:period_idx + 1]
                    else:
                        description = description + "..."
                
                papers.append({
                    "id": idx + 1,
                    "title": title,
                    "url": url,
                    "description": description,
                    "snippet": snippet[:500],
                    "doi": dois[0] if dois else None
                })
            
            print(f"[LITERATURE SEARCH] Returning {len(papers)} papers")
            
            return {
                "query": research_question,
                "papers": papers,
                "total_found": len(papers),
                "dois_extracted": list(set(all_dois))[:10]
            }
        except Exception as e:
            print(f"[LITERATURE SEARCH ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "query": research_question,
                "papers": [],
                "total_found": 0,
                "dois_extracted": [],
                "error": str(e)
            }

    def generate_disease_pathways(self, disease: str) -> Dict[str, any]:
        """Generate therapeutic pathways for a specific disease.
        
        Returns detailed pathways with molecular targets, mechanisms, and clinical relevance.
        """
        prompt = f"""You are a translational medicine expert specializing in {disease}. 

Generate 3-4 distinct therapeutic pathways for treating {disease}. For each pathway, provide:

1. Pathway name (concise, mechanism-based)
2. Molecular target(s) (specific proteins, receptors, enzymes)
3. Mechanism of action (2-3 sentences, technical detail)
4. Development stage (preclinical/Phase I/II/III/approved)
5. Key compounds or drug classes (if applicable)

Format as JSON array:
[
  {{
    "id": 1,
    "pathway_name": "...",
    "molecular_targets": ["target1", "target2"],
    "mechanism": "detailed mechanism...",
    "stage": "...",
    "compounds": ["compound1", "compound2"]
  }}
]

Use precise pharmacological and biochemical terminology. Focus on scientifically validated approaches.
Return ONLY valid JSON, no markdown formatting."""

        resp = self._local_llm(prompt, max_tokens=2048)
        
        # Clean response - remove markdown code blocks if present
        resp = resp.strip()
        if resp.startswith('```'):
            lines = resp.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            resp = '\n'.join(lines).strip()
        
        # Try to parse JSON
        try:
            data = json.loads(resp)
            if isinstance(data, list):
                return {
                    "disease": disease,
                    "pathways": data,
                    "total_pathways": len(data)
                }
        except Exception as e:
            # Fallback: create structured default pathways
            return {
                "disease": disease,
                "pathways": [
                    {
                        "id": 1,
                        "pathway_name": f"Targeted Therapy for {disease}",
                        "molecular_targets": ["Unknown"],
                        "mechanism": f"Therapeutic intervention targeting specific molecular pathways in {disease}",
                        "stage": "Research",
                        "compounds": []
                    }
                ],
                "total_pathways": 1,
                "error": "Failed to parse LLM response"
            }

    def deep_analyze_pathway(self, disease: str, pathway_text: str) -> Dict[str, object]:
        """Produce a deep analysis of the selected pathway and propose candidate compounds.
        
        Uses web scraping to find real research papers and DOI references.

        Returns a dict with keys: deep_analysis (str), candidates (list of {name,smiles,rationale}).
        """
        # Step 1: Search for relevant research papers
        search_query = f"{disease} {pathway_text} drug candidates clinical trials"
        search_results = self._search_web(search_query, num=8)
        
        # Step 2: Scrape content from the search results
        extracted = []
        for r in search_results:
            url = r.get("url")
            if not url:
                continue
            content = self._scrape_url(url)
            extracted.append({"url": url, "content": content[:3000]})
        
        # Step 3: Combine scraped content
        combined_text = "\n\n".join(f"Source: {e['url']}\n{e['content']}" for e in extracted)
        
        # Step 4: Create prompt with real research data
        prompt = f"""
You are a senior translational researcher specializing in {disease}. 

Selected therapeutic pathway:
{pathway_text}

Based on the following research articles and clinical data:

{combined_text[:12000]}

Provide a comprehensive deep analysis covering:

1. **MECHANISM OF ACTION**: How does this pathway work at the molecular level?
2. **MOLECULAR TARGETS**: What proteins, receptors, or pathways are targeted?
3. **FEASIBILITY**: Clinical and scientific rationale based on the research
4. **CHEMICAL SCAFFOLDS**: What molecular structures would be effective?
5. **CANDIDATE MOLECULES**: Propose 2-3 specific small-molecule drug candidates based on the literature

For each candidate, provide:
- name: Clear compound name (use actual compounds from the research if available)
- smiles: SMILES notation if known (or empty string)
- rationale: Why this molecule targets the pathway (2-3 sentences, cite evidence)

IMPORTANT: Include DOI references where available. Extract DOIs from the research text (format: 10.xxxx/xxxxx)

Return ONLY valid JSON with this structure:
{{
  "deep_analysis": "detailed analysis text here (at least 300 words). Include citations and DOI references in format: (DOI: 10.xxxx/xxxxx)",
  "candidates": [
    {{"name": "Compound Name", "smiles": "SMILES or empty", "rationale": "explanation with DOI if available"}},
    {{"name": "Compound Name 2", "smiles": "SMILES or empty", "rationale": "explanation with DOI if available"}}
  ]
}}

Do not include any text outside the JSON structure. No markdown code blocks.
"""
        resp = self._local_llm(prompt, max_tokens=2048)
        
        # Clean response - remove markdown code blocks if present
        resp = resp.strip()
        if resp.startswith('```'):
            lines = resp.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            resp = '\n'.join(lines).strip()
        
        try:
            data = json.loads(resp)
            return data
        except Exception as e:
            # If parsing fails, return raw text fallback
            return {"deep_analysis": resp, "candidates": []}
