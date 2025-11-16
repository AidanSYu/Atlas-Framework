import requests
from typing import List
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import json
from typing import Dict

class ResearcherAgent:
    def _local_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        """Query the local Ollama server for research prompts using `llama3.2`.

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
        try:
            r = requests.get(url, timeout=6)
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs)
            return text[:8000]
        except Exception:
            return ""

    def _search_web(self, query: str, num: int = 5):
        try:
            results = list(DDGS().text(query, max_results=num))
            clean = []
            for r in results:
                url = r.get("href") or r.get("url") or r.get("link")
                if url:
                    clean.append({"url": url, "title": r.get("title")})
            return clean
        except Exception:
            return []

    def research_disease(self, disease: str):
        query = f"latest treatment research for {disease}"
        search_results = self._search_web(query, num=5)
        extracted = []
        for r in search_results:
            url = r.get("url")
            if not url:
                continue
            content = self._scrape_url(url)
            extracted.append({"url": url, "content": content[:2000]})

        combined_text = "\n\n".join(e["content"] for e in extracted)
        summary_prompt = f"""
You are a scientific research assistant. Provide detailed synthetic plans based on previous research. There should be 2-3 plans. Give lab protocals.

Summarize the most important findings below regarding: {disease}

TEXT:
{combined_text}

Return for each synthetic plan:
- A concise research summary (3-4 sentences)
- 3–5 key findings
- 3 citations optimaly DOI linking back to the sources
"""
        summary = self._local_llm(summary_prompt)
        return {
            "query": query,
            "summary": summary,
            "sources": [r.get("url") for r in search_results if r.get("url")],
        }

    def generate_pathways(self, disease: str):
        """Generate three brief therapeutic pathways for the given disease.

        Returns a list of dicts: {id, title, short_description}
        """
        prompt = f"""
You are a concise biomedical researcher. For the disease: {disease}
Provide exactly THREE brief therapeutic pathways or strategies (not full protocols).
For each pathway return a JSON object with keys: "id" (1..3), "title" (short), "summary" (1-2 sentences).
Return ONLY a valid JSON array with no markdown formatting, no code blocks, no explanation.
Example format: [{{"id": 1, "title": "...", "summary": "..."}}, {{"id": 2, "title": "...", "summary": "..."}}, {{"id": 3, "title": "...", "summary": "..."}}]
"""
        resp = self._local_llm(prompt, max_tokens=1024)
        
        # Clean response - remove markdown code blocks if present
        resp = resp.strip()
        if resp.startswith('```'):
            # Remove markdown code block markers
            lines = resp.split('\n')
            # Remove first line (```json or ```)
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            resp = '\n'.join(lines).strip()
        
        # Try to parse JSON; if parsing fails, attempt to extract lines
        try:
            data = json.loads(resp)
            if isinstance(data, list):
                return data
        except Exception as e:
            # Fallback: split by lines and construct simple objects
            lines = [l.strip() for l in resp.splitlines() if l.strip() and not l.startswith('{') and not l.startswith('[')]
            pathways = []
            for i, line in enumerate(lines[:3]):
                if line and len(line) > 10:  # Skip empty or very short lines
                    pathways.append({"id": i+1, "title": f"Pathway {i+1}", "summary": line})
            # If no valid pathways, create defaults
            if len(pathways) == 0:
                pathways = [
                    {"id": 1, "title": "Pathway 1", "summary": f"Therapeutic approach for {disease} (parsing failed)"},
                    {"id": 2, "title": "Pathway 2", "summary": f"Alternative treatment strategy for {disease}"},
                    {"id": 3, "title": "Pathway 3", "summary": f"Experimental therapy for {disease}"}
                ]
            return pathways

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
