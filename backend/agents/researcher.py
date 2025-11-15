import requests
from typing import List
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

class ResearcherAgent:
    def _local_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Uses the local Ollama server to get model output.
        Make sure `ollama serve` is running.
        """
        try:
            url = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "ConnectTimeout" in error_msg:
                raise RuntimeError(
                    "Ollama is not running. Please start it:\n"
                    "1. Install: brew install ollama (or https://ollama.ai)\n"
                    "2. Pull model: ollama pull mistral\n"
                    "3. Start server: ollama serve"
                )
            elif "Read timed out" in error_msg:
                raise RuntimeError(
                    "Ollama request timed out. The model may be taking too long.\n"
                    "Try a smaller/faster model: ollama pull llama2:7b"
                )
            else:
                raise RuntimeError(f"Ollama error: {error_msg}")

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
