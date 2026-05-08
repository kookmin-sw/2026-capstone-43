"""
LLM Wrapper - Gemini / Qwen / Llama 통합 인터페이스

Usage:
    from llm_wrapper import create_llm
    llm = create_llm("gemini")   # or "qwen", "llama"
    response_text = llm.generate(prompt)
"""
import time
import json
import re


class LLMResponse:
    """Gemini response 객체와 동일한 인터페이스."""
    def __init__(self, text):
        self.text = text


class GeminiLLM:
    """Vertex AI Gemini backend."""
    def __init__(self):
        import vertexai
        from vertexai.generative_models import GenerativeModel
        from config import GCP_PROJECT_ID, GCP_LOCATION
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        self.model = GenerativeModel('gemini-2.5-flash')
        self.name = "gemini-2.5-flash"

    def generate(self, prompt, max_retries=5):
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.model.generate_content(prompt)
            except Exception as e:
                last_err = e
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt + 1
                    print(f"[LLM] Rate limited, retrying in {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise
        raise last_err


class OllamaLLM:
    """Ollama local model backend (Qwen, Llama, etc.).

    Uses /api/chat with system message separation and format:"json"
    to enforce valid JSON output from small models.
    """
    def __init__(self, model_name, ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.name = model_name
        self._check_connection()

    def _check_connection(self):
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                available = [m["name"] for m in data.get("models", [])]
                model_base = self.model_name.split(":")[0]
                found = any(model_base in m for m in available)
                if not found:
                    print(f"[LLM] WARNING: Model '{self.model_name}' not found in Ollama.")
                    print(f"[LLM] Available models: {available}")
                    print(f"[LLM] Run: ollama pull {self.model_name}")
                else:
                    print(f"[LLM] Ollama model '{self.model_name}' ready.")
        except Exception as e:
            print(f"[LLM] WARNING: Cannot connect to Ollama at {self.ollama_url}: {e}")
            print(f"[LLM] Make sure Ollama is running: ollama serve")

    def generate(self, prompt, max_retries=3):
        """Generate using /api/chat with system/user message separation.

        Automatically splits prompt at the '----' delimiter:
        everything before it is the system message, everything after is user input.
        Uses format:"json" to enforce valid JSON from small models.
        """
        import urllib.request

        # Split system vs user parts at the delimiter
        system_msg = ""
        user_msg = prompt
        delimiter_match = re.search(r'\n-{10,}\n', prompt)
        if delimiter_match:
            system_msg = prompt[:delimiter_match.start()].strip()
            user_msg = prompt[delimiter_match.end():].strip()

        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})

        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,
                "num_predict": 4096,
            }
        }).encode("utf-8")

        last_err = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.ollama_url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                    text = data.get("message", {}).get("content", "")
                    return LLMResponse(text)
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + 1
                    print(f"[LLM] Ollama error, retrying in {wait}s... ({attempt+1}/{max_retries}): {e}")
                    time.sleep(wait)
                else:
                    raise
        raise last_err


# Model name mapping
_OLLAMA_MODELS = {
    "qwen": "qwen2.5:7b",
    "llama": "llama3.1:8b",
}


def create_llm(backend="gemini"):
    """Create LLM instance.

    Args:
        backend: "gemini", "qwen", or "llama"
    """
    if backend == "gemini":
        print("[LLM] Using Gemini 2.5 Flash (Vertex AI)")
        return GeminiLLM()
    elif backend in _OLLAMA_MODELS:
        model_name = _OLLAMA_MODELS[backend]
        print(f"[LLM] Using {model_name} (Ollama local)")
        return OllamaLLM(model_name)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Choose from: gemini, qwen, llama")
