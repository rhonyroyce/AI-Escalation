"""
AI Engine module - Handles Ollama-based embeddings and text generation.
"""

import re
import requests
import logging
import numpy as np
import pandas as pd
from typing import List
import tkinter.messagebox as messagebox

from escalation_ai.core.config import OLLAMA_BASE_URL, EMBED_MODEL, GEN_MODEL

logger = logging.getLogger(__name__)


class OllamaBrain:
    """Handles both Embedding (Left Brain) and Generation (Right Brain)"""
   
    def __init__(self):
        self.embed_model = EMBED_MODEL
        self.gen_model = GEN_MODEL
        self._embed_dim = None

    def get_embedding(self, text):
        """Get vector for a single string"""
        if pd.isna(text) or text == "":
            return np.zeros(self.get_dim())
        try:
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": self.embed_model, "input": str(text)},
                timeout=30
            )
            if res.status_code == 200:
                vec = res.json().get('embedding') or res.json().get('embeddings', [[]])[0]
                return np.array(vec)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        return np.zeros(self.get_dim())

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get vectors for multiple strings in one API call"""
        if not texts:
            return []
        
        # Filter out empty texts, keep track of indices
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if pd.isna(text) or text == "":
                continue
            valid_indices.append(i)
            valid_texts.append(str(text))
        
        # Initialize result with zero vectors
        result = [np.zeros(self.get_dim()) for _ in texts]
        
        if not valid_texts:
            return result
        
        try:
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": self.embed_model, "input": valid_texts},
                timeout=120
            )
            if res.status_code == 200:
                embeddings = res.json().get('embeddings', [])
                for idx, vec in zip(valid_indices, embeddings):
                    result[idx] = np.array(vec)
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
        
        return result

    def get_dim(self):
        """Get embedding dimension"""
        if self._embed_dim:
            return self._embed_dim
        # Test call to get dimension
        v = self.get_embedding("test")
        self._embed_dim = len(v)
        return self._embed_dim

    def _strip_thinking_tags(self, text):
        """Remove <think>...</think> blocks from LLM output"""
        if not text:
            return text
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def generate_synthesis(self, context_text):
        """Use the LLM to write a comprehensive executive summary"""
        prompt = f"""You are a Principal Consultant at McKinsey & Company, specializing in Telecom Operations Risk Management. You have been engaged to analyze escalation data for a major telecommunications network deployment project.

ROLE & EXPERTISE:
- You are an expert in identifying systemic operational failures
- You understand telecom infrastructure (RAN, transmission, fiber, site access)
- You focus on strategic risks that threaten project timelines, costs, and reputation
- You communicate with C-suite executives who need actionable insights

ANALYSIS FRAMEWORK:
1. PATTERN RECOGNITION: Identify recurring failure modes and root causes
2. RISK STRATIFICATION: Distinguish between isolated incidents vs systemic issues
3. ORGANIZATIONAL LEARNING: Assess whether past lessons are being applied
4. BUSINESS IMPACT: Translate technical issues to business consequences

DATA CONTEXT (Analyzed Escalation Report):
{context_text}

YOUR TASK:
Produce a comprehensive Executive Risk Assessment with the following sections:

1. CRITICAL ALERT (1-2 sentences): The single most urgent finding that requires immediate executive attention.

2. KEY FINDINGS (3-4 bullet points): Major patterns, systemic issues, or concerning trends discovered in the data.

3. ROOT CAUSE HYPOTHESIS: Based on the data patterns, what underlying organizational or process issues might be driving these escalations?

4. RECOMMENDED ACTIONS (2-3 specific actions): Concrete steps leadership should take this week to address the risks.

5. RISK OUTLOOK: One sentence on what will happen if these issues are not addressed.

FORMATTING RULES:
- Write in clear, professional business English
- Be specific - reference actual categories, percentages, and issue types from the data
- Be direct and urgent where warranted
- Do NOT use markdown formatting (no #, *, or bullet symbols)
- Use plain text with clear section headers
- Total length: 250-400 words
"""
        try:
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.gen_model, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {"num_predict": 800}
                },
                timeout=120
            )
            if res.status_code == 200:
                raw_response = res.json()['response'].strip()
                cleaned = self._strip_thinking_tags(raw_response)
                return cleaned if cleaned else self._generate_fallback_summary(context_text)
        except Exception as e:
            logger.error(f"AI Synthesis Failed: {e}")
            return self._generate_fallback_summary(context_text)
        return self._generate_fallback_summary(context_text)

    def _generate_fallback_summary(self, context_text):
        """Generate a basic summary when AI is unavailable"""
        logger.warning("Using fallback summary generation (AI unavailable)")
        
        total_match = re.search(r'Total Weighted Friction Score: ([\d,]+)', context_text)
        external_match = re.search(r'External.*?: ([\d.]+)%', context_text)
        repeat_match = re.search(r'Confirmed Repeat Offenses: (\d+)', context_text)
        
        total = total_match.group(1) if total_match else "N/A"
        external = external_match.group(1) if external_match else "N/A"
        repeats = repeat_match.group(1) if repeat_match else "0"
        
        return (
            f"[AUTO-GENERATED SUMMARY - AI Unavailable]\n\n"
            f"This report analyzed escalation data with a total weighted friction score of {total}. "
            f"External-facing issues account for {external}% of total risk. "
            f"There are {repeats} confirmed repeat offenses indicating potential organizational learning gaps. "
            f"Please review the detailed data and charts below for full analysis."
        )

    def unload(self):
        """Unload both models to free VRAM"""
        for m in [self.embed_model, self.gen_model]:
            try:
                requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": m, "keep_alive": 0})
            except Exception as e:
                logger.warning(f"Failed to unload model {m}: {e}")


def check_models(ai):
    """Quick probe to ensure required AI models exist and are working"""
    logger.info("Checking AI model availability...")
    
    # Check embedding model
    try:
        test_vec = ai.get_embedding("test connection")
        if len(test_vec) == 0 or np.all(test_vec == 0):
            raise ValueError("Embedding returned zero vector")
        logger.info(f"✓ Embedding model '{EMBED_MODEL}' is active (dim={len(test_vec)})")
    except Exception as e:
        logger.error(f"Embedding model check failed: {e}")
        messagebox.showerror(
            "Model Not Found", 
            f"Embedding model '{EMBED_MODEL}' is not available.\n\n"
            f"Please run:\n  ollama pull {EMBED_MODEL}\n\n"
            f"Error: {e}"
        )
        return False
    
    # Check generation model (optional - just warn, don't block)
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": "test", "stream": False},
            timeout=10
        )
        if res.status_code == 200:
            logger.info(f"✓ Generation model '{GEN_MODEL}' is active")
        else:
            logger.warning(f"Generation model '{GEN_MODEL}' may not be available (status: {res.status_code})")
    except Exception as e:
        logger.warning(f"Generation model '{GEN_MODEL}' check failed: {e}. Executive summary may not work.")
    
    return True


def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        requests.get(f"{OLLAMA_BASE_URL}/", timeout=3)
        return True
    except Exception as e:
        logger.error(f"Ollama server not reachable: {e}")
        messagebox.showerror(
            "Ollama Not Running",
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}\n\n"
            "Please ensure Ollama is running:\n  ollama serve"
        )
        return False
