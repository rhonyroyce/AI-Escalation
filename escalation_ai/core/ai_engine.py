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
from escalation_ai.core.gpu_utils import get_optimal_embedding_batch_size

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

    def get_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """Get vectors for multiple strings in batched API calls
        
        Args:
            texts: List of strings to embed
            batch_size: Items per batch (auto-detected from GPU VRAM if None)
        """
        # Auto-detect optimal batch size based on GPU VRAM
        if batch_size is None:
            batch_size = get_optimal_embedding_batch_size()
            logger.info(f"Auto-detected embedding batch size: {batch_size}")
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
        
        # Process in batches to avoid timeout
        all_embeddings = []
        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            
            try:
                res = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": self.embed_model, "input": batch_texts},
                    timeout=600  # 10 minutes per batch for slower GPUs
                )
                if res.status_code == 200:
                    batch_embeddings = res.json().get('embeddings', [])
                    all_embeddings.extend(batch_embeddings)
                else:
                    logger.warning(f"Batch embedding returned status {res.status_code}")
                    all_embeddings.extend([np.zeros(self.get_dim()).tolist() for _ in batch_texts])
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                all_embeddings.extend([np.zeros(self.get_dim()).tolist() for _ in batch_texts])
        
        # Map embeddings back to original indices
        for idx, vec in zip(valid_indices, all_embeddings):
            result[idx] = np.array(vec)
        
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
        """Use the LLM to write a comprehensive executive summary using McKinsey frameworks"""
        prompt = f"""You are a McKinsey Principal Consultant specializing in Telecom Operations. Write an executive-ready analysis using consulting best practices.

CRITICAL RULES:
1. ONLY use numbers, percentages, dollar amounts from the DATA CONTEXT below
2. DO NOT fabricate statistics - every claim must be traceable to the data
3. Apply MECE (Mutually Exclusive, Collectively Exhaustive) structure
4. Use PYRAMID PRINCIPLE: Lead with the answer, then support
5. Every finding needs a "SO WHAT?" implication

DATA CONTEXT:
{context_text}

═══════════════════════════════════════════════════════════════
PRODUCE THIS STRUCTURED OUTPUT:
═══════════════════════════════════════════════════════════════

SECTION 1 - THE BOTTOM LINE (Pyramid Principle - Answer First)
Write 2-3 sentences with THE KEY MESSAGE. State:
- Total financial exposure ($X)
- The #1 problem causing it
- The recommended action
Example format: "Analysis reveals $X in financial exposure driven primarily by [top category]. Immediate focus on [action] can reduce costs by X%."

SECTION 2 - SITUATION OVERVIEW
In 3-4 bullet points, provide context:
- Total tickets analyzed and time period scope
- Total financial impact and revenue at risk
- Critical vs Major vs Minor distribution
- Overall health assessment (Critical/At Risk/Stable)

SECTION 3 - KEY FINDINGS (MECE Structure)
Organize findings into 4 MUTUALLY EXCLUSIVE categories. For each, include the data AND the "SO WHAT":

A) PROCESS GAPS (scheduling, workflow, compliance issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

B) KNOWLEDGE GAPS (documentation, training, expertise issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

C) SYSTEM/TOOL ISSUES (configuration, data, technical issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

D) COMMUNICATION FAILURES (response, handoff, coordination issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

SECTION 4 - 80/20 ANALYSIS (Pareto Principle)
Identify the vital few driving majority of impact:
- Which 2-3 categories drive 80% of financial impact? (use actual $ from data)
- Which 2-3 root causes drive 80% of ticket volume?
- Where should resources be concentrated for maximum ROI?

SECTION 5 - PRIORITIZED RECOMMENDATIONS (Impact-Effort Matrix)
List 4-5 actions categorized by:

QUICK WINS (High Impact, Low Effort) - Do First:
1. [Action] | Impact: $X savings | Timeline: Week 1-2

MAJOR PROJECTS (High Impact, High Effort) - Plan For:
2. [Action] | Impact: $X savings | Timeline: Month 2-3

FILL-INS (Low Impact, Low Effort) - If Time Permits:
3. [Action] | Impact: $X savings | Timeline: Ongoing

For each, cite specific financial benefit from the data.

SECTION 6 - RISK ASSESSMENT (RAG Status)
Provide traffic light status for each area:
- Financial Exposure: [RED/AMBER/GREEN] - [reason with $ figure]
- Recurrence Risk: [RED/AMBER/GREEN] - [reason with %]
- Process Maturity: [RED/AMBER/GREEN] - [reason]
- Resolution Capability: [RED/AMBER/GREEN] - [reason]

SECTION 7 - EXECUTIVE CALLOUT
One powerful statement (2 sentences max) that a CEO should remember:
"[Key insight with financial stake]. [Recommended action with expected outcome]."

FORMATTING:
- Use **SECTION X - TITLE** for headers
- Blank line between sections
- Use - for bullets, numbers for ordered lists
- Include $ figures throughout
- 600-800 words total
- Be direct and actionable, not academic"""
        try:
            logger.info(f"  Requesting AI synthesis from {self.gen_model}...")
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.gen_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 12000,  # Increased for complete executive summary
                        "num_ctx": 32768,      # Larger context window for full analysis
                        "temperature": 0.5,    # More factual for financial data
                    }
                },
                timeout=600  # 10 minutes for comprehensive financial analysis
            )
            if res.status_code == 200:
                raw_response = res.json()['response'].strip()
                cleaned = self._strip_thinking_tags(raw_response)
                if cleaned:
                    logger.info(f"  ✓ AI synthesis complete ({len(cleaned)} chars)")
                    return cleaned
                else:
                    logger.warning(f"  AI returned empty response after cleaning")
                    return self._generate_fallback_summary(context_text)
            else:
                logger.error(f"  AI request failed with status {res.status_code}: {res.text[:200]}")
                return self._generate_fallback_summary(context_text)
        except requests.exceptions.Timeout:
            logger.error(f"AI Synthesis timed out after 480s - model may be loading. Try again or use smaller model.")
            return self._generate_fallback_summary(context_text)
        except Exception as e:
            logger.error(f"AI Synthesis Failed: {e}")
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
