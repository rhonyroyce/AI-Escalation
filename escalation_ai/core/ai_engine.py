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
        """Use the LLM to write a comprehensive executive summary with financial analysis"""
        prompt = f"""You are a Principal Consultant specializing in Telecom Operations Risk Management and Financial Impact Analysis. Analyze this escalation data for a telecommunications network deployment project.

CRITICAL RULES - READ CAREFULLY:
1. ONLY use numbers, percentages, dollar amounts, and counts that appear in the DATA CONTEXT below
2. The data includes FINANCIAL IMPACT METRICS - USE THEM in your analysis
3. DO NOT fabricate statistics, dates, or specific incidents not in the data
4. Every claim must be traceable to the data provided
5. Reference actual dollar figures from the Financial Impact section

DATA CONTEXT (Analyzed Escalation Report with Financial Metrics):
{context_text}

YOUR TASK:
Produce a comprehensive Executive Risk & Financial Assessment using ONLY the data provided above.

SECTION 1 - CRITICAL ALERT (2-3 sentences):
The single most urgent finding combining operational AND financial impact. Use actual ticket counts, percentages, AND the total financial impact figures from the data. State the direct cost exposure and revenue at risk.

SECTION 2 - KEY FINDINGS (6-8 detailed observations):
For each finding, cite specific numbers from the data:
- Ticket counts and percentages by category
- Financial impact by category (use the dollar figures provided)
- Average cost per escalation and highest-cost categories
- Severity distributions and their cost implications
- High-cost tickets concentration (top 10% analysis)
- Recurrence risk exposure in dollar terms
- Friction score concentrations and cost correlation
Explain both operational AND financial implications.

SECTION 3 - FINANCIAL IMPACT ANALYSIS (2-3 paragraphs):
Deep dive into the cost data:
- Which categories are the biggest cost drivers? (use actual $ figures)
- What is the labor cost vs opportunity cost breakdown?
- How does severity correlate with financial impact?
- What is the total revenue at risk and recurrence exposure?
- What are the highest cost-per-ticket categories (even if low volume)?
Connect financial patterns to operational root causes.

SECTION 4 - ROOT CAUSE HYPOTHESIS (2-3 paragraphs):
Based on the data patterns, what organizational or process issues might be driving these escalations? Consider:
- What do the category distributions and their costs suggest about systemic issues?
- What do repeat offenses indicate about organizational learning and cost waste?
- What do the severity patterns reveal about escalation discipline?
- Why might certain categories have higher cost-per-ticket despite lower volume?
Ground your hypothesis in the actual data patterns.

SECTION 5 - STRATEGIC RECOMMENDATIONS (5-6 specific actions):
Concrete steps leadership should take with financial justification:
- Address the highest-cost categories (cite specific $ savings potential)
- Tackle high cost-per-ticket categories even if low volume
- Reduce recurrence risk exposure
- Improve areas with highest friction scores
- Fix process/documentation gaps
For each recommendation, reference the financial benefit using data from the context.

SECTION 6 - RISK OUTLOOK & COST PROJECTION:
Describe likely trajectory if issues continue:
- Which cost patterns suggest escalating financial exposure?
- What does the recurrence risk indicate about future costs?
- Which categories are trending toward higher financial impact?
- What is the quarterly/annual exposure if current patterns continue? (extrapolate from the data period)

SECTION 7 - EXECUTIVE BOTTOM LINE (3-4 sentences):
Summarize the core message with specific financial stakes. State the total financial impact, revenue at risk, and the cost of inaction. Be direct about severity.

FORMATTING RULES:
- Format section headers in BOLD using **SECTION X - TITLE** format
- Leave a BLANK LINE between each section for readability
- Use dashes (-) for bullet points within sections
- Reference actual numbers AND dollar figures from the data
- Total length: 800-1000 words
- Use $ figures from the Financial Impact section liberally

EXAMPLE FORMAT:
**SECTION 1 - CRITICAL ALERT**
Content here...

**SECTION 2 - KEY FINDINGS**
- Finding 1
- Finding 2"""
        try:
            logger.info(f"  Requesting AI synthesis from {self.gen_model}...")
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.gen_model, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {
                        "num_predict": 5000,  # Increased for financial analysis
                        "temperature": 0.5,   # More factual for financial data
                    }
                },
                timeout=480  # 8 minutes for comprehensive financial analysis
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
