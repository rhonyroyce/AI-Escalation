"""
Pipeline Orchestrator - Main execution flow for Escalation AI.

Coordinates all 7 phases of the analysis pipeline:
1. Data Loading & Validation
2. AI Classification
3. Strategic Friction Scoring
4. Recidivism/Learning Analysis
5. Similar Ticket Analysis
6. Resolution Time Prediction
7. Report Generation
"""

import os
import logging
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import pandas as pd
import numpy as np

from ..core.config import (
    OLLAMA_BASE_URL, EMBED_MODEL, GEN_MODEL,
    COL_SUMMARY, COL_CATEGORY, COL_DATETIME, COL_TYPE,
    COL_ORIGIN, COL_SEVERITY
)
from ..core.ai_engine import OllamaBrain
from ..core.utils import clean_text
from ..classification import classify_rows
from ..scoring import calculate_strategic_friction
from ..feedback import FeedbackLearning, PriceCatalog
from ..predictors import (
    apply_recurrence_predictions,
    apply_similar_ticket_analysis,
    apply_resolution_time_prediction
)

logger = logging.getLogger(__name__)

# Global instances
feedback_learner = None
price_catalog = None


def get_feedback_learner():
    """Get or create the global feedback learner instance."""
    global feedback_learner
    if feedback_learner is None:
        feedback_learner = FeedbackLearning()
    return feedback_learner


def get_price_catalog():
    """Get or create the global price catalog instance."""
    global price_catalog
    if price_catalog is None:
        price_catalog = PriceCatalog()
    return price_catalog


def check_ollama_server():
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("[Ollama] Server is running")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"[Ollama] Server check failed: {e}")
    
    messagebox.showerror(
        "Ollama Not Running",
        "Ollama server is not running.\n\n"
        "Please start Ollama with: ollama serve\n"
        f"Expected at: {OLLAMA_BASE_URL}"
    )
    return False


def check_models(ai):
    """Verify that required AI models are available."""
    try:
        # Test embedding model
        test_embed = ai.get_embedding("test")
        if test_embed is None or len(test_embed) == 0:
            messagebox.showerror(
                "Model Error",
                f"Embedding model '{EMBED_MODEL}' is not available.\n\n"
                f"Please install with: ollama pull {EMBED_MODEL}"
            )
            return False
        
        logger.info(f"[Ollama] Embedding model verified: {EMBED_MODEL}")
        return True
        
    except Exception as e:
        logger.error(f"[Ollama] Model check failed: {e}")
        messagebox.showerror("Model Error", f"Model verification failed: {e}")
        return False


def validate_data_quality(df):
    """
    Validate that the dataframe has enough usable data.
    
    Returns: bool indicating if data is valid
    """
    if df is None or len(df) == 0:
        return False
    
    # Check for at least one text column
    text_cols = [COL_SUMMARY, COL_CATEGORY]
    has_text = any(col in df.columns for col in text_cols) or len(df.columns) > 0
    
    if not has_text:
        return False
    
    # Check for minimum rows
    if len(df) < 1:
        return False
    
    return True


def audit_learning(df, ai):
    """
    Enhanced recidivism analysis using embeddings.
    
    Identifies tickets similar to past issues to flag potential repeat failures.
    
    Args:
        df: DataFrame with ticket data
        ai: OllamaBrain instance for embeddings
        
    Returns:
        DataFrame with learning analysis columns added
    """
    logger.info("[Audit Engine] Starting recidivism and learning analysis...")
    
    # Initialize columns
    df['Learning_Status'] = 'New'
    df['Recidivism_Score'] = 0.0
    df['Similar_Historical_Issue'] = ''
    
    # Get embeddings for all tickets
    texts = df['Combined_Text'].tolist()
    embeddings = []
    
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == '':
            embeddings.append(None)
        else:
            emb = ai.get_embedding(str(text))
            embeddings.append(emb)
    
    df['embedding'] = embeddings
    
    # Calculate similarity between all pairs
    valid_embeddings = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    
    for i, emb_i in valid_embeddings:
        max_similarity = 0.0
        most_similar_idx = -1
        
        for j, emb_j in valid_embeddings:
            if i == j:
                continue
            
            # Calculate cosine similarity
            dot = np.dot(emb_i, emb_j)
            norm_i = np.linalg.norm(emb_i)
            norm_j = np.linalg.norm(emb_j)
            
            if norm_i > 0 and norm_j > 0:
                similarity = dot / (norm_i * norm_j)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_idx = j
        
        df.at[df.index[i], 'Recidivism_Score'] = max_similarity
        
        # Classify based on similarity - using higher thresholds for accuracy
        # 0.85+ = Very similar (near duplicate/same root cause)
        # 0.75-0.85 = Likely related issues
        # 0.65-0.75 = Possibly related, needs review
        # <0.65 = Different enough to be considered separate issues
        if max_similarity >= 0.85:
            df.at[df.index[i], 'Learning_Status'] = '🔴 REPEAT OFFENSE'
            if most_similar_idx >= 0:
                similar_text = str(df.iloc[most_similar_idx].get(COL_SUMMARY, ''))[:100]
                df.at[df.index[i], 'Similar_Historical_Issue'] = similar_text
        elif max_similarity >= 0.75:
            df.at[df.index[i], 'Learning_Status'] = '🟡 POSSIBLE REPEAT'
        elif max_similarity >= 0.65:
            df.at[df.index[i], 'Learning_Status'] = '🟢 Monitored'
        else:
            df.at[df.index[i], 'Learning_Status'] = '🆕 New Issue'
    
    # Log summary
    repeat_count = (df['Learning_Status'].str.contains('REPEAT', na=False)).sum()
    possible_count = (df['Learning_Status'].str.contains('POSSIBLE', na=False)).sum()
    
    logger.info(f"[Audit Engine] Learning analysis complete:")
    logger.info(f"  → {repeat_count} confirmed repeat offenses")
    logger.info(f"  → {possible_count} possible repeats")
    
    return df


class EscalationPipeline:
    """
    Main pipeline orchestrator for Escalation AI.
    
    Coordinates all phases of analysis from data loading to report generation.
    """
    
    def __init__(self):
        self.ai = None
        self.df = None
        self.df_raw = None
        self.file_path = None
        self.output_path = None
        self.feedback_learner = None
        self.price_catalog = None
        
    def initialize(self):
        """Initialize the pipeline components."""
        # Check Ollama server
        if not check_ollama_server():
            return False
        
        # Initialize AI engine
        self.ai = OllamaBrain()
        logger.info(f"AI Initialized. Embed: {self.ai.embed_model} | Gen: {self.ai.gen_model}")
        
        # Validate models
        if not check_models(self.ai):
            return False
        
        # Initialize feedback and pricing systems
        self.feedback_learner = get_feedback_learner()
        self.feedback_learner.load_feedback(self.ai)
        
        self.price_catalog = get_price_catalog()
        self.price_catalog.load_catalog()
        
        return True
    
    def load_data(self, file_path=None):
        """Load data from file."""
        if file_path is None:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select Log File")
            if not file_path:
                return False
        
        self.file_path = file_path
        
        try:
            xls = pd.ExcelFile(file_path)
            sheet = next((s for s in xls.sheet_names if 'raw' in str(s).lower()), xls.sheet_names[0])
            logger.info(f"Loading Sheet: {sheet}")
            self.df = pd.read_excel(file_path, sheet_name=sheet)
            self.df_raw = self.df.copy()
        except Exception as e:
            logger.warning(f"Excel read failed, trying CSV: {e}")
            self.df = pd.read_csv(file_path, engine='python')
            self.df_raw = self.df.copy()
        
        if not validate_data_quality(self.df):
            messagebox.showerror("Data Error", "The selected file contains no usable data.")
            return False
        
        return True
    
    def prepare_text(self):
        """Prepare combined text column for analysis."""
        text_cols = [COL_SUMMARY, COL_CATEGORY]
        actual_cols = [c for c in self.df.columns if c.strip().lower() in [t.lower() for t in text_cols]]
        
        if actual_cols:
            self.df['Combined_Text'] = self.df[actual_cols].apply(
                lambda x: ' - '.join(x.dropna().astype(str)), axis=1
            )
        else:
            self.df['Combined_Text'] = self.df.iloc[:, 0].astype(str)
        
        self.df['Combined_Text'] = self.df['Combined_Text'].apply(clean_text)
    
    def run_classification(self):
        """Phase 1: AI Classification."""
        logger.info("=" * 60)
        logger.info("PHASE 1: AI CLASSIFICATION")
        logger.info("=" * 60)
        self.df = classify_rows(self.df, self.ai)
    
    def run_scoring(self):
        """Phase 2: Strategic Friction Scoring."""
        logger.info("=" * 60)
        logger.info("PHASE 2: STRATEGIC FRICTION SCORING")
        logger.info("=" * 60)
        self.df = calculate_strategic_friction(self.df)
    
    def run_recidivism_analysis(self):
        """Phase 3: Recidivism & Learning Analysis."""
        logger.info("=" * 60)
        logger.info("PHASE 3: RECIDIVISM ANALYSIS")
        logger.info("=" * 60)
        self.df = audit_learning(self.df, self.ai)
    
    def run_recurrence_prediction(self):
        """Phase 4: ML-based Recurrence Prediction."""
        logger.info("=" * 60)
        logger.info("PHASE 4: RECURRENCE PREDICTION")
        logger.info("=" * 60)
        self.df = apply_recurrence_predictions(self.df)
    
    def run_similar_ticket_analysis(self):
        """Phase 5: Similar Ticket Analysis."""
        logger.info("=" * 60)
        logger.info("PHASE 5: SIMILAR TICKET ANALYSIS")
        logger.info("=" * 60)
        self.df = apply_similar_ticket_analysis(self.df, self.ai)
    
    def run_resolution_time_prediction(self):
        """Phase 6: Resolution Time Prediction."""
        logger.info("=" * 60)
        logger.info("PHASE 6: RESOLUTION TIME PREDICTION")
        logger.info("=" * 60)
        self.df = apply_resolution_time_prediction(self.df)
    
    def generate_executive_summary(self):
        """Generate AI executive summary."""
        logger.info(f"[GenAI] Drafting Executive Summary using {GEN_MODEL}...")
        
        total_friction = self.df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in self.df.columns else 0
        total_tickets = len(self.df)
        
        # Build context for synthesis
        context_lines = [
            "=" * 60,
            "ESCALATION REPORT STATISTICAL SUMMARY",
            "=" * 60,
            "",
            f"Total Tickets Analyzed: {total_tickets}",
            f"Total Weighted Friction Score: {total_friction:,.0f}",
        ]
        
        # Add severity breakdown
        if 'Severity_Norm' in self.df.columns:
            severity_counts = self.df['Severity_Norm'].value_counts()
            context_lines.append(f"\nSeverity Distribution:")
            for sev, count in severity_counts.items():
                context_lines.append(f"  - {sev}: {count}")
        
        # Add category breakdown
        if 'AI_Category' in self.df.columns:
            cat_counts = self.df['AI_Category'].value_counts().head(5)
            context_lines.append(f"\nTop Categories:")
            for cat, count in cat_counts.items():
                context_lines.append(f"  - {cat}: {count}")
        
        context = "\n".join(context_lines)
        
        summary = self.ai.generate_synthesis(context)
        return summary
    
    def run_all_phases(self):
        """Run all pipeline phases in sequence."""
        self.prepare_text()
        self.run_classification()
        self.run_scoring()
        self.run_recidivism_analysis()
        self.run_recurrence_prediction()
        self.run_similar_ticket_analysis()
        self.run_resolution_time_prediction()
        
        return self.df
    
    def get_results(self):
        """Get the processed dataframe."""
        return self.df


def main_pipeline():
    """
    Main entry point for the Escalation AI pipeline.
    
    This function:
    1. Shows file selection dialog
    2. Runs all analysis phases
    3. Generates reports and charts
    4. Exports results to Excel
    """
    root = tk.Tk()
    root.withdraw()
    
    try:
        pipeline = EscalationPipeline()
        
        # Initialize
        if not pipeline.initialize():
            return
        
        # Load data
        if not pipeline.load_data():
            return
        
        # Run all phases
        df = pipeline.run_all_phases()
        
        # Generate executive summary
        exec_summary = pipeline.generate_executive_summary()
        logger.info(f"AI Insight: {exec_summary[:100]}...")
        
        # Select output path
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile="Strategic_Report.xlsx"
        )
        if not save_path:
            return
        
        # Import report generator and generate report
        from ..reports import generate_report
        generate_report(df, save_path, exec_summary, pipeline.df_raw)
        
        messagebox.showinfo(
            "Analysis Complete",
            f"Report generated successfully!\n\nSaved to: {save_path}"
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        messagebox.showerror("Error", f"Analysis failed: {e}")
    finally:
        try:
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main_pipeline()
