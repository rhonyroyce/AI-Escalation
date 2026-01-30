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
import sys
import logging
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

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
from ..financial import (
    calculate_financial_metrics,
    calculate_roi_metrics,
    calculate_cost_avoidance,
    calculate_efficiency_metrics,
    calculate_financial_forecasts,
    generate_financial_insights
)

logger = logging.getLogger(__name__)

# Global instances
feedback_learner = None
price_catalog = None


def print_banner(text, char="=", width=60):
    """Print a formatted banner to console."""
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_status(phase, message, icon="→"):
    """Print a status message."""
    print(f"  {icon} {message}")
    sys.stdout.flush()


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
            print_status("init", "Ollama server is running", "✅")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"[Ollama] Server check failed: {e}")
    
    print_status("init", "Ollama server NOT running!", "❌")
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
        print_status("init", f"Testing embedding model: {EMBED_MODEL}...", "🔄")
        test_embed = ai.get_embedding("test")
        if test_embed is None or len(test_embed) == 0:
            print_status("init", f"Embedding model '{EMBED_MODEL}' not available!", "❌")
            messagebox.showerror(
                "Model Error",
                f"Embedding model '{EMBED_MODEL}' is not available.\n\n"
                f"Please install with: ollama pull {EMBED_MODEL}"
            )
            return False
        
        print_status("init", f"Embedding model verified: {EMBED_MODEL} ({len(test_embed)} dims)", "✅")
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


def audit_learning(df, ai, show_progress=True):
    """
    Enhanced recidivism analysis using embeddings.
    
    Identifies tickets similar to past issues to flag potential repeat failures.
    
    Args:
        df: DataFrame with ticket data
        ai: OllamaBrain instance for embeddings
        show_progress: Whether to show progress bar
        
    Returns:
        DataFrame with learning analysis columns added
    """
    print_status("Phase 3", "Calculating embeddings for similarity analysis...", "🧠")
    
    # Initialize columns
    df['Learning_Status'] = 'New'
    df['Recidivism_Score'] = 0.0
    df['Similar_Historical_Issue'] = ''
    
    # Get embeddings for all tickets with progress bar
    texts = df['Combined_Text'].tolist()
    embeddings = []
    
    iterator = tqdm(texts, desc="  Embedding tickets", unit="ticket", 
                   disable=not show_progress, ncols=80, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for text in iterator:
        if pd.isna(text) or str(text).strip() == '':
            embeddings.append(None)
        else:
            emb = ai.get_embedding(str(text))
            embeddings.append(emb)
    
    df['embedding'] = embeddings
    
    # Calculate similarity between all pairs
    print_status("Phase 3", "Computing similarity matrix...", "🔍")
    valid_embeddings = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    
    iterator = tqdm(valid_embeddings, desc="  Finding similar tickets", unit="ticket",
                   disable=not show_progress, ncols=80,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for i, emb_i in iterator:
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
    
    print_status("Phase 3", f"Found {repeat_count} repeat offenses, {possible_count} possible repeats", "📊")
    
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
        self.show_progress = True
        
    def initialize(self):
        """Initialize the pipeline components."""
        print_banner("INITIALIZING ESCALATION AI", "=")
        
        # Check Ollama server
        if not check_ollama_server():
            return False
        
        # Initialize AI engine
        self.ai = OllamaBrain()
        print_status("init", f"Embedding model: {self.ai.embed_model}", "🧠")
        print_status("init", f"Generation model: {self.ai.gen_model}", "🤖")
        
        # Validate models
        if not check_models(self.ai):
            return False
        
        # Initialize feedback and pricing systems
        print_status("init", "Loading feedback learning system...", "📚")
        self.feedback_learner = get_feedback_learner()
        self.feedback_learner.load_feedback(self.ai)
        
        self.price_catalog = get_price_catalog()
        self.price_catalog.load_catalog()
        print_status("init", "Initialization complete!", "✅")
        
        return True
    
    def load_data(self, file_path=None):
        """Load data from file."""
        print_banner("LOADING DATA", "-")
        
        if file_path is None:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select Log File")
            if not file_path:
                return False
        
        self.file_path = file_path
        print_status("load", f"File: {os.path.basename(file_path)}", "📁")
        
        try:
            xls = pd.ExcelFile(file_path)
            sheet = next((s for s in xls.sheet_names if 'raw' in str(s).lower()), xls.sheet_names[0])
            print_status("load", f"Sheet: {sheet}", "📄")
            self.df = pd.read_excel(file_path, sheet_name=sheet)
            self.df_raw = self.df.copy()
        except Exception as e:
            logger.warning(f"Excel read failed, trying CSV: {e}")
            self.df = pd.read_csv(file_path, engine='python')
            self.df_raw = self.df.copy()
        
        if not validate_data_quality(self.df):
            messagebox.showerror("Data Error", "The selected file contains no usable data.")
            return False
        
        print_status("load", f"Loaded {len(self.df):,} tickets with {len(self.df.columns)} columns", "✅")
        return True
    
    def prepare_text(self):
        """Prepare combined text column for analysis."""
        print_status("prep", "Preparing text for analysis...", "📝")
        
        text_cols = [COL_SUMMARY, COL_CATEGORY]
        actual_cols = [c for c in self.df.columns if c.strip().lower() in [t.lower() for t in text_cols]]
        
        if actual_cols:
            self.df['Combined_Text'] = self.df[actual_cols].apply(
                lambda x: ' - '.join(x.dropna().astype(str)), axis=1
            )
        else:
            self.df['Combined_Text'] = self.df.iloc[:, 0].astype(str)
        
        self.df['Combined_Text'] = self.df['Combined_Text'].apply(clean_text)
        print_status("prep", "Text preparation complete", "✅")
    
    def run_classification(self):
        """Phase 1: AI Classification."""
        print_banner("PHASE 1: AI CLASSIFICATION", "─")
        print_status("Phase 1", f"Classifying {len(self.df):,} tickets using {EMBED_MODEL}...", "🏷️")
        self.df = classify_rows(self.df, self.ai, show_progress=self.show_progress)
        
        # Show summary
        if 'AI_Category' in self.df.columns:
            n_categories = self.df['AI_Category'].nunique()
            print_status("Phase 1", f"Classified into {n_categories} categories", "✅")
    
    def run_scoring(self):
        """Phase 2: Strategic Friction Scoring."""
        print_banner("PHASE 2: STRATEGIC FRICTION SCORING", "─")
        print_status("Phase 2", "Calculating friction scores...", "📊")
        self.df = calculate_strategic_friction(self.df)
        
        if 'Strategic_Friction_Score' in self.df.columns:
            total_friction = self.df['Strategic_Friction_Score'].sum()
            avg_friction = self.df['Strategic_Friction_Score'].mean()
            print_status("Phase 2", f"Total friction: {total_friction:,.0f} | Avg: {avg_friction:.1f}", "✅")
    
    def run_recidivism_analysis(self):
        """Phase 3: Recidivism & Learning Analysis."""
        print_banner("PHASE 3: RECIDIVISM ANALYSIS", "─")
        self.df = audit_learning(self.df, self.ai, show_progress=self.show_progress)
    
    def run_recurrence_prediction(self):
        """Phase 4: ML-based Recurrence Prediction."""
        print_banner("PHASE 4: RECURRENCE PREDICTION", "─")
        print_status("Phase 4", "Training recurrence model...", "🔮")
        self.df = apply_recurrence_predictions(self.df)
        
        if 'AI_Recurrence_Probability' in self.df.columns:
            high_risk = (self.df['AI_Recurrence_Probability'] > 0.7).sum()
            print_status("Phase 4", f"Identified {high_risk} high-risk tickets (>70% recurrence)", "✅")
    
    def run_similar_ticket_analysis(self):
        """Phase 5: Similar Ticket Analysis."""
        print_banner("PHASE 5: SIMILAR TICKET ANALYSIS", "─")
        print_status("Phase 5", "Finding similar resolved tickets...", "🔍")
        self.df = apply_similar_ticket_analysis(self.df, self.ai)
        print_status("Phase 5", "Similar ticket analysis complete", "✅")
    
    def run_resolution_time_prediction(self):
        """Phase 6: Resolution Time Prediction."""
        print_banner("PHASE 6: RESOLUTION TIME PREDICTION", "─")
        print_status("Phase 6", "Training resolution time model...", "⏱️")
        self.df = apply_resolution_time_prediction(self.df)
        
        if 'AI_Predicted_Resolution_Hours' in self.df.columns:
            avg_pred = self.df['AI_Predicted_Resolution_Hours'].mean()
            print_status("Phase 6", f"Average predicted resolution: {avg_pred:.1f} hours", "✅")
    
    def generate_executive_summary(self):
        """Generate AI executive summary."""
        print_banner("PHASE 7: EXECUTIVE SUMMARY", "─")
        print_status("Phase 7", f"Generating insights with {GEN_MODEL}...", "✍️")

        total_friction = self.df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in self.df.columns else 0
        total_tickets = len(self.df)

        # Calculate comprehensive financial metrics
        financial_metrics = calculate_financial_metrics(self.df)
        roi_metrics = calculate_roi_metrics(self.df)
        cost_avoidance = calculate_cost_avoidance(self.df)
        efficiency_metrics = calculate_efficiency_metrics(self.df)
        forecasts = calculate_financial_forecasts(self.df)
        insights = generate_financial_insights(self.df)

        # Store metrics for dashboard access
        self.financial_metrics = financial_metrics
        self.roi_metrics = roi_metrics
        self.cost_avoidance = cost_avoidance
        self.efficiency_metrics = efficiency_metrics
        self.financial_forecasts = forecasts
        self.financial_insights = insights

        # Legacy metrics for backward compatibility
        total_financial_impact = financial_metrics.total_cost
        avg_cost_per_ticket = financial_metrics.avg_cost_per_ticket
        max_single_ticket_cost = self.df['Financial_Impact'].max() if 'Financial_Impact' in self.df.columns else 0
        revenue_at_risk = financial_metrics.revenue_at_risk
        labor_cost = total_financial_impact * 0.65
        opportunity_cost = financial_metrics.opportunity_cost
        high_cost_tickets = financial_metrics.high_cost_tickets_count
        high_cost_total = self.df['Financial_Impact'].quantile(0.9) * high_cost_tickets if 'Financial_Impact' in self.df.columns else 0
        recurrence_exposure = financial_metrics.recurrence_exposure
        
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
                context_lines.append(f"  - {sev}: {count} tickets")
        
        # Add category breakdown with counts
        if 'AI_Category' in self.df.columns:
            cat_counts = self.df['AI_Category'].value_counts().head(5)
            context_lines.append(f"\nTop Categories by Ticket Count:")
            for cat, count in cat_counts.items():
                pct = (count / total_tickets * 100)
                context_lines.append(f"  - {cat}: {count} tickets ({pct:.1f}%)")
        
        # FINANCIAL IMPACT SECTION
        context_lines.extend([
            "",
            "=" * 60,
            "FINANCIAL IMPACT METRICS",
            "=" * 60,
            "",
            f"Total Direct Financial Impact: ${total_financial_impact:,.2f}",
            f"Average Cost per Escalation: ${avg_cost_per_ticket:,.2f}",
            f"Median Cost per Escalation: ${financial_metrics.median_cost:,.2f}",
            f"Highest Single Ticket Cost: ${max_single_ticket_cost:,.2f}",
            f"Revenue at Risk (downstream business impact): ${revenue_at_risk:,.2f}",
            f"Labor Cost Component (65%): ${labor_cost:,.2f}",
            f"Opportunity Cost Component (35%): ${opportunity_cost:,.2f}",
            f"Customer Impact Cost (external issues): ${financial_metrics.customer_impact_cost:,.2f}",
            f"SLA Penalty Exposure: ${financial_metrics.sla_penalty_exposure:,.2f}",
            "",
            f"High-Cost Tickets (top 10%): {high_cost_tickets} tickets = ${high_cost_total:,.2f}",
            f"Cost Concentration: {financial_metrics.cost_concentration_ratio*100:.0f}% of costs from top 20% tickets",
            f"Recurrence Risk Exposure: ${recurrence_exposure:,.2f}",
            "",
            "ROI & COST OPTIMIZATION",
            f"Preventable Cost (process improvements): ${financial_metrics.preventable_cost:,.2f}",
            f"Recurring Issue Cost (root cause fixes): ${financial_metrics.recurring_issue_cost:,.2f}",
            f"Total Cost Avoidance Potential: ${cost_avoidance['total_avoidance']:,.2f}",
            f"ROI Opportunity (from prevention): ${financial_metrics.roi_opportunity:,.2f}",
            f"Cost Efficiency Score: {financial_metrics.cost_efficiency_score:.0f}/100",
        ])

        # Add ROI opportunities
        if roi_metrics['top_opportunities']:
            context_lines.append("\nTop ROI Opportunities:")
            for opp in roi_metrics['top_opportunities'][:3]:
                context_lines.append(
                    f"  - {opp['category']}: Invest ${opp['investment_required']:,.0f} → "
                    f"Save ${opp['annual_savings']:,.0f}/year (ROI: {opp['roi_percentage']:.0f}%, "
                    f"Payback: {opp['payback_months']:.1f} months)"
                )

        # Add financial forecasts
        if forecasts['trend'] != 'stable':
            context_lines.extend([
                "",
                "FINANCIAL FORECAST",
                f"Trend: {forecasts['trend'].upper()} (confidence: {forecasts['confidence']})",
                f"30-Day Projection: ${financial_metrics.cost_forecast_30d:,.2f}",
                f"90-Day Projection: ${financial_metrics.cost_forecast_90d:,.2f}",
                f"Annual Projection: ${forecasts['annual_projection']:,.2f}",
            ])

        # Add top insights
        if insights:
            context_lines.append("\nKEY FINANCIAL INSIGHTS:")
            for insight in insights[:3]:
                context_lines.append(
                    f"  [{insight['priority'].upper()}] {insight['title']}: "
                    f"{insight['description']} | {insight['recommendation']}"
                )

        # Financial impact by category
        if 'Financial_Impact' in self.df.columns and 'AI_Category' in self.df.columns:
            fin_by_cat = self.df.groupby('AI_Category')['Financial_Impact'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(5)
            context_lines.append("\nFinancial Impact by Category:")
            for cat in fin_by_cat.index:
                total = fin_by_cat.loc[cat, 'sum']
                avg = fin_by_cat.loc[cat, 'mean']
                count = int(fin_by_cat.loc[cat, 'count'])
                pct = (total / total_financial_impact * 100) if total_financial_impact > 0 else 0
                context_lines.append(f"  - {cat}: ${total:,.2f} total ({pct:.1f}%), ${avg:,.2f} avg, {count} tickets")
        
        # Financial impact by severity
        if 'Financial_Impact' in self.df.columns and 'Severity_Norm' in self.df.columns:
            fin_by_sev = self.df.groupby('Severity_Norm')['Financial_Impact'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
            context_lines.append("\nFinancial Impact by Severity:")
            for sev in fin_by_sev.index:
                total = fin_by_sev.loc[sev, 'sum']
                avg = fin_by_sev.loc[sev, 'mean']
                pct = (total / total_financial_impact * 100) if total_financial_impact > 0 else 0
                context_lines.append(f"  - {sev}: ${total:,.2f} ({pct:.1f}%), ${avg:,.2f} avg per ticket")
        
        # Cost efficiency insight
        if 'Financial_Impact' in self.df.columns and 'AI_Category' in self.df.columns:
            # Find highest cost-per-ticket categories
            cost_efficiency = self.df.groupby('AI_Category')['Financial_Impact'].mean().sort_values(ascending=False).head(3)
            context_lines.append("\nHighest Avg Cost per Ticket (categories to watch):")
            for cat, avg in cost_efficiency.items():
                context_lines.append(f"  - {cat}: ${avg:,.2f} per ticket")
        
        context = "\n".join(context_lines)
        
        summary = self.ai.generate_synthesis(context)
        print_status("Phase 7", "Executive summary generated", "✅")
        return summary
    
    def run_all_phases(self):
        """Run all pipeline phases in sequence."""
        total_phases = 6
        
        print_banner(f"RUNNING ANALYSIS PIPELINE ({len(self.df):,} tickets)", "═")
        print()
        
        self.prepare_text()
        self.run_classification()       # Phase 1
        self.run_scoring()              # Phase 2
        self.run_recidivism_analysis()  # Phase 3
        self.run_recurrence_prediction() # Phase 4
        self.run_similar_ticket_analysis() # Phase 5
        self.run_resolution_time_prediction() # Phase 6
        
        print_banner("ALL PHASES COMPLETE", "═")
        
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
        
        # Select output path
        print_banner("SAVING REPORT", "-")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile="Strategic_Report.xlsx"
        )
        if not save_path:
            return
        
        # Import report generator and generate report
        print_status("save", "Generating Excel report with charts...", "📊")
        from ..reports import generate_report
        generate_report(df, save_path, exec_summary, pipeline.df_raw)
        
        print_status("save", f"Saved to: {save_path}", "✅")
        print_banner("ANALYSIS COMPLETE! 🎉", "═")
        
        messagebox.showinfo(
            "Analysis Complete",
            f"Report generated successfully!\n\nSaved to: {save_path}"
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        messagebox.showerror("Error", f"Analysis failed: {e}")
    finally:
        try:
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main_pipeline()
