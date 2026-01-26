"""
Human-in-the-loop feedback and learning system.
Enables iterative improvement of AI classification through user corrections.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

from ..core.config import (
    FEEDBACK_FILE, FEEDBACK_WEIGHT, ANCHORS, 
    REPORT_VERSION, MIN_CLASSIFICATION_CONFIDENCE
)

logger = logging.getLogger(__name__)


class FeedbackLearning:
    """
    Human-in-the-loop learning system for classification improvement.
    
    Workflow:
    1. Run analysis → generates classification_feedback.xlsx
    2. User reviews and corrects categories in the Excel file
    3. Next run loads corrections and adjusts centroids
    """
    
    def __init__(self, feedback_path: Optional[str] = None):
        self.feedback_path = feedback_path or FEEDBACK_FILE
        self.corrections: Dict[str, List[Dict]] = {}  # category -> list of {text, embedding}
        self.stats = {'loaded': 0, 'categories_enhanced': 0}
    
    def load_feedback(self, ai) -> bool:
        """Load user feedback from Excel file and compute embeddings for corrections."""
        if not os.path.exists(self.feedback_path):
            logger.info(f"No feedback file found at {self.feedback_path} - using default anchors")
            return False
        
        try:
            df_feedback = pd.read_excel(self.feedback_path, sheet_name='Classifications')
            
            corrections_list = []
            for _, row in df_feedback.iterrows():
                original = str(row.get('AI_Category', '')).strip()
                corrected = str(row.get('Corrected_Category', '')).strip()
                text = str(row.get('Text', '')).strip()
                
                if corrected and corrected.lower() not in ['', 'nan', 'none'] and corrected != original and text:
                    corrections_list.append({
                        'text': text,
                        'category': corrected,
                        'original': original
                    })
            
            if not corrections_list:
                logger.info("Feedback file found but no corrections to apply")
                return False
            
            logger.info(f"Found {len(corrections_list)} user corrections in feedback file")
            
            # Compute embeddings for corrected texts
            texts = [c['text'] for c in corrections_list]
            embeddings = ai.get_embeddings_batch(texts)
            
            # Group by corrected category
            for corr, emb in zip(corrections_list, embeddings):
                cat = corr['category']
                if cat not in self.corrections:
                    self.corrections[cat] = []
                self.corrections[cat].append({
                    'text': corr['text'],
                    'embedding': emb,
                    'original': corr['original']
                })
            
            self.stats['loaded'] = len(corrections_list)
            self.stats['categories_enhanced'] = len(self.corrections)
            
            logger.info(f"✓ Loaded {self.stats['loaded']} corrections across {self.stats['categories_enhanced']} categories")
            for cat, items in self.corrections.items():
                logger.info(f"  - {cat}: {len(items)} examples")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load feedback file: {e}")
            return False
    
    def adjust_centroids(self, original_centroids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Blend user feedback with original anchor centroids."""
        if not self.corrections:
            return original_centroids
        
        adjusted = {}
        
        for category, centroid in original_centroids.items():
            if category in self.corrections:
                feedback_vecs = [c['embedding'] for c in self.corrections[category]]
                feedback_centroid = np.mean(feedback_vecs, axis=0)
                
                adjusted[category] = (
                    (1 - FEEDBACK_WEIGHT) * centroid + 
                    FEEDBACK_WEIGHT * feedback_centroid
                )
                logger.info(f"  → Adjusted '{category}' centroid with {len(feedback_vecs)} user examples")
            else:
                adjusted[category] = centroid
        
        # Handle new categories from feedback
        for category in self.corrections:
            if category not in adjusted:
                feedback_vecs = [c['embedding'] for c in self.corrections[category]]
                adjusted[category] = np.mean(feedback_vecs, axis=0)
                logger.info(f"  → Created NEW category '{category}' from {len(feedback_vecs)} user examples")
        
        return adjusted
    
    def save_for_review(self, df: pd.DataFrame, output_dir: str) -> str:
        """Export classifications to Excel for user review and correction."""
        
        feedback_rows = []
        for idx, row in df.iterrows():
            feedback_rows.append({
                'ID': str(idx),
                'Text': str(row.get('Combined_Text', ''))[:500],
                'AI_Category': row.get('AI_Category', 'Unclassified'),
                'AI_Confidence': round(float(row.get('AI_Confidence', 0)), 3),
                'Root_Cause_Category': row.get('Root_Cause_Category', 'Unclassified'),
                'PM_Recurrence_Risk': row.get('PM_Recurrence_Risk_Norm', 'Unknown'),
                'AI_Recurrence_Risk': row.get('AI_Recurrence_Risk', 'Unknown'),
                'AI_Recurrence_Prob': f"{row.get('AI_Recurrence_Probability', 0)*100:.0f}%",
                'PM_Prediction_Accuracy': row.get('PM_Prediction_Accuracy', 'Pending'),
                'LOB': row.get('LOB', 'Unknown'),
                'Corrected_Category': '',
                'Notes': ''
            })
        
        df_feedback = pd.DataFrame(feedback_rows)
        export_path = os.path.join(output_dir, 'classification_feedback.xlsx')
        
        with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
            # Instructions sheet
            instructions_df = self._create_instructions_df()
            instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
            
            # Classifications sheet
            df_feedback.to_excel(writer, sheet_name='Classifications', index=False)
            
            # Category reference sheet
            categories_df = pd.DataFrame({
                'Available_Categories': list(ANCHORS.keys()),
                'Example_Keywords': [', '.join(phrases[:3]) for phrases in ANCHORS.values()]
            })
            categories_df.to_excel(writer, sheet_name='Category Reference', index=False)
            
            # Format the workbook
            self._format_feedback_workbook(writer, df_feedback)
        
        # Save to working directory for next run
        try:
            df_feedback.to_excel(self.feedback_path, sheet_name='Classifications', index=False)
        except Exception as e:
            logger.warning(f"Could not save feedback to working directory: {e}")
        
        logger.info(f"✓ Feedback Excel saved to: {export_path}")
        return export_path
    
    def _create_instructions_df(self) -> pd.DataFrame:
        """Create instructions dataframe for feedback file."""
        return pd.DataFrame({
            'Instructions': [
                'HOW TO IMPROVE AI CLASSIFICATION',
                '',
                '1. Go to the "Classifications" sheet',
                '2. Review each row - check if AI_Category is correct',
                '3. If WRONG: Select the correct category from "Corrected_Category" dropdown',
                '4. If CORRECT: Leave "Corrected_Category" blank',
                '5. Save this file',
                '6. Run the analysis again - AI will learn from your corrections!',
                '',
                'AVAILABLE CATEGORIES:',
                *[f'  • {cat}' for cat in ANCHORS.keys()],
                '',
                'TIP: Yellow rows = Low confidence (<50%) - priority for review!',
                '',
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                f'Version: {REPORT_VERSION}'
            ]
        })
    
    def _format_feedback_workbook(self, writer, df_feedback: pd.DataFrame):
        """Apply formatting to feedback workbook."""
        ws = writer.sheets['Classifications']
        
        # Column widths
        widths = {'A': 8, 'B': 80, 'C': 25, 'D': 12, 'E': 18, 'F': 16, 
                  'G': 22, 'H': 14, 'I': 20, 'J': 12, 'K': 25, 'L': 30}
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        
        # Category dropdown
        category_list = ','.join(list(ANCHORS.keys()))
        category_validation = DataValidation(
            type='list',
            formula1=f'"{category_list}"',
            allow_blank=True,
            showDropDown=False
        )
        category_validation.add(f'K2:K{len(df_feedback) + 1}')
        ws.add_data_validation(category_validation)
        
        # Header styling
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="004C97", end_color="004C97", fill_type="solid")
        
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
        
        # Highlight low confidence rows
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        for row_idx in range(2, len(df_feedback) + 2):
            try:
                confidence = float(ws.cell(row=row_idx, column=4).value or 0)
                if confidence < 0.5:
                    for col_idx in range(1, 13):
                        ws.cell(row=row_idx, column=col_idx).fill = yellow_fill
            except (ValueError, TypeError):
                pass
        
        ws.freeze_panes = 'A2'


# Global feedback learner instance
_feedback_learner: Optional[FeedbackLearning] = None

def get_feedback_learner() -> FeedbackLearning:
    """Get or create the feedback learner singleton."""
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = FeedbackLearning()
    return _feedback_learner
