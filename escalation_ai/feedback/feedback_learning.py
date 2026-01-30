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
    FEEDBACK_FILE, FEEDBACK_WEIGHT, ANCHORS, SUB_CATEGORIES,
    REPORT_VERSION, MIN_CLASSIFICATION_CONFIDENCE
)

logger = logging.getLogger(__name__)


class FeedbackLearning:
    """
    Human-in-the-loop learning system for classification improvement.

    Workflow:
    1. Run analysis → generates classification_feedback.xlsx
    2. User reviews and corrects categories in the Excel file
    3. User can add NEW categories in Category Reference tab
    4. Next run loads corrections and new categories, adjusts centroids
    """

    def __init__(self, feedback_path: Optional[str] = None):
        self.feedback_path = feedback_path or FEEDBACK_FILE
        self.corrections: Dict[str, List[Dict]] = {}  # category -> list of {text, embedding}
        self.custom_categories: Dict[str, List[str]] = {}  # category -> list of keywords (from Category Reference)
        self.stats = {'loaded': 0, 'categories_enhanced': 0, 'custom_categories': 0}

    def load_category_reference(self) -> Dict[str, List[str]]:
        """
        Load categories from the Category Reference tab.

        This allows users to add new categories by editing the Excel file.
        Returns dict of {category_name: [keywords]} including both ANCHORS and user-added.
        """
        # Start with default ANCHORS
        all_categories = {cat: list(keywords) for cat, keywords in ANCHORS.items()}

        if not os.path.exists(self.feedback_path):
            return all_categories

        try:
            df_ref = pd.read_excel(self.feedback_path, sheet_name='Category Reference')

            for _, row in df_ref.iterrows():
                cat_name = str(row.get('Available_Categories', '')).strip()
                keywords_str = str(row.get('Example_Keywords', '')).strip()

                if cat_name and cat_name.lower() not in ['', 'nan', 'none']:
                    # Parse keywords (comma-separated)
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

                    if cat_name not in all_categories:
                        # New user-added category
                        all_categories[cat_name] = keywords if keywords else [cat_name.lower()]
                        self.custom_categories[cat_name] = keywords if keywords else [cat_name.lower()]
                        logger.info(f"  → Loaded custom category '{cat_name}' with {len(keywords)} keywords")
                    elif keywords and cat_name in ANCHORS:
                        # User may have added additional keywords to existing category
                        existing_keywords = set(ANCHORS[cat_name])
                        new_keywords = [k for k in keywords if k not in existing_keywords]
                        if new_keywords:
                            all_categories[cat_name] = list(ANCHORS[cat_name]) + new_keywords

            if self.custom_categories:
                self.stats['custom_categories'] = len(self.custom_categories)
                logger.info(f"✓ Loaded {len(self.custom_categories)} custom categories from Category Reference")

        except Exception as e:
            logger.warning(f"Could not load Category Reference: {e}")

        return all_categories
    
    def load_feedback(self, ai) -> bool:
        """
        Load user feedback from Excel file and compute embeddings for corrections.

        Also loads custom categories from Category Reference tab and creates
        embeddings for them based on their keywords.
        """
        if not os.path.exists(self.feedback_path):
            logger.info(f"No feedback file found at {self.feedback_path} - using default anchors")
            return False

        has_feedback = False

        # First, load custom categories from Category Reference tab
        try:
            all_categories = self.load_category_reference()

            # Create embeddings for custom categories (not in ANCHORS)
            if self.custom_categories:
                logger.info(f"Creating embeddings for {len(self.custom_categories)} custom categories...")
                for cat_name, keywords in self.custom_categories.items():
                    if cat_name not in self.corrections:
                        self.corrections[cat_name] = []

                    # Create embeddings from keywords
                    keyword_embeddings = ai.get_embeddings_batch(keywords)
                    for kw, emb in zip(keywords, keyword_embeddings):
                        if emb is not None:
                            self.corrections[cat_name].append({
                                'text': kw,
                                'embedding': emb,
                                'original': 'custom_category'
                            })

                    logger.info(f"  → Created {len(keyword_embeddings)} embeddings for '{cat_name}'")
                has_feedback = True

        except Exception as e:
            logger.warning(f"Could not load custom categories: {e}")

        # Then load corrections from Classifications sheet
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

            if corrections_list:
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
                has_feedback = True
            else:
                logger.info("No corrections found in Classifications sheet")

        except Exception as e:
            logger.warning(f"Failed to load feedback corrections: {e}")

        # Update stats
        self.stats['categories_enhanced'] = len(self.corrections)

        if has_feedback:
            logger.info(f"✓ Loaded feedback: {self.stats['loaded']} corrections, "
                       f"{self.stats['custom_categories']} custom categories, "
                       f"{self.stats['categories_enhanced']} total categories enhanced")
            for cat, items in self.corrections.items():
                logger.info(f"  - {cat}: {len(items)} examples")

        return has_feedback
    
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
        """
        Export classifications to Excel for user review and correction.

        IMPORTANT: This method preserves existing Corrected_Category entries
        from previous feedback files, allowing continuous human-in-the-loop learning.
        New tickets are appended, existing tickets retain their user corrections.
        """

        # Load existing feedback to preserve user corrections
        existing_corrections = {}
        existing_notes = {}
        if os.path.exists(self.feedback_path):
            try:
                df_existing = pd.read_excel(self.feedback_path, sheet_name='Classifications')
                for _, row in df_existing.iterrows():
                    # Create a key from ID and truncated text for matching
                    text_key = str(row.get('Text', ''))[:200].strip().lower()
                    row_id = str(row.get('ID', ''))
                    key = f"{row_id}_{text_key}"

                    corrected = str(row.get('Corrected_Category', '')).strip()
                    notes = str(row.get('Notes', '')).strip()

                    if corrected and corrected.lower() not in ['', 'nan', 'none']:
                        existing_corrections[key] = corrected
                    if notes and notes.lower() not in ['', 'nan', 'none']:
                        existing_notes[key] = notes

                logger.info(f"Loaded {len(existing_corrections)} existing corrections from feedback file")
            except Exception as e:
                logger.warning(f"Could not load existing feedback for preservation: {e}")

        feedback_rows = []
        for idx, row in df.iterrows():
            text_truncated = str(row.get('Combined_Text', ''))[:500]
            text_key = text_truncated[:200].strip().lower()
            row_id = str(idx)
            key = f"{row_id}_{text_key}"

            # Preserve existing Corrected_Category if it exists
            preserved_correction = existing_corrections.get(key, '')
            preserved_notes = existing_notes.get(key, '')

            feedback_rows.append({
                'ID': row_id,
                'Text': text_truncated,
                'AI_Category': row.get('AI_Category', 'Unclassified'),
                'AI_Sub_Category': row.get('AI_Sub_Category', 'General'),
                'AI_Confidence': round(float(row.get('AI_Confidence', 0)), 3),
                'Root_Cause_Category': row.get('Root_Cause_Category', 'Unclassified'),
                'PM_Recurrence_Risk': row.get('PM_Recurrence_Risk_Norm', 'Unknown'),
                'AI_Recurrence_Risk': row.get('AI_Recurrence_Risk', 'Unknown'),
                'AI_Recurrence_Prob': f"{row.get('AI_Recurrence_Probability', 0)*100:.0f}%",
                'PM_Prediction_Accuracy': row.get('PM_Prediction_Accuracy', 'Pending'),
                'LOB': row.get('LOB', 'Unknown'),
                'Corrected_Category': preserved_correction,
                'Notes': preserved_notes
            })

        df_feedback = pd.DataFrame(feedback_rows)
        export_path = os.path.join(output_dir, 'classification_feedback.xlsx')

        # Build category reference - preserve user-added categories
        all_categories = self._build_category_reference()
        categories_df = pd.DataFrame({
            'Available_Categories': list(all_categories.keys()),
            'Example_Keywords': [', '.join(kw[:5]) for kw in all_categories.values()]
        })

        # Build sub-category reference
        sub_cat_rows = []
        for cat, sub_cats in SUB_CATEGORIES.items():
            for sub_cat, keywords in sub_cats.items():
                sub_cat_rows.append({
                    'Category': cat,
                    'Sub_Category': sub_cat,
                    'Keywords': ', '.join(keywords[:5])
                })
        sub_categories_df = pd.DataFrame(sub_cat_rows)

        with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
            # Instructions sheet
            instructions_df = self._create_instructions_df(all_categories)
            instructions_df.to_excel(writer, sheet_name='Instructions', index=False)

            # Classifications sheet
            df_feedback.to_excel(writer, sheet_name='Classifications', index=False)

            # Category reference sheet
            categories_df.to_excel(writer, sheet_name='Category Reference', index=False)

            # Sub-Category reference sheet
            sub_categories_df.to_excel(writer, sheet_name='Sub-Category Reference', index=False)

            # Format the workbook with dropdown referencing Category Reference
            self._format_feedback_workbook(writer, df_feedback, all_categories)

        # Save to working directory for next run (with all sheets for proper formatting)
        try:
            with pd.ExcelWriter(self.feedback_path, engine='openpyxl') as wd_writer:
                # Instructions sheet
                instructions_df = self._create_instructions_df(all_categories)
                instructions_df.to_excel(wd_writer, sheet_name='Instructions', index=False)

                # Classifications sheet
                df_feedback.to_excel(wd_writer, sheet_name='Classifications', index=False)

                # Category reference sheet
                categories_df.to_excel(wd_writer, sheet_name='Category Reference', index=False)

                # Sub-Category reference sheet
                sub_categories_df.to_excel(wd_writer, sheet_name='Sub-Category Reference', index=False)

                # Apply formatting
                self._format_feedback_workbook(wd_writer, df_feedback, all_categories)

            logger.info(f"✓ Feedback file updated at: {self.feedback_path}")
        except Exception as e:
            logger.warning(f"Could not save feedback to working directory: {e}")
        
        logger.info(f"✓ Feedback Excel saved to: {export_path}")
        return export_path

    def _build_category_reference(self) -> Dict[str, List[str]]:
        """
        Build category reference by merging ANCHORS with user-added categories.

        Preserves any categories the user has added to the Category Reference tab.
        """
        # Start with default ANCHORS
        all_categories = {cat: list(keywords) for cat, keywords in ANCHORS.items()}

        # Load existing user-added categories from feedback file
        if os.path.exists(self.feedback_path):
            try:
                df_ref = pd.read_excel(self.feedback_path, sheet_name='Category Reference')

                for _, row in df_ref.iterrows():
                    cat_name = str(row.get('Available_Categories', '')).strip()
                    keywords_str = str(row.get('Example_Keywords', '')).strip()

                    if cat_name and cat_name.lower() not in ['', 'nan', 'none']:
                        # Parse keywords
                        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

                        if cat_name not in all_categories:
                            # User-added category - preserve it
                            all_categories[cat_name] = keywords if keywords else [cat_name.lower()]
                            logger.info(f"  → Preserved user category: '{cat_name}'")
                        elif keywords:
                            # User may have added keywords to existing category
                            existing = set(ANCHORS.get(cat_name, []))
                            new_kw = [k for k in keywords if k not in existing]
                            if new_kw:
                                all_categories[cat_name] = list(ANCHORS[cat_name]) + new_kw

            except Exception as e:
                logger.debug(f"Could not load existing Category Reference: {e}")

        return all_categories

    def _create_instructions_df(self, all_categories: Dict[str, List[str]] = None) -> pd.DataFrame:
        """Create instructions dataframe for feedback file."""
        if all_categories is None:
            all_categories = ANCHORS

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
                'ADDING NEW CATEGORIES:',
                '1. Go to the "Category Reference" sheet',
                '2. Add a new row with category name in "Available_Categories"',
                '3. Add example keywords in "Example_Keywords" (comma-separated)',
                '4. Save and re-run - the new category will be available!',
                '',
                'AVAILABLE CATEGORIES:',
                *[f'  • {cat}' for cat in all_categories.keys()],
                '',
                'TIP: Yellow rows = Low confidence (<50%) - priority for review!',
                '',
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                f'Version: {REPORT_VERSION}'
            ]
        })
    
    def _format_feedback_workbook(self, writer, df_feedback: pd.DataFrame,
                                    all_categories: Dict[str, List[str]] = None):
        """Apply formatting to feedback workbook."""
        ws = writer.sheets['Classifications']

        # Column widths
        widths = {'A': 8, 'B': 80, 'C': 25, 'D': 12, 'E': 18, 'F': 16,
                  'G': 22, 'H': 14, 'I': 20, 'J': 12, 'K': 25, 'L': 30}
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

        # Category dropdown - reference Category Reference sheet directly
        # This allows dropdown to update when user adds categories to Category Reference
        if all_categories is None:
            all_categories = ANCHORS

        # Use a reference to the Category Reference sheet's Available_Categories column
        # Use a large range (A2:A100) so users can add more categories without editing the formula
        # Empty cells in the range are ignored by Excel's dropdown
        category_validation = DataValidation(
            type='list',
            formula1="'Category Reference'!$A$2:$A$100",
            allow_blank=True,
            showDropDown=False  # In openpyxl, False actually shows the dropdown
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

        # Also format the Category Reference sheet
        if 'Category Reference' in writer.sheets:
            ws_ref = writer.sheets['Category Reference']
            ws_ref.column_dimensions['A'].width = 30
            ws_ref.column_dimensions['B'].width = 60

            # Header styling for Category Reference
            for cell in ws_ref[1]:
                cell.font = header_font
                cell.fill = header_fill

            ws_ref.freeze_panes = 'A2'


# Global feedback learner instance
_feedback_learner: Optional[FeedbackLearning] = None

def get_feedback_learner() -> FeedbackLearning:
    """Get or create the feedback learner singleton."""
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = FeedbackLearning()
    return _feedback_learner
