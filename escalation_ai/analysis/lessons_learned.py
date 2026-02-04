"""
Lessons Learned Effectiveness Analyzer.

Analyzes how well the organization is learning from past incidents by correlating:
- Recurrence rates (from similarity analysis)
- Lesson documentation status
- Resolution consistency
- Repeat patterns by engineer/market

Provides grades (A-F) and AI-generated recommendations for improvement.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.config import (
    COL_LESSON_TITLE, COL_LESSON_STATUS, COL_ENGINEER, COL_LOB,
    OLLAMA_BASE_URL, GEN_MODEL
)

logger = logging.getLogger(__name__)


class LearningGrade(Enum):
    """Learning effectiveness grade."""
    A = "A"  # Excellent - Low recurrence, lessons documented, consistent resolution
    B = "B"  # Good - Moderate recurrence, lessons documented
    C = "C"  # Improving - High recurrence but lessons being documented
    D = "D"  # Poor - High recurrence, few lessons documented
    F = "F"  # Failing - High recurrence, no lessons, inconsistent resolutions


@dataclass
class LearningScore:
    """Learning effectiveness score for a category/entity."""
    entity: str
    entity_type: str  # 'category', 'engineer', 'lob'
    grade: LearningGrade
    score: float  # 0-100
    recurrence_rate: float
    lesson_completion_rate: float
    resolution_consistency_rate: float
    ticket_count: int
    lessons_documented: int
    lessons_completed: int
    recurring_issues: int
    recommendation: str = ""


class LessonsLearnedAnalyzer:
    """
    Analyzes lessons learned effectiveness across categories, engineers, and LOBs.

    Combines multiple signals:
    - AI_Recurrence_Probability from similarity analysis
    - tickets_data_lessons_learned_title/status
    - Resolution_Consistency from similarity matching
    - Repeat issue patterns
    """

    # Grade thresholds (score out of 100)
    GRADE_THRESHOLDS = {
        LearningGrade.A: 80,
        LearningGrade.B: 65,
        LearningGrade.C: 50,
        LearningGrade.D: 35,
        LearningGrade.F: 0,
    }

    # Weight factors for score calculation
    WEIGHTS = {
        'recurrence': 0.35,      # Lower recurrence = better
        'lesson_completion': 0.30,  # Higher completion = better
        'consistency': 0.25,     # More consistent = better
        'lesson_documented': 0.10,  # Has any lesson = bonus
    }

    def __init__(self):
        self.category_scores: Dict[str, LearningScore] = {}
        self.engineer_scores: Dict[str, LearningScore] = {}
        self.lob_scores: Dict[str, LearningScore] = {}
        self.overall_stats: Dict[str, Any] = {}
        self.recommendations: List[Dict[str, str]] = []

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full lessons learned analysis on the dataset.

        Returns:
            Dict with category_scores, engineer_scores, lob_scores,
            overall_stats, and recommendations
        """
        logger.info("Starting Lessons Learned effectiveness analysis...")

        # Identify relevant columns
        lesson_title_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title', 'Lesson_Title'])
        lesson_status_col = self._find_column(df, [COL_LESSON_STATUS, 'lessons_learned_status', 'Lesson_Status'])
        engineer_col = self._find_column(df, [COL_ENGINEER, 'Engineer', 'engineer_name'])
        lob_col = self._find_column(df, [COL_LOB, 'LOB', 'lob'])

        # Analyze by category
        if 'AI_Category' in df.columns:
            self.category_scores = self._analyze_by_group(
                df, 'AI_Category', 'category',
                lesson_title_col, lesson_status_col
            )

        # Analyze by engineer
        if engineer_col:
            self.engineer_scores = self._analyze_by_group(
                df, engineer_col, 'engineer',
                lesson_title_col, lesson_status_col
            )

        # Analyze by LOB/Market
        if lob_col:
            self.lob_scores = self._analyze_by_group(
                df, lob_col, 'lob',
                lesson_title_col, lesson_status_col
            )

        # Calculate overall statistics
        self.overall_stats = self._calculate_overall_stats(df, lesson_title_col, lesson_status_col)

        # Generate AI recommendations
        self.recommendations = self._generate_recommendations()

        results = {
            'category_scores': self.category_scores,
            'engineer_scores': self.engineer_scores,
            'lob_scores': self.lob_scores,
            'overall_stats': self.overall_stats,
            'recommendations': self.recommendations,
        }

        logger.info(f"Lessons Learned analysis complete: {len(self.category_scores)} categories, "
                   f"{len(self.engineer_scores)} engineers, {len(self.lob_scores)} LOBs analyzed")

        return results

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _analyze_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        entity_type: str,
        lesson_title_col: Optional[str],
        lesson_status_col: Optional[str]
    ) -> Dict[str, LearningScore]:
        """Analyze learning effectiveness by a grouping column."""
        scores = {}

        for entity, group in df.groupby(group_col):
            if pd.isna(entity) or str(entity).strip() in ['', 'nan', 'None', '0']:
                continue

            entity_str = str(entity)
            ticket_count = len(group)

            # Skip if too few tickets
            if ticket_count < 3:
                continue

            # Calculate recurrence rate
            recurrence_rate = 0.0
            if 'AI_Recurrence_Probability' in group.columns:
                recurrence_rate = group['AI_Recurrence_Probability'].mean() * 100

            # Calculate lesson metrics
            lessons_documented = 0
            lessons_completed = 0
            lesson_completion_rate = 0.0

            if lesson_title_col and lesson_title_col in group.columns:
                # Count non-empty lesson titles
                lessons_documented = group[lesson_title_col].notna().sum()
                lessons_documented += (group[lesson_title_col].astype(str).str.strip() != '').sum()
                lessons_documented = min(lessons_documented, ticket_count)  # Cap at ticket count

            if lesson_status_col and lesson_status_col in group.columns:
                # Count completed lessons
                status_lower = group[lesson_status_col].astype(str).str.lower()
                lessons_completed = (
                    status_lower.str.contains('complete', na=False) |
                    status_lower.str.contains('done', na=False) |
                    status_lower.str.contains('closed', na=False)
                ).sum()

            if lessons_documented > 0:
                lesson_completion_rate = (lessons_completed / lessons_documented) * 100
            elif lessons_completed > 0:
                lesson_completion_rate = 100.0

            # Calculate resolution consistency rate
            consistency_rate = 50.0  # Default
            if 'Resolution_Consistency' in group.columns:
                consistent = (group['Resolution_Consistency'] == 'Consistent').sum()
                total_with_data = (group['Resolution_Consistency'] != 'No Similar Data').sum()
                if total_with_data > 0:
                    consistency_rate = (consistent / total_with_data) * 100

            # Count recurring issues
            recurring_issues = 0
            if 'Similar_Ticket_Count' in group.columns:
                recurring_issues = (group['Similar_Ticket_Count'] >= 3).sum()

            # Calculate overall score
            score = self._calculate_score(
                recurrence_rate=recurrence_rate,
                lesson_completion_rate=lesson_completion_rate,
                consistency_rate=consistency_rate,
                has_lessons=lessons_documented > 0
            )

            # Assign grade
            grade = self._score_to_grade(score)

            scores[entity_str] = LearningScore(
                entity=entity_str,
                entity_type=entity_type,
                grade=grade,
                score=score,
                recurrence_rate=recurrence_rate,
                lesson_completion_rate=lesson_completion_rate,
                resolution_consistency_rate=consistency_rate,
                ticket_count=ticket_count,
                lessons_documented=lessons_documented,
                lessons_completed=lessons_completed,
                recurring_issues=recurring_issues
            )

        return scores

    def _calculate_score(
        self,
        recurrence_rate: float,
        lesson_completion_rate: float,
        consistency_rate: float,
        has_lessons: bool
    ) -> float:
        """
        Calculate learning effectiveness score (0-100).

        Components:
        - Lower recurrence = higher score
        - Higher lesson completion = higher score
        - Higher consistency = higher score
        - Having lessons documented = bonus
        """
        # Recurrence component (inverse - lower is better)
        # 0% recurrence = 100 points, 100% recurrence = 0 points
        recurrence_score = max(0, 100 - recurrence_rate)

        # Lesson completion component
        completion_score = lesson_completion_rate

        # Consistency component
        consistency_score = consistency_rate

        # Lesson documented bonus
        lesson_bonus = 100 if has_lessons else 0

        # Weighted average
        score = (
            self.WEIGHTS['recurrence'] * recurrence_score +
            self.WEIGHTS['lesson_completion'] * completion_score +
            self.WEIGHTS['consistency'] * consistency_score +
            self.WEIGHTS['lesson_documented'] * lesson_bonus
        )

        return round(min(100, max(0, score)), 1)

    def _score_to_grade(self, score: float) -> LearningGrade:
        """Convert numeric score to letter grade."""
        if score >= self.GRADE_THRESHOLDS[LearningGrade.A]:
            return LearningGrade.A
        elif score >= self.GRADE_THRESHOLDS[LearningGrade.B]:
            return LearningGrade.B
        elif score >= self.GRADE_THRESHOLDS[LearningGrade.C]:
            return LearningGrade.C
        elif score >= self.GRADE_THRESHOLDS[LearningGrade.D]:
            return LearningGrade.D
        else:
            return LearningGrade.F

    def _calculate_overall_stats(
        self,
        df: pd.DataFrame,
        lesson_title_col: Optional[str],
        lesson_status_col: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate organization-wide learning statistics."""
        stats = {
            'total_tickets': len(df),
            'avg_recurrence_rate': 0.0,
            'total_lessons_documented': 0,
            'total_lessons_completed': 0,
            'lesson_documentation_rate': 0.0,
            'lesson_completion_rate': 0.0,
            'overall_grade': LearningGrade.C,
            'overall_score': 50.0,
            'categories_by_grade': {g.value: 0 for g in LearningGrade},
            'at_risk_categories': [],
            'top_learners': [],
            'needs_attention': [],
        }

        if len(df) == 0:
            return stats

        # Recurrence rate
        if 'AI_Recurrence_Probability' in df.columns:
            stats['avg_recurrence_rate'] = df['AI_Recurrence_Probability'].mean() * 100

        # Lesson metrics
        if lesson_title_col and lesson_title_col in df.columns:
            stats['total_lessons_documented'] = df[lesson_title_col].notna().sum()

        if lesson_status_col and lesson_status_col in df.columns:
            status_lower = df[lesson_status_col].astype(str).str.lower()
            stats['total_lessons_completed'] = (
                status_lower.str.contains('complete', na=False) |
                status_lower.str.contains('done', na=False)
            ).sum()

        if stats['total_tickets'] > 0:
            stats['lesson_documentation_rate'] = (stats['total_lessons_documented'] / stats['total_tickets']) * 100

        if stats['total_lessons_documented'] > 0:
            stats['lesson_completion_rate'] = (stats['total_lessons_completed'] / stats['total_lessons_documented']) * 100

        # Grade distribution from categories
        for score in self.category_scores.values():
            stats['categories_by_grade'][score.grade.value] += 1

        # Identify at-risk categories (grade D or F)
        stats['at_risk_categories'] = [
            s.entity for s in self.category_scores.values()
            if s.grade in [LearningGrade.D, LearningGrade.F]
        ]

        # Top learners (grade A or B)
        stats['top_learners'] = [
            s.entity for s in self.category_scores.values()
            if s.grade in [LearningGrade.A, LearningGrade.B]
        ]

        # Categories needing attention (high recurrence, low lessons)
        stats['needs_attention'] = [
            s.entity for s in self.category_scores.values()
            if s.recurrence_rate > 30 and s.lesson_completion_rate < 50
        ]

        # Overall score (average of category scores)
        if self.category_scores:
            stats['overall_score'] = np.mean([s.score for s in self.category_scores.values()])
            stats['overall_grade'] = self._score_to_grade(stats['overall_score'])

        return stats

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Sort categories by score (lowest first = most urgent)
        sorted_categories = sorted(
            self.category_scores.values(),
            key=lambda x: x.score
        )

        # Recommendations for worst performers
        for score in sorted_categories[:5]:
            if score.grade in [LearningGrade.D, LearningGrade.F]:
                rec = self._create_recommendation(score)
                if rec:
                    recommendations.append(rec)

        # High recurrence but no lessons
        high_recurrence_no_lessons = [
            s for s in self.category_scores.values()
            if s.recurrence_rate > 40 and s.lessons_documented < 3
        ]
        for score in high_recurrence_no_lessons[:3]:
            recommendations.append({
                'category': score.entity,
                'priority': 'HIGH',
                'type': 'documentation_gap',
                'issue': f"{score.recurrence_rate:.0f}% recurrence with only {score.lessons_documented} lessons documented",
                'recommendation': f"Prioritize documenting lessons for {score.entity}. "
                                f"High recurrence ({score.recurrence_rate:.0f}%) suggests systemic issues "
                                f"that could be prevented with proper knowledge capture.",
                'impact': f"Could prevent ~{score.recurring_issues} recurring issues"
            })

        # Engineers with repeat patterns
        engineer_issues = [
            s for s in self.engineer_scores.values()
            if s.recurring_issues >= 5 and s.lesson_completion_rate < 30
        ]
        for score in engineer_issues[:3]:
            recommendations.append({
                'category': score.entity,
                'priority': 'MEDIUM',
                'type': 'training_opportunity',
                'issue': f"Engineer handling {score.recurring_issues} repeat issues with {score.lesson_completion_rate:.0f}% lesson completion",
                'recommendation': f"Consider targeted training for {score.entity}. "
                                f"They've encountered {score.recurring_issues} similar issues - "
                                f"creating knowledge base articles could help both them and the team.",
                'impact': f"Potential reduction in repeat issue handling"
            })

        # LOB/Market patterns
        lob_issues = [
            s for s in self.lob_scores.values()
            if s.recurrence_rate > 35 and s.grade in [LearningGrade.D, LearningGrade.F]
        ]
        for score in lob_issues[:2]:
            recommendations.append({
                'category': score.entity,
                'priority': 'MEDIUM',
                'type': 'regional_pattern',
                'issue': f"Market/LOB has {score.recurrence_rate:.0f}% recurrence rate (Grade {score.grade.value})",
                'recommendation': f"Investigate regional training gaps in {score.entity}. "
                                f"Higher than average recurrence may indicate localized knowledge gaps "
                                f"or process variations.",
                'impact': f"Could improve {score.entity} performance by addressing root causes"
            })

        return recommendations

    def _create_recommendation(self, score: LearningScore) -> Optional[Dict[str, str]]:
        """Create a specific recommendation for a learning score."""
        if score.grade == LearningGrade.F:
            return {
                'category': score.entity,
                'priority': 'CRITICAL',
                'type': 'failing_grade',
                'issue': f"Grade F - {score.recurrence_rate:.0f}% recurrence, {score.lesson_completion_rate:.0f}% lesson completion",
                'recommendation': f"URGENT: {score.entity} requires immediate attention. "
                                f"Create mandatory lesson documentation process. "
                                f"Consider root cause analysis workshop for this category.",
                'impact': f"Currently {score.ticket_count} tickets affected, {score.recurring_issues} are repeats"
            }
        elif score.grade == LearningGrade.D:
            return {
                'category': score.entity,
                'priority': 'HIGH',
                'type': 'poor_grade',
                'issue': f"Grade D - Needs improvement in lesson documentation and recurrence prevention",
                'recommendation': f"{score.entity} shows signs of inadequate learning capture. "
                                f"Assign dedicated owner for lesson documentation. "
                                f"Review similar historical tickets for patterns.",
                'impact': f"Improving to Grade C could prevent ~{int(score.recurring_issues * 0.3)} recurring issues"
            }
        return None

    def get_grade_summary_df(self) -> pd.DataFrame:
        """Get a summary DataFrame of all category grades."""
        rows = []
        for entity, score in self.category_scores.items():
            rows.append({
                'Category': entity,
                'Grade': score.grade.value,
                'Score': score.score,
                'Recurrence %': round(score.recurrence_rate, 1),
                'Lesson Completion %': round(score.lesson_completion_rate, 1),
                'Consistency %': round(score.resolution_consistency_rate, 1),
                'Tickets': score.ticket_count,
                'Lessons': score.lessons_documented,
                'Completed': score.lessons_completed,
                'Recurring': score.recurring_issues
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Score', ascending=True)
        return df

    def generate_ai_recommendations(self, use_ollama: bool = True) -> List[str]:
        """
        Generate AI-powered recommendations using local Ollama model.

        Args:
            use_ollama: If True, use Ollama for enhanced recommendations

        Returns:
            List of AI-generated recommendation strings
        """
        if not use_ollama:
            return [r['recommendation'] for r in self.recommendations]

        try:
            import requests

            # Build context for AI
            context = self._build_ai_context()

            prompt = f"""You are an expert in operational excellence and continuous improvement.
Analyze the following lessons learned effectiveness data and provide 3-5 specific, actionable recommendations.

{context}

Provide recommendations in this format:
1. [PRIORITY: HIGH/MEDIUM/LOW] Category/Area: Specific recommendation
   - Why: Brief explanation
   - Action: Concrete next step

Focus on:
- Categories with high recurrence but low lesson documentation
- Patterns that suggest systemic issues
- Quick wins that could have immediate impact
- Training or process gaps

Be specific and actionable. Reference actual category names and numbers from the data."""

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": GEN_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 800
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                ai_text = response.json().get("response", "").strip()
                if ai_text:
                    return [ai_text]

        except Exception as e:
            logger.warning(f"AI recommendation generation failed: {e}")

        # Fall back to rule-based recommendations
        return [r['recommendation'] for r in self.recommendations[:5]]

    def _build_ai_context(self) -> str:
        """Build context string for AI recommendation generation."""
        lines = [
            "=== LESSONS LEARNED EFFECTIVENESS ANALYSIS ===",
            f"Overall Grade: {self.overall_stats.get('overall_grade', 'N/A')}",
            f"Overall Score: {self.overall_stats.get('overall_score', 0):.0f}/100",
            f"Avg Recurrence Rate: {self.overall_stats.get('avg_recurrence_rate', 0):.1f}%",
            f"Lesson Documentation Rate: {self.overall_stats.get('lesson_documentation_rate', 0):.1f}%",
            f"Lesson Completion Rate: {self.overall_stats.get('lesson_completion_rate', 0):.1f}%",
            "",
            "=== CATEGORY GRADES (Worst to Best) ==="
        ]

        sorted_cats = sorted(self.category_scores.values(), key=lambda x: x.score)
        for score in sorted_cats[:10]:
            lines.append(
                f"- {score.entity}: Grade {score.grade.value} (Score: {score.score:.0f}, "
                f"Recurrence: {score.recurrence_rate:.0f}%, Lessons: {score.lessons_documented}/{score.ticket_count})"
            )

        lines.append("")
        lines.append("=== AT-RISK CATEGORIES ===")
        for cat in self.overall_stats.get('at_risk_categories', [])[:5]:
            lines.append(f"- {cat}")

        lines.append("")
        lines.append("=== KEY ISSUES ===")
        for rec in self.recommendations[:5]:
            lines.append(f"- {rec['category']}: {rec['issue']}")

        return "\n".join(lines)


# Global analyzer instance
_lessons_analyzer: Optional[LessonsLearnedAnalyzer] = None


def get_lessons_analyzer() -> LessonsLearnedAnalyzer:
    """Get or create the lessons learned analyzer singleton."""
    global _lessons_analyzer
    if _lessons_analyzer is None:
        _lessons_analyzer = LessonsLearnedAnalyzer()
    return _lessons_analyzer


def analyze_lessons_learned(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to run lessons learned analysis.

    Args:
        df: DataFrame with ticket data

    Returns:
        Analysis results dictionary
    """
    analyzer = get_lessons_analyzer()
    return analyzer.analyze(df)


# =============================================================================
# COMPREHENSIVE LEARNING EFFECTIVENESS SCORECARD
# =============================================================================

@dataclass
class PillarScore:
    """Score for a single pillar of the learning effectiveness scorecard."""
    name: str
    score: float  # 0-100
    weight: float  # Weight in overall calculation
    sub_scores: Dict[str, float]  # Individual factor scores
    insights: List[str]  # Key insights for this pillar
    trend: str  # 'improving', 'stable', 'degrading'


@dataclass
class CategoryScorecard:
    """Complete scorecard for a category."""
    category: str
    overall_score: float
    overall_grade: str
    pillars: Dict[str, PillarScore]
    rank: int  # Rank among all categories
    recommendations: List[str]
    strengths: List[str]
    weaknesses: List[str]


class LearningEffectivenessScorecard:
    """
    Comprehensive 6-pillar learning effectiveness scoring system.

    Pillars:
    1. Learning Velocity - Trend-based improvement metrics
    2. Impact Management - Severity/financial weighted scoring
    3. Knowledge Quality - AI-assessed lesson quality
    4. Process Maturity - Consistency & documentation
    5. Knowledge Transfer - Cross-team learning & silos
    6. Outcome Effectiveness - Resolution improvements

    Each pillar has multiple sub-factors and produces both a score (0-100)
    and actionable insights.
    """

    # Default pillar weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'learning_velocity': 0.20,
        'impact_management': 0.20,
        'knowledge_quality': 0.15,
        'process_maturity': 0.15,
        'knowledge_transfer': 0.15,
        'outcome_effectiveness': 0.15,
    }

    # Grade thresholds
    GRADE_THRESHOLDS = [
        (90, 'A+'), (85, 'A'), (80, 'A-'),
        (77, 'B+'), (73, 'B'), (70, 'B-'),
        (67, 'C+'), (63, 'C'), (60, 'C-'),
        (55, 'D+'), (50, 'D'), (45, 'D-'),
        (0, 'F')
    ]

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize scorecard with optional custom weights.

        Args:
            weights: Custom pillar weights (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
        self.category_scorecards: Dict[str, CategoryScorecard] = {}
        self.df: Optional[pd.DataFrame] = None
        self._column_cache: Dict[str, Optional[str]] = {}

    def _validate_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates with caching."""
        cache_key = tuple(candidates)
        if cache_key in self._column_cache:
            return self._column_cache[cache_key]

        for col in candidates:
            if col in df.columns:
                self._column_cache[cache_key] = col
                return col
        self._column_cache[cache_key] = None
        return None

    def score_category(self, df: pd.DataFrame, category: str) -> CategoryScorecard:
        """
        Calculate comprehensive scorecard for a single category.

        Args:
            df: Full DataFrame
            category: Category to score

        Returns:
            CategoryScorecard with all pillar scores
        """
        cat_df = df[df['AI_Category'] == category].copy()

        if len(cat_df) < 2:
            return self._empty_scorecard(category)

        pillars = {}

        # Score each pillar
        pillars['learning_velocity'] = self._score_learning_velocity(df, cat_df, category)
        pillars['impact_management'] = self._score_impact_management(df, cat_df, category)
        pillars['knowledge_quality'] = self._score_knowledge_quality(df, cat_df, category)
        pillars['process_maturity'] = self._score_process_maturity(df, cat_df, category)
        pillars['knowledge_transfer'] = self._score_knowledge_transfer(df, cat_df, category)
        pillars['outcome_effectiveness'] = self._score_outcome_effectiveness(df, cat_df, category)

        # Calculate overall score
        overall_score = sum(
            pillars[name].score * self.weights[name]
            for name in pillars
        )

        # Determine grade
        overall_grade = self._score_to_grade(overall_score)

        # Identify strengths and weaknesses
        sorted_pillars = sorted(pillars.items(), key=lambda x: x[1].score, reverse=True)
        strengths = [f"{p[0].replace('_', ' ').title()}: {p[1].score:.0f}"
                    for p in sorted_pillars[:2] if p[1].score >= 70]
        weaknesses = [f"{p[0].replace('_', ' ').title()}: {p[1].score:.0f}"
                     for p in sorted_pillars[-2:] if p[1].score < 50]

        # Compile recommendations from all pillars
        recommendations = []
        for pillar in pillars.values():
            recommendations.extend(pillar.insights[:2])  # Top 2 insights per pillar

        return CategoryScorecard(
            category=category,
            overall_score=overall_score,
            overall_grade=overall_grade,
            pillars=pillars,
            rank=0,  # Set later when all categories scored
            recommendations=recommendations[:10],
            strengths=strengths,
            weaknesses=weaknesses
        )

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        for threshold, grade in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return 'F'

    def _empty_scorecard(self, category: str) -> CategoryScorecard:
        """Return empty scorecard for categories with insufficient data."""
        empty_pillar = PillarScore(
            name="N/A", score=0, weight=0,
            sub_scores={}, insights=["Insufficient data"], trend="unknown"
        )
        return CategoryScorecard(
            category=category,
            overall_score=0,
            overall_grade="N/A",
            pillars={name: empty_pillar for name in self.weights},
            rank=0,
            recommendations=["Insufficient data for scoring"],
            strengths=[],
            weaknesses=[]
        )

    # =========================================================================
    # PILLAR 1: LEARNING VELOCITY (Trend-based)
    # =========================================================================

    def _score_learning_velocity(self, df: pd.DataFrame, cat_df: pd.DataFrame,
                                  category: str) -> PillarScore:
        """
        Score based on improvement trends over time.

        Sub-factors:
        - Recurrence trend (is recurrence rate decreasing?)
        - Incident frequency trend (are incidents becoming less frequent?)
        - Lesson creation velocity (are lessons being created proactively?)
        - Time-to-lesson (how quickly are lessons documented after incidents?)
        """
        sub_scores = {}
        insights = []

        # Find date column
        date_col = self._find_column(df, [
            'tickets_data_creation_date', 'Created', 'Creation_Date', 'Date'
        ])

        # 1. Recurrence Trend (is recurrence decreasing over time?)
        recurrence_trend_score = 50  # Default neutral
        if 'AI_Recurrence_Probability' in cat_df.columns and date_col:
            try:
                cat_df_sorted = cat_df.sort_values(date_col)
                if len(cat_df_sorted) >= 4:
                    mid = len(cat_df_sorted) // 2
                    first_half_avg = cat_df_sorted.iloc[:mid]['AI_Recurrence_Probability'].mean()
                    second_half_avg = cat_df_sorted.iloc[mid:]['AI_Recurrence_Probability'].mean()

                    # Improvement = recurrence going down
                    improvement = (first_half_avg - second_half_avg) / max(first_half_avg, 0.01)
                    recurrence_trend_score = min(100, max(0, 50 + improvement * 100))

                    if improvement > 0.1:
                        insights.append(f"✅ Recurrence rate improving ({improvement*100:.0f}% reduction)")
                    elif improvement < -0.1:
                        insights.append(f"⚠️ Recurrence rate worsening ({-improvement*100:.0f}% increase)")
            except Exception:
                pass
        sub_scores['recurrence_trend'] = recurrence_trend_score

        # 2. Incident Frequency Trend
        frequency_trend_score = 50
        if date_col:
            try:
                cat_df_dated = cat_df[cat_df[date_col].notna()].copy()
                cat_df_dated[date_col] = pd.to_datetime(cat_df_dated[date_col], errors='coerce')
                cat_df_dated = cat_df_dated[cat_df_dated[date_col].notna()]

                if len(cat_df_dated) >= 4:
                    cat_df_dated['month'] = cat_df_dated[date_col].dt.to_period('M')
                    monthly_counts = cat_df_dated.groupby('month').size()

                    if len(monthly_counts) >= 2:
                        mid = len(monthly_counts) // 2
                        first_half_avg = monthly_counts.iloc[:mid].mean()
                        second_half_avg = monthly_counts.iloc[mid:].mean()

                        # Improvement = fewer incidents
                        if first_half_avg > 0:
                            improvement = (first_half_avg - second_half_avg) / first_half_avg
                            frequency_trend_score = min(100, max(0, 50 + improvement * 50))
            except Exception:
                pass
        sub_scores['frequency_trend'] = frequency_trend_score

        # 3. Lesson Creation Velocity
        lesson_velocity_score = 50
        lesson_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title', 'Lesson_Title'])
        if lesson_col and date_col:
            try:
                has_lesson = cat_df[lesson_col].notna()
                total = len(cat_df)
                with_lesson = has_lesson.sum()

                # Score based on percentage with lessons
                lesson_velocity_score = min(100, (with_lesson / max(total, 1)) * 120)

                if with_lesson / max(total, 1) < 0.3:
                    insights.append(f"📝 Low lesson documentation rate ({with_lesson}/{total} tickets)")
            except Exception:
                pass
        sub_scores['lesson_velocity'] = lesson_velocity_score

        # 4. Time-to-Lesson (placeholder - would need lesson creation date)
        sub_scores['time_to_lesson'] = 50  # Neutral default

        # Calculate pillar score
        pillar_score = np.mean(list(sub_scores.values()))

        # Determine trend
        trend_scores = [sub_scores['recurrence_trend'], sub_scores['frequency_trend']]
        avg_trend = np.mean(trend_scores)
        trend = 'improving' if avg_trend > 60 else ('degrading' if avg_trend < 40 else 'stable')

        return PillarScore(
            name='Learning Velocity',
            score=pillar_score,
            weight=self.weights['learning_velocity'],
            sub_scores=sub_scores,
            insights=insights,
            trend=trend
        )

    # =========================================================================
    # PILLAR 2: IMPACT MANAGEMENT (Severity/Financial weighted)
    # =========================================================================

    def _score_impact_management(self, df: pd.DataFrame, cat_df: pd.DataFrame,
                                  category: str) -> PillarScore:
        """
        Score based on how well high-impact issues are being addressed.

        Sub-factors:
        - Severity-weighted recurrence (high severity repeats penalized more)
        - Financial impact of repeats
        - SLA breach rate for recurring issues
        - Priority alignment (high priority = faster lessons)
        """
        sub_scores = {}
        insights = []

        # 1. Severity-Weighted Recurrence
        severity_score = 50
        severity_col = self._find_column(df, [
            'tickets_data_priority', 'Priority', 'Severity', 'tickets_data_severity'
        ])

        if severity_col and 'AI_Recurrence_Probability' in cat_df.columns:
            try:
                # Map severity to weights
                severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1,
                                   '1': 4, '2': 3, '3': 2, '4': 1, '5': 1}

                cat_df_sev = cat_df.copy()
                cat_df_sev['sev_weight'] = cat_df_sev[severity_col].str.lower().map(
                    lambda x: severity_weights.get(str(x).lower(), 2)
                )

                # Weighted recurrence
                weighted_recurrence = (
                    cat_df_sev['AI_Recurrence_Probability'] * cat_df_sev['sev_weight']
                ).mean() / 2.5  # Normalize

                severity_score = max(0, 100 - weighted_recurrence * 100)

                high_sev_recurring = cat_df_sev[
                    (cat_df_sev['sev_weight'] >= 3) &
                    (cat_df_sev['AI_Recurrence_Probability'] > 0.5)
                ]
                if len(high_sev_recurring) > 0:
                    insights.append(f"🔴 {len(high_sev_recurring)} high-severity recurring issues need attention")
            except Exception:
                pass
        sub_scores['severity_weighted'] = severity_score

        # 2. Financial Impact
        financial_score = 50
        cost_col = self._find_column(df, [
            'Estimated_Cost', 'tickets_data_estimated_cost', 'Cost', 'Financial_Impact'
        ])

        if cost_col and 'Similar_Ticket_Count' in cat_df.columns:
            try:
                recurring = cat_df[cat_df['Similar_Ticket_Count'] > 0]
                non_recurring = cat_df[cat_df['Similar_Ticket_Count'] == 0]

                recurring_cost = recurring[cost_col].sum() if len(recurring) > 0 else 0
                total_cost = cat_df[cost_col].sum()

                if total_cost > 0:
                    repeat_cost_pct = recurring_cost / total_cost
                    financial_score = max(0, 100 - repeat_cost_pct * 100)

                    if repeat_cost_pct > 0.5:
                        insights.append(f"💰 {repeat_cost_pct*100:.0f}% of costs from recurring issues")
            except Exception:
                pass
        sub_scores['financial_impact'] = financial_score

        # 3. SLA Breach Rate
        sla_score = 50
        sla_col = self._find_column(df, [
            'SLA_Breached', 'tickets_data_sla_breached', 'SLA_Status', 'Within_SLA'
        ])

        if sla_col:
            try:
                breached = cat_df[sla_col].str.lower().str.contains('breach|no|fail', na=False).sum()
                total = len(cat_df)
                breach_rate = breached / max(total, 1)
                sla_score = max(0, 100 - breach_rate * 150)  # Penalize breaches heavily

                if breach_rate > 0.2:
                    insights.append(f"⏰ {breach_rate*100:.0f}% SLA breach rate")
            except Exception:
                pass
        sub_scores['sla_performance'] = sla_score

        # 4. Priority Alignment (high priority should have lessons faster)
        priority_score = 50
        lesson_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title'])

        if severity_col and lesson_col:
            try:
                high_priority = cat_df[cat_df[severity_col].str.lower().isin(
                    ['critical', 'high', '1', '2']
                )]
                if len(high_priority) > 0:
                    hp_with_lessons = high_priority[lesson_col].notna().sum()
                    priority_score = min(100, (hp_with_lessons / len(high_priority)) * 120)

                    if hp_with_lessons / len(high_priority) < 0.5:
                        insights.append(f"⚡ Only {hp_with_lessons}/{len(high_priority)} high-priority issues have lessons")
            except Exception:
                pass
        sub_scores['priority_alignment'] = priority_score

        pillar_score = np.mean(list(sub_scores.values()))
        trend = 'stable'  # Would need historical data for trend

        return PillarScore(
            name='Impact Management',
            score=pillar_score,
            weight=self.weights['impact_management'],
            sub_scores=sub_scores,
            insights=insights,
            trend=trend
        )

    # =========================================================================
    # PILLAR 3: KNOWLEDGE QUALITY (AI-assessed)
    # =========================================================================

    def _score_knowledge_quality(self, df: pd.DataFrame, cat_df: pd.DataFrame,
                                  category: str) -> PillarScore:
        """
        Score based on quality of documented lessons (AI-assessed).

        Sub-factors:
        - Lesson actionability (does it specify what to do?)
        - Lesson specificity (generic vs detailed)
        - Root cause documentation quality
        - Lesson-to-resolution linkage
        """
        sub_scores = {}
        insights = []

        lesson_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title', 'Lesson_Title'])
        root_cause_col = self._find_column(df, [
            'tickets_data_root_cause', 'Root_Cause', 'AI_Root_Cause', 'root_cause'
        ])

        # 1. Lesson Actionability (keyword-based heuristic)
        actionability_score = 50
        if lesson_col:
            lessons = cat_df[lesson_col].dropna()
            if len(lessons) > 0:
                action_keywords = ['implement', 'create', 'update', 'change', 'add',
                                  'remove', 'configure', 'ensure', 'verify', 'check',
                                  'monitor', 'automate', 'prevent', 'fix', 'resolve']

                actionable_count = sum(
                    1 for lesson in lessons
                    if any(kw in str(lesson).lower() for kw in action_keywords)
                )
                actionability_score = min(100, (actionable_count / len(lessons)) * 110)

                if actionability_score < 50:
                    insights.append("📋 Lessons lack actionable language - make them more specific")
        sub_scores['actionability'] = actionability_score

        # 2. Lesson Specificity (length and detail heuristic)
        specificity_score = 50
        if lesson_col:
            lessons = cat_df[lesson_col].dropna()
            if len(lessons) > 0:
                avg_length = lessons.str.len().mean()
                # Ideal length: 50-200 chars (not too short, not too long)
                if avg_length < 20:
                    specificity_score = 30
                    insights.append("📝 Lessons too brief - add more detail")
                elif avg_length < 50:
                    specificity_score = 60
                elif avg_length < 200:
                    specificity_score = 90
                else:
                    specificity_score = 70  # Too long might be unfocused
        sub_scores['specificity'] = specificity_score

        # 3. Root Cause Documentation
        root_cause_score = 50
        if root_cause_col:
            root_causes = cat_df[root_cause_col].dropna()
            total = len(cat_df)

            if total > 0:
                documented = len(root_causes)
                root_cause_score = min(100, (documented / total) * 110)

                # Check for generic root causes
                generic_patterns = ['unknown', 'other', 'misc', 'n/a', 'tbd', 'none']
                generic_count = sum(
                    1 for rc in root_causes
                    if any(p in str(rc).lower() for p in generic_patterns)
                )
                if generic_count > 0:
                    root_cause_score = max(0, root_cause_score - generic_count / total * 50)
                    insights.append(f"🔍 {generic_count} generic root causes - need more analysis")
        sub_scores['root_cause_quality'] = root_cause_score

        # 4. Lesson-Resolution Linkage
        linkage_score = 50
        resolution_col = self._find_column(df, [
            'Resolution_Recommendation', 'tickets_data_resolution', 'Resolution'
        ])

        if lesson_col and resolution_col:
            # Check if lessons reference resolution approaches
            lessons = cat_df[lesson_col].dropna()
            resolutions = cat_df[resolution_col].dropna()

            if len(lessons) > 0 and len(resolutions) > 0:
                # Simple overlap check
                linked = 0
                for idx in cat_df.index:
                    lesson = str(cat_df.loc[idx, lesson_col] if pd.notna(cat_df.loc[idx, lesson_col]) else '')
                    resolution = str(cat_df.loc[idx, resolution_col] if pd.notna(cat_df.loc[idx, resolution_col]) else '')

                    if lesson and resolution:
                        # Check for word overlap
                        lesson_words = set(lesson.lower().split())
                        resolution_words = set(resolution.lower().split())
                        if len(lesson_words & resolution_words) >= 2:
                            linked += 1

                linkage_score = min(100, (linked / len(lessons)) * 100)
        sub_scores['resolution_linkage'] = linkage_score

        pillar_score = np.mean(list(sub_scores.values()))
        trend = 'stable'

        return PillarScore(
            name='Knowledge Quality',
            score=pillar_score,
            weight=self.weights['knowledge_quality'],
            sub_scores=sub_scores,
            insights=insights,
            trend=trend
        )

    # =========================================================================
    # PILLAR 4: PROCESS MATURITY (Consistency & Documentation)
    # =========================================================================

    def _score_process_maturity(self, df: pd.DataFrame, cat_df: pd.DataFrame,
                                 category: str) -> PillarScore:
        """
        Score based on process consistency and documentation completeness.

        Sub-factors:
        - Resolution consistency rate
        - Documentation completeness
        - Standard approach adherence
        - Lesson completion rate
        """
        sub_scores = {}
        insights = []

        # 1. Resolution Consistency
        consistency_score = 50
        if 'Resolution_Consistency' in cat_df.columns:
            consistent = cat_df['Resolution_Consistency'].str.contains(
                'consistent|Consistent', na=False
            ).sum()
            total = len(cat_df)
            consistency_score = min(100, (consistent / max(total, 1)) * 110)

            inconsistent = total - consistent
            if inconsistent > total * 0.3:
                insights.append(f"⚖️ {inconsistent}/{total} tickets have inconsistent resolutions")
        sub_scores['resolution_consistency'] = consistency_score

        # 2. Documentation Completeness
        doc_score = 50
        doc_fields = [
            self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title']),
            self._find_column(df, ['tickets_data_root_cause', 'Root_Cause']),
            self._find_column(df, ['tickets_data_resolution', 'Resolution']),
        ]
        doc_fields = [f for f in doc_fields if f]

        if doc_fields:
            completeness_scores = []
            for field in doc_fields:
                if field in cat_df.columns:
                    complete = cat_df[field].notna().sum()
                    completeness_scores.append(complete / len(cat_df))

            if completeness_scores:
                doc_score = np.mean(completeness_scores) * 100

                if doc_score < 50:
                    insights.append("📄 Documentation incomplete - enforce required fields")
        sub_scores['documentation_completeness'] = doc_score

        # 3. Standard Approach Adherence
        adherence_score = 50
        if 'Similar_Ticket_Count' in cat_df.columns and 'Resolution_Consistency' in cat_df.columns:
            # Tickets with similar matches should have consistent resolutions
            has_similar = cat_df[cat_df['Similar_Ticket_Count'] > 0]
            if len(has_similar) > 0:
                following_standard = has_similar['Resolution_Consistency'].str.contains(
                    'consistent|Consistent', na=False
                ).sum()
                adherence_score = (following_standard / len(has_similar)) * 100
        sub_scores['standard_adherence'] = adherence_score

        # 4. Lesson Completion Rate
        completion_score = 50
        lesson_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title'])
        status_col = self._find_column(df, [COL_LESSON_STATUS, 'lessons_learned_status'])

        if lesson_col and status_col:
            documented = cat_df[lesson_col].notna().sum()
            if documented > 0:
                completed = cat_df[status_col].str.lower().str.contains(
                    'complete|done|closed', na=False
                ).sum()
                completion_score = (completed / documented) * 100

                if completion_score < 50:
                    insights.append(f"✅ Only {completed}/{documented} lessons marked complete")
        sub_scores['lesson_completion'] = completion_score

        pillar_score = np.mean(list(sub_scores.values()))
        trend = 'stable'

        return PillarScore(
            name='Process Maturity',
            score=pillar_score,
            weight=self.weights['process_maturity'],
            sub_scores=sub_scores,
            insights=insights,
            trend=trend
        )

    # =========================================================================
    # PILLAR 5: KNOWLEDGE TRANSFER (Cross-team learning)
    # =========================================================================

    def _score_knowledge_transfer(self, df: pd.DataFrame, cat_df: pd.DataFrame,
                                   category: str) -> PillarScore:
        """
        Score based on how well knowledge is being shared.

        Sub-factors:
        - Cross-team lesson application
        - Engineer improvement rate
        - Knowledge silo detection (inverse)
        - Lesson reuse rate
        """
        sub_scores = {}
        insights = []

        engineer_col = self._find_column(df, [COL_ENGINEER, 'Engineer', 'engineer_name'])
        lob_col = self._find_column(df, [COL_LOB, 'LOB', 'tickets_data_market'])

        # 1. Cross-LOB Consistency (same category resolved similarly across LOBs)
        cross_lob_score = 50
        if lob_col and 'Resolution_Consistency' in cat_df.columns:
            lob_consistency = {}
            for lob in cat_df[lob_col].dropna().unique():
                lob_df = cat_df[cat_df[lob_col] == lob]
                if len(lob_df) >= 2:
                    consistent = lob_df['Resolution_Consistency'].str.contains(
                        'consistent|Consistent', na=False
                    ).mean()
                    lob_consistency[lob] = consistent

            if len(lob_consistency) >= 2:
                variance = np.std(list(lob_consistency.values()))
                # Low variance = good cross-LOB consistency
                cross_lob_score = max(0, 100 - variance * 200)

                if variance > 0.3:
                    insights.append("🌐 Inconsistent practices across LOBs - share best practices")
        sub_scores['cross_lob_consistency'] = cross_lob_score

        # 2. Engineer Improvement (same engineer's tickets improving over time)
        engineer_improvement_score = 50
        if engineer_col and 'AI_Recurrence_Probability' in cat_df.columns:
            date_col = self._find_column(df, ['tickets_data_creation_date', 'Created', 'Date'])
            if date_col:
                try:
                    engineer_trends = []
                    for eng in cat_df[engineer_col].dropna().unique():
                        eng_df = cat_df[cat_df[engineer_col] == eng].sort_values(date_col)
                        if len(eng_df) >= 3:
                            mid = len(eng_df) // 2
                            first_avg = eng_df.iloc[:mid]['AI_Recurrence_Probability'].mean()
                            second_avg = eng_df.iloc[mid:]['AI_Recurrence_Probability'].mean()
                            improvement = first_avg - second_avg
                            engineer_trends.append(improvement)

                    if engineer_trends:
                        avg_improvement = np.mean(engineer_trends)
                        engineer_improvement_score = min(100, max(0, 50 + avg_improvement * 100))
                except Exception:
                    pass
        sub_scores['engineer_improvement'] = engineer_improvement_score

        # 3. Knowledge Silo Detection (is one engineer/team hoarding knowledge?)
        silo_score = 50
        if engineer_col:
            lesson_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title'])
            if lesson_col:
                engineers_with_lessons = cat_df[cat_df[lesson_col].notna()][engineer_col].value_counts()
                total_engineers = cat_df[engineer_col].nunique()

                if total_engineers > 1 and len(engineers_with_lessons) > 0:
                    # Check if lessons are concentrated with few engineers
                    top_engineer_pct = engineers_with_lessons.iloc[0] / max(engineers_with_lessons.sum(), 1)
                    engineer_coverage = len(engineers_with_lessons) / total_engineers

                    # High coverage, low concentration = good
                    silo_score = (engineer_coverage * 60) + ((1 - top_engineer_pct) * 40)

                    if engineer_coverage < 0.3:
                        insights.append(f"🔒 Only {len(engineers_with_lessons)}/{total_engineers} engineers documenting lessons")
        sub_scores['silo_avoidance'] = silo_score

        # 4. Lesson Reuse (similar tickets referencing same lessons)
        reuse_score = 50
        if 'Similar_Ticket_IDs' in cat_df.columns:
            lesson_col = self._find_column(df, [COL_LESSON_TITLE, 'lessons_learned_title'])
            if lesson_col:
                has_similar = cat_df[cat_df['Similar_Ticket_IDs'].notna() & (cat_df['Similar_Ticket_IDs'] != '')]
                has_lesson = cat_df[cat_df[lesson_col].notna()]

                # Tickets with similar matches should be learning from them
                if len(has_similar) > 0:
                    learning = len(set(has_similar.index) & set(has_lesson.index))
                    reuse_score = (learning / len(has_similar)) * 100
        sub_scores['lesson_reuse'] = reuse_score

        pillar_score = np.mean(list(sub_scores.values()))
        trend = 'stable'

        return PillarScore(
            name='Knowledge Transfer',
            score=pillar_score,
            weight=self.weights['knowledge_transfer'],
            sub_scores=sub_scores,
            insights=insights,
            trend=trend
        )

    # =========================================================================
    # PILLAR 6: OUTCOME EFFECTIVENESS (Results)
    # =========================================================================

    def _score_outcome_effectiveness(self, df: pd.DataFrame, cat_df: pd.DataFrame,
                                      category: str) -> PillarScore:
        """
        Score based on actual outcomes and improvements.

        Sub-factors:
        - Resolution time improvement
        - First-time resolution rate
        - Recurrence reduction
        - Customer satisfaction (if available)
        """
        sub_scores = {}
        insights = []

        date_col = self._find_column(df, ['tickets_data_creation_date', 'Created', 'Date'])

        # 1. Resolution Time Improvement
        res_time_score = 50
        res_time_col = self._find_column(df, [
            'Predicted_Resolution_Days', 'Resolution_Days', 'Days_To_Resolution',
            'tickets_data_resolution_days'
        ])

        if res_time_col and date_col:
            try:
                cat_df_sorted = cat_df.sort_values(date_col)
                res_times = cat_df_sorted[res_time_col].dropna()

                if len(res_times) >= 4:
                    mid = len(res_times) // 2
                    first_avg = res_times.iloc[:mid].mean()
                    second_avg = res_times.iloc[mid:].mean()

                    if first_avg > 0:
                        improvement = (first_avg - second_avg) / first_avg
                        res_time_score = min(100, max(0, 50 + improvement * 100))

                        if improvement > 0.2:
                            insights.append(f"⏱️ Resolution time improved by {improvement*100:.0f}%")
                        elif improvement < -0.2:
                            insights.append(f"⚠️ Resolution time worsened by {-improvement*100:.0f}%")
            except Exception:
                pass
        sub_scores['resolution_time'] = res_time_score

        # 2. First-Time Resolution Rate (inverse of recurrence)
        ftr_score = 50
        if 'AI_Recurrence_Probability' in cat_df.columns:
            avg_recurrence = cat_df['AI_Recurrence_Probability'].mean()
            ftr_score = max(0, (1 - avg_recurrence) * 100)
        elif 'Similar_Ticket_Count' in cat_df.columns:
            non_recurring = (cat_df['Similar_Ticket_Count'] == 0).sum()
            ftr_score = (non_recurring / len(cat_df)) * 100
        sub_scores['first_time_resolution'] = ftr_score

        # 3. Recurrence Reduction Over Time
        recurrence_reduction_score = 50
        if 'AI_Recurrence_Probability' in cat_df.columns and date_col:
            try:
                cat_df_sorted = cat_df.sort_values(date_col)
                if len(cat_df_sorted) >= 4:
                    mid = len(cat_df_sorted) // 2
                    first_avg = cat_df_sorted.iloc[:mid]['AI_Recurrence_Probability'].mean()
                    second_avg = cat_df_sorted.iloc[mid:]['AI_Recurrence_Probability'].mean()

                    if first_avg > 0:
                        reduction = (first_avg - second_avg) / first_avg
                        recurrence_reduction_score = min(100, max(0, 50 + reduction * 100))
            except Exception:
                pass
        sub_scores['recurrence_reduction'] = recurrence_reduction_score

        # 4. Expected vs Actual (if we have both)
        accuracy_score = 50
        if 'Expected_Resolution_Days' in cat_df.columns and res_time_col:
            try:
                expected = cat_df['Expected_Resolution_Days'].dropna()
                actual = cat_df[res_time_col].dropna()

                if len(expected) > 0 and len(actual) > 0:
                    # Get matching indices
                    common_idx = expected.index.intersection(actual.index)
                    if len(common_idx) > 0:
                        mae = np.abs(expected.loc[common_idx] - actual.loc[common_idx]).mean()
                        # Lower MAE = better accuracy = higher score
                        accuracy_score = max(0, 100 - mae * 10)
            except Exception:
                pass
        sub_scores['prediction_accuracy'] = accuracy_score

        pillar_score = np.mean(list(sub_scores.values()))

        # Determine trend based on outcome metrics
        trend_scores = [sub_scores['resolution_time'], sub_scores['recurrence_reduction']]
        avg_trend = np.mean(trend_scores)
        trend = 'improving' if avg_trend > 60 else ('degrading' if avg_trend < 40 else 'stable')

        return PillarScore(
            name='Outcome Effectiveness',
            score=pillar_score,
            weight=self.weights['outcome_effectiveness'],
            sub_scores=sub_scores,
            insights=insights,
            trend=trend
        )

    # =========================================================================
    # MAIN ANALYSIS METHODS
    # =========================================================================

    def analyze(self, df: pd.DataFrame) -> Dict[str, CategoryScorecard]:
        """
        Analyze all categories and produce comprehensive scorecards.

        Args:
            df: DataFrame with ticket data

        Returns:
            Dict mapping category names to their scorecards
        """
        self.df = df
        self._column_cache = {}  # Clear cache

        if 'AI_Category' not in df.columns:
            logger.warning("AI_Category column not found - cannot score categories")
            return {}

        categories = df['AI_Category'].dropna().unique()
        logger.info(f"Analyzing {len(categories)} categories with comprehensive scorecard...")

        # Score each category
        for category in categories:
            self.category_scorecards[category] = self.score_category(df, category)

        # Assign ranks
        sorted_cats = sorted(
            self.category_scorecards.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        for rank, (cat, scorecard) in enumerate(sorted_cats, 1):
            scorecard.rank = rank

        logger.info(f"Scorecard analysis complete: {len(self.category_scorecards)} categories scored")
        return self.category_scorecards

    def get_summary_df(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all category scores.

        Returns:
            DataFrame with category scores, grades, and pillar breakdowns
        """
        rows = []
        for cat, scorecard in sorted(
            self.category_scorecards.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        ):
            row = {
                'Category': cat,
                'Rank': scorecard.rank,
                'Overall Score': scorecard.overall_score,
                'Grade': scorecard.overall_grade,
            }

            # Add pillar scores
            for pillar_name, pillar in scorecard.pillars.items():
                readable_name = pillar_name.replace('_', ' ').title()
                row[readable_name] = pillar.score

            rows.append(row)

        return pd.DataFrame(rows)

    def get_at_risk_categories(self, threshold_grade: str = 'C-') -> List[CategoryScorecard]:
        """Get categories below a threshold grade."""
        grade_order = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
        threshold_idx = grade_order.index(threshold_grade)

        at_risk = []
        for scorecard in self.category_scorecards.values():
            if scorecard.overall_grade in grade_order:
                grade_idx = grade_order.index(scorecard.overall_grade)
                if grade_idx > threshold_idx:
                    at_risk.append(scorecard)

        return sorted(at_risk, key=lambda x: x.overall_score)

    def get_top_recommendations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top recommendations across all categories."""
        all_recs = []

        for cat, scorecard in self.category_scorecards.items():
            for rec in scorecard.recommendations[:3]:
                all_recs.append({
                    'category': cat,
                    'grade': scorecard.overall_grade,
                    'score': scorecard.overall_score,
                    'recommendation': rec
                })

        # Prioritize by grade (worse grades first)
        grade_priority = {'F': 0, 'D-': 1, 'D': 2, 'D+': 3, 'C-': 4, 'C': 5, 'C+': 6}
        all_recs.sort(key=lambda x: (grade_priority.get(x['grade'], 10), x['score']))

        return all_recs[:n]

    def generate_ai_summary(self, use_ollama: bool = True) -> str:
        """
        Generate AI-powered executive summary of the scorecard analysis.

        Args:
            use_ollama: Whether to use Ollama for AI generation

        Returns:
            Executive summary string
        """
        if not self.category_scorecards:
            return "No categories analyzed yet."

        # Build context
        summary_df = self.get_summary_df()
        at_risk = self.get_at_risk_categories()
        top_recs = self.get_top_recommendations(5)

        context = f"""
Learning Effectiveness Scorecard Analysis Summary:

Total Categories: {len(self.category_scorecards)}
Average Score: {summary_df['Overall Score'].mean():.1f}/100

Grade Distribution:
{summary_df['Grade'].value_counts().to_string()}

Top 5 Performing Categories:
{summary_df.head(5)[['Category', 'Grade', 'Overall Score']].to_string(index=False)}

At-Risk Categories (Below C-): {len(at_risk)}
{', '.join([f"{s.category} ({s.overall_grade})" for s in at_risk[:5]])}

Key Issues:
{chr(10).join([f"- {r['category']}: {r['recommendation']}" for r in top_recs[:5]])}
"""

        if not use_ollama:
            return context

        try:
            import requests

            prompt = f"""You are a management consultant analyzing organizational learning effectiveness.
Based on this scorecard data, provide a brief executive summary (3-4 paragraphs):

{context}

Include:
1. Overall assessment of organizational learning health
2. Key patterns or concerns identified
3. Top 3 strategic recommendations with expected impact
4. Quick wins that can be implemented immediately

Be specific and actionable. Use business language."""

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": GEN_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 800}
                },
                timeout=60
            )

            if response.status_code == 200:
                ai_text = response.json().get("response", "").strip()
                if ai_text:
                    return ai_text
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")

        return context


# Convenience function
def create_scorecard(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> LearningEffectivenessScorecard:
    """
    Create and run a comprehensive learning effectiveness scorecard analysis.

    Args:
        df: DataFrame with ticket data
        weights: Optional custom pillar weights

    Returns:
        Populated LearningEffectivenessScorecard instance
    """
    scorecard = LearningEffectivenessScorecard(weights)
    scorecard.analyze(df)
    return scorecard
