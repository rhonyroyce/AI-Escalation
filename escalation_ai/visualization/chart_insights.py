"""
Vision-based Chart Insight Analyzer.

Uses local vision models (Ollama) to analyze chart images and generate
deep, data-driven insights for the Dashboard.

Recommended model: llama3.2-vision:latest
- Consistently generates accurate, data-specific insights
- Works reliably with Ollama vision API
- Average response time: 3-20 seconds depending on chart complexity

Note: qwen3-vl models may have compatibility issues with Ollama image API
"""

import os
import base64
import logging
import json
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.config import VISION_MODEL, VISION_MODEL_TIMEOUT, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class ChartInsightAnalyzer:
    """
    Analyzes chart images using vision models to generate insights.

    Uses local Ollama models (llama3.2-vision, qwen3-vl) to:
    - Identify patterns and trends visible in charts
    - Detect anomalies and outliers
    - Generate actionable recommendations
    """

    def __init__(self, model: str = None, base_url: str = None, timeout: int = None):
        """
        Initialize the chart insight analyzer.

        Args:
            model: Ollama vision model to use (default from config: llama3.2-vision:latest)
            base_url: Ollama API base URL (default from config)
            timeout: Request timeout in seconds (default from config: 120)
        """
        self.model = model or VISION_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.timeout = timeout or VISION_MODEL_TIMEOUT
        self.insights_cache: Dict[str, Dict] = {}

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_ollama_vision(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Call Ollama vision API with an image.

        Args:
            image_path: Path to the chart image
            prompt: Analysis prompt

        Returns:
            Model response text or None on error
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library required for Ollama API calls")
            return None

        try:
            # Encode image
            image_b64 = self._encode_image(image_path)

            # Ollama API payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused analysis
                    "num_predict": 500,  # Limit response length
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.warning(f"Ollama API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to Ollama. Is it running?")
            return None
        except Exception as e:
            logger.warning(f"Vision analysis failed for {image_path}: {e}")
            return None

    def analyze_chart(self, chart_path: str, chart_context: str = "") -> Dict[str, Any]:
        """
        Analyze a single chart image and generate insights.

        Args:
            chart_path: Path to the chart image file
            chart_context: Optional context about what the chart shows

        Returns:
            Dict with keys: summary, key_insight, trend, recommendation
        """
        # Check cache first
        if chart_path in self.insights_cache:
            return self.insights_cache[chart_path]

        # Build analysis prompt
        prompt = f"""Analyze this chart image and provide insights.

{f"Context: {chart_context}" if chart_context else ""}

Provide a concise analysis in this exact format:
SUMMARY: [One sentence describing what the chart shows]
KEY_INSIGHT: [The most important pattern or finding visible in the chart]
TREND: [Any trends - increasing, decreasing, stable, or notable patterns]
RECOMMENDATION: [One actionable recommendation based on what you see]

Be specific about numbers, categories, or values you can see in the chart. Keep each section to 1-2 sentences max."""

        response = self._call_ollama_vision(chart_path, prompt)

        if not response:
            # Return default insight if vision analysis fails
            return self._get_default_insight(chart_path)

        # Parse the response
        insight = self._parse_insight_response(response, chart_path)

        # Cache the result
        self.insights_cache[chart_path] = insight

        return insight

    def _parse_insight_response(self, response: str, chart_path: str) -> Dict[str, Any]:
        """Parse the model response into structured insight."""
        import re

        insight = {
            "summary": "",
            "key_insight": "",
            "trend": "",
            "recommendation": "",
            "raw_response": response,
            "source": "vision_model"
        }

        # Clean markdown formatting (remove ** bold markers)
        clean_response = re.sub(r'\*\*', '', response)

        # Parse each section
        lines = clean_response.split("\n")
        current_key = None

        for line in lines:
            line = line.strip()
            line_upper = line.upper()

            # Check for section headers (with or without colon)
            if line_upper.startswith("SUMMARY"):
                current_key = "summary"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif "KEY_INSIGHT" in line_upper or "KEY INSIGHT" in line_upper:
                current_key = "key_insight"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_upper.startswith("TREND"):
                current_key = "trend"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_upper.startswith("RECOMMENDATION"):
                current_key = "recommendation"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif current_key and line:
                # Continue previous section
                insight[current_key] += " " + line

        # Clean up and limit length
        for key in ["summary", "key_insight", "trend", "recommendation"]:
            insight[key] = insight[key].strip()[:300]

        return insight

    def _get_default_insight(self, chart_path: str) -> Dict[str, Any]:
        """Get default insight when vision analysis is unavailable."""
        filename = os.path.basename(chart_path).lower()

        # Map chart types to default insights
        defaults = {
            "friction_by_category": {
                "summary": "Shows friction scores distributed across issue categories.",
                "key_insight": "Higher friction categories indicate areas requiring process improvement.",
                "trend": "Review top categories for concentration patterns.",
                "recommendation": "Focus improvement efforts on top 3 friction categories."
            },
            "friction_by_engineer": {
                "summary": "Displays average friction scores by engineer.",
                "key_insight": "Engineers with higher friction may be handling more complex issues.",
                "trend": "Compare against ticket volume for context.",
                "recommendation": "Investigate high-friction engineers for training or workload balancing."
            },
            "financial_impact": {
                "summary": "Visualizes financial impact across categories.",
                "key_insight": "Direct and indirect costs vary significantly by issue type.",
                "trend": "Financial exposure concentrates in specific categories.",
                "recommendation": "Prioritize cost reduction in highest-impact categories."
            },
            "recurrence": {
                "summary": "Shows recurrence prediction vs actual rates.",
                "key_insight": "Some categories have higher repeat rates than predicted.",
                "trend": "Monitor categories where actual exceeds predicted.",
                "recommendation": "Implement preventive measures for high-recurrence categories."
            },
        }

        # Find matching default
        for key, insight in defaults.items():
            if key in filename:
                return {**insight, "source": "default", "raw_response": ""}

        # Generic default
        return {
            "summary": "Visual analysis of escalation metrics.",
            "key_insight": "Review chart for patterns and anomalies.",
            "trend": "Compare against historical baselines.",
            "recommendation": "Focus on outliers and high-value areas.",
            "source": "default",
            "raw_response": ""
        }

    def analyze_charts_batch(self, chart_paths: List[str], max_workers: int = 3) -> Dict[str, Dict]:
        """
        Analyze multiple charts in parallel.

        Args:
            chart_paths: List of chart image paths
            max_workers: Number of parallel workers

        Returns:
            Dict mapping chart_path -> insight dict
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.analyze_chart, path): path
                for path in chart_paths if os.path.exists(path)
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.warning(f"Failed to analyze {path}: {e}")
                    results[path] = self._get_default_insight(path)

        return results

    def format_insight_for_dashboard(self, insight: Dict[str, Any]) -> str:
        """
        Format insight dict into a display string for Dashboard.

        Args:
            insight: Insight dictionary from analyze_chart

        Returns:
            Formatted string for Excel cell
        """
        parts = []

        if insight.get("summary"):
            parts.append(insight["summary"])

        if insight.get("key_insight"):
            parts.append(f"Key: {insight['key_insight']}")

        if insight.get("recommendation"):
            parts.append(f"Action: {insight['recommendation']}")

        return " | ".join(parts) if parts else "Analysis pending."

    def is_available(self) -> bool:
        """Check if Ollama vision model is available."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model is available
                return any(self.model.split(":")[0] in name for name in model_names)
            return False
        except:
            return False


# Global analyzer instance
_chart_analyzer: Optional[ChartInsightAnalyzer] = None


def get_chart_analyzer(model: str = None, timeout: int = None) -> ChartInsightAnalyzer:
    """
    Get or create the chart analyzer singleton.

    Args:
        model: Vision model to use (default: from config, recommended: llama3.2-vision:latest)
        timeout: Request timeout in seconds (default: from config)

    Returns:
        ChartInsightAnalyzer instance
    """
    global _chart_analyzer
    target_model = model or VISION_MODEL
    if _chart_analyzer is None or _chart_analyzer.model != target_model:
        _chart_analyzer = ChartInsightAnalyzer(model=model, timeout=timeout)
    return _chart_analyzer


def analyze_chart_image(chart_path: str, context: str = "") -> str:
    """
    Convenience function to analyze a single chart and return formatted insight.

    Args:
        chart_path: Path to chart image
        context: Optional context about the chart

    Returns:
        Formatted insight string for display
    """
    analyzer = get_chart_analyzer()
    if not analyzer.is_available():
        logger.info("Vision model not available, using default insights")
        insight = analyzer._get_default_insight(chart_path)
    else:
        insight = analyzer.analyze_chart(chart_path, context)

    return analyzer.format_insight_for_dashboard(insight)
