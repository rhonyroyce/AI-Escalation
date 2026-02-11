"""
Vision-based Chart Insight Analyzer.

Uses local vision models (Ollama) to analyze chart images and generate
deep, data-driven insights for the Dashboard.

Architecture Overview:
    This module provides AI-powered chart analysis by sending rendered chart PNG
    images to a local Ollama vision model (multimodal LLM). The vision model
    "reads" each chart image and produces structured textual insights including
    a summary, key finding, trend description, and actionable recommendation.

    These insights are embedded into the Excel report's "Visual Analytics"
    dashboard sheet (see report_generator.py -> write_dashboard()) as descriptive
    text above each chart image, giving stakeholders contextual AI commentary
    alongside the visual data.

Integration Points:
    - Called by: ExcelReportWriter.write_dashboard() in reports/report_generator.py
    - Config from: core/config.py (VISION_MODEL, VISION_MODEL_TIMEOUT, OLLAMA_BASE_URL)
    - Fallback: When Ollama is unavailable, returns static default insights keyed
      by chart filename patterns (e.g., "friction_by_category", "financial_impact")

Recommended model: llama3.2-vision:latest
- Consistently generates accurate, data-specific insights
- Works reliably with Ollama vision API
- Average response time: 3-20 seconds depending on chart complexity

Note: qwen3-vl models may have compatibility issues with Ollama image API

Typical Usage:
    # Quick one-liner for a single chart:
    insight_text = analyze_chart_image("charts/01_risk/friction_pareto.png",
                                       context="Pareto of friction by category")

    # Batch analysis for multiple charts:
    analyzer = get_chart_analyzer()
    results = analyzer.analyze_charts_batch(list_of_chart_paths, max_workers=3)

Insight Schema (returned by analyze_chart):
    {
        "summary": str,          # One-sentence chart description
        "key_insight": str,      # Most important pattern or finding
        "trend": str,            # Trend direction or notable pattern
        "recommendation": str,   # One actionable recommendation
        "raw_response": str,     # Full model response (for debugging)
        "source": str            # "vision_model" or "default"
    }
"""

import os
import base64
import logging
import json
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import vision model configuration from centralized config.
# VISION_MODEL: model name string (e.g., "llama3.2-vision:latest")
# VISION_MODEL_TIMEOUT: max seconds to wait for Ollama response (default: 120)
# OLLAMA_BASE_URL: Ollama server URL (default: "http://localhost:11434")
from ..core.config import VISION_MODEL, VISION_MODEL_TIMEOUT, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class ChartInsightAnalyzer:
    """
    Analyzes chart images using vision models to generate insights.

    Uses local Ollama models (llama3.2-vision, qwen3-vl) to:
    - Identify patterns and trends visible in charts
    - Detect anomalies and outliers
    - Generate actionable recommendations

    The analyzer implements a simple in-memory cache (self.insights_cache) keyed
    by chart file path, so repeated requests for the same chart image within a
    single report generation run avoid redundant API calls. The cache is NOT
    persisted across runs -- it lives only for the lifetime of the analyzer instance.

    Graceful Degradation:
        If Ollama is not running, the model is unavailable, or the API call fails
        for any reason, the analyzer falls back to static default insights based on
        the chart filename. This ensures the report always has *some* descriptive
        text for each chart, even without a running vision model.
    """

    def __init__(self, model: str = None, base_url: str = None, timeout: int = None):
        """
        Initialize the chart insight analyzer.

        Args:
            model: Ollama vision model to use (default from config: llama3.2-vision:latest)
            base_url: Ollama API base URL (default from config)
            timeout: Request timeout in seconds (default from config: 120)
        """
        # Fall back to config defaults if not explicitly provided.
        self.model = model or VISION_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.timeout = timeout or VISION_MODEL_TIMEOUT

        # In-memory cache: maps chart_path (str) -> insight dict.
        # Prevents duplicate Ollama calls when the same chart is requested multiple times.
        self.insights_cache: Dict[str, Dict] = {}

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for the Ollama vision API.

        Ollama's /api/generate endpoint expects images as base64-encoded strings
        in the "images" array. This reads the raw PNG bytes and encodes them.

        Args:
            image_path: Absolute or relative path to the chart PNG file.

        Returns:
            Base64-encoded string representation of the image bytes.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_ollama_vision(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Call Ollama vision API with an image.

        Sends a POST request to the Ollama /api/generate endpoint with the
        base64-encoded chart image and a structured analysis prompt. The model
        processes the image and returns textual analysis.

        API Payload Structure:
            {
                "model": "llama3.2-vision:latest",
                "prompt": "<analysis prompt>",
                "images": ["<base64 image>"],
                "stream": false,          # Get complete response, not streaming
                "options": {
                    "temperature": 0.3,   # Low temperature for focused, factual analysis
                    "num_predict": 500    # Cap response length to avoid verbose output
                }
            }

        Error Handling:
            - ImportError: requests library not installed -> returns None
            - ConnectionError: Ollama server not running -> returns None
            - Non-200 status: API error (wrong model, etc.) -> returns None
            - Any other exception: logged as warning -> returns None

        Args:
            image_path: Path to the chart image
            prompt: Analysis prompt instructing the model what to extract

        Returns:
            Model response text or None on error
        """
        # Lazy import of requests to avoid hard dependency at module level.
        # This allows the module to be imported even if requests is not installed,
        # with graceful failure only when actually attempting API calls.
        try:
            import requests
        except ImportError:
            logger.error("requests library required for Ollama API calls")
            return None

        try:
            # Encode the chart image as base64 for the API payload
            image_b64 = self._encode_image(image_path)

            # Build the Ollama /api/generate request payload.
            # "stream": False ensures we get the complete response in one JSON object
            # rather than a stream of token-by-token chunks.
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

            # POST to Ollama's generate endpoint with the configured timeout.
            # Typical response times range from 3-20 seconds depending on chart complexity
            # and model size, so the default 120s timeout provides generous headroom.
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                # Successful response: extract the "response" field from the JSON body.
                # The Ollama API returns {"response": "...", "done": true, ...}
                result = response.json()
                return result.get("response", "").strip()
            else:
                # Non-200 status indicates an API-level error (e.g., model not found,
                # invalid request format, server overloaded).
                logger.warning(f"Ollama API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            # Ollama server is not running or not reachable at the configured URL.
            logger.warning("Could not connect to Ollama. Is it running?")
            return None
        except Exception as e:
            # Catch-all for unexpected errors (file read failures, JSON parse errors, etc.)
            logger.warning(f"Vision analysis failed for {image_path}: {e}")
            return None

    def analyze_chart(self, chart_path: str, chart_context: str = "") -> Dict[str, Any]:
        """
        Analyze a single chart image and generate insights.

        This is the primary entry point for chart analysis. It checks the cache first,
        then calls the Ollama vision model with a structured prompt that requests
        analysis in four sections: SUMMARY, KEY_INSIGHT, TREND, RECOMMENDATION.

        The prompt is designed to elicit specific, data-driven responses rather than
        generic commentary. The model is instructed to reference actual numbers,
        categories, and values visible in the chart.

        Flow:
            1. Check cache -> return cached result if available
            2. Build structured analysis prompt (with optional context)
            3. Call Ollama vision API
            4. If API fails -> return default insight (filename-based fallback)
            5. Parse structured response into insight dict
            6. Cache and return the parsed insight

        Args:
            chart_path: Path to the chart image file (PNG)
            chart_context: Optional context about what the chart shows, e.g.,
                          "Pareto analysis of friction scores by AI_Category"

        Returns:
            Dict with keys: summary, key_insight, trend, recommendation,
            raw_response, source
        """
        # Check cache first to avoid redundant API calls for the same chart
        if chart_path in self.insights_cache:
            return self.insights_cache[chart_path]

        # Build the analysis prompt with structured output format.
        # The prompt uses explicit section headers (SUMMARY:, KEY_INSIGHT:, etc.)
        # to make parsing reliable. Optional chart_context is injected if provided
        # to give the model domain knowledge about what the chart represents.
        prompt = f"""Analyze this chart image and provide insights.

{f"Context: {chart_context}" if chart_context else ""}

Provide a concise analysis in this exact format:
SUMMARY: [One sentence describing what the chart shows]
KEY_INSIGHT: [The most important pattern or finding visible in the chart]
TREND: [Any trends - increasing, decreasing, stable, or notable patterns]
RECOMMENDATION: [One actionable recommendation based on what you see]

Be specific about numbers, categories, or values you can see in the chart. Keep each section to 1-2 sentences max."""

        # Send the chart image to the vision model for analysis
        response = self._call_ollama_vision(chart_path, prompt)

        if not response:
            # Return default insight if vision analysis fails.
            # This ensures every chart in the report has descriptive text
            # even when Ollama is unavailable.
            return self._get_default_insight(chart_path)

        # Parse the structured text response into a dict with named fields
        insight = self._parse_insight_response(response, chart_path)

        # Cache the result so subsequent requests for the same chart are instant
        self.insights_cache[chart_path] = insight

        return insight

    def _parse_insight_response(self, response: str, chart_path: str) -> Dict[str, Any]:
        """
        Parse the model response into structured insight dict.

        The vision model is prompted to respond in a specific format with labeled
        sections (SUMMARY:, KEY_INSIGHT:, TREND:, RECOMMENDATION:). This parser
        extracts each section by scanning lines for these headers.

        Parsing Strategy:
            - Lines are scanned top-to-bottom
            - When a section header is found, text after the colon is captured
            - Subsequent lines without headers are appended to the current section
            - Markdown bold markers (**) are stripped before parsing
            - Each field is truncated to 300 characters to fit Excel cells

        The parser is intentionally lenient:
            - Handles "KEY_INSIGHT" and "KEY INSIGHT" (with or without underscore)
            - Works with or without colons after section names
            - Continues accumulating text across multiple lines per section

        Args:
            response: Raw text response from the vision model
            chart_path: Path to the chart (for potential fallback/logging)

        Returns:
            Dict with keys: summary, key_insight, trend, recommendation,
            raw_response (original text), source ("vision_model")
        """
        import re

        # Initialize the insight dict with empty strings for all fields.
        # raw_response preserves the original model output for debugging.
        # source identifies whether the insight came from the model or defaults.
        insight = {
            "summary": "",
            "key_insight": "",
            "trend": "",
            "recommendation": "",
            "raw_response": response,
            "source": "vision_model"
        }

        # Clean markdown formatting: remove ** bold markers that vision models
        # sometimes include in their output (e.g., "**SUMMARY:**")
        clean_response = re.sub(r'\*\*', '', response)

        # Parse each section by scanning lines for known headers.
        # current_key tracks which section we are currently accumulating text for,
        # allowing multi-line sections to be captured correctly.
        lines = clean_response.split("\n")
        current_key = None

        for line in lines:
            line = line.strip()
            line_upper = line.upper()

            # Check for section headers (with or without colon).
            # Each branch: set current_key, extract text after the colon.
            if line_upper.startswith("SUMMARY"):
                current_key = "summary"
                # Split on first colon only to preserve colons within the content
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif "KEY_INSIGHT" in line_upper or "KEY INSIGHT" in line_upper:
                # Handle both underscore and space variants of the header
                current_key = "key_insight"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_upper.startswith("TREND"):
                current_key = "trend"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_upper.startswith("RECOMMENDATION"):
                current_key = "recommendation"
                insight[current_key] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif current_key and line:
                # Continue previous section: append non-header, non-empty lines
                # to the currently active section (handles multi-line responses)
                insight[current_key] += " " + line

        # Clean up whitespace and enforce 300-character limit per field.
        # This prevents excessively long text from overflowing Excel cells
        # in the dashboard layout.
        for key in ["summary", "key_insight", "trend", "recommendation"]:
            insight[key] = insight[key].strip()[:300]

        return insight

    def _get_default_insight(self, chart_path: str) -> Dict[str, Any]:
        """
        Get default insight when vision analysis is unavailable.

        Provides static, pre-written insights as a fallback when the Ollama
        vision model cannot be reached or returns an error. Insights are
        matched by checking if known keywords appear in the chart filename.

        Filename Matching Strategy:
            The chart filename (lowercased) is checked against a set of known
            chart type keywords. For example, a file named
            "friction_by_category.png" matches the "friction_by_category" key.
            If no keyword matches, a generic default is returned.

        Currently Supported Chart Types:
            - friction_by_category: Friction score distribution across categories
            - friction_by_engineer: Average friction scores per engineer
            - financial_impact: Financial impact visualization
            - recurrence: Recurrence prediction vs actual rates

        Args:
            chart_path: Path to the chart image (used for filename matching)

        Returns:
            Dict with summary, key_insight, trend, recommendation fields,
            plus source="default" and empty raw_response.
        """
        # Extract just the filename for keyword matching (case-insensitive)
        filename = os.path.basename(chart_path).lower()

        # Map chart type keywords to pre-written default insights.
        # These provide reasonable generic commentary when AI analysis
        # is not available, ensuring the report dashboard always has text.
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

        # Find matching default by checking if any keyword is a substring of the filename.
        # First match wins, so order matters if filenames could match multiple keys.
        for key, insight in defaults.items():
            if key in filename:
                # Merge the matched insight with metadata fields
                return {**insight, "source": "default", "raw_response": ""}

        # Generic default for chart types not in the defaults map.
        # This ensures every chart always has some descriptive text.
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

        Uses a ThreadPoolExecutor to send multiple chart images to the Ollama
        vision model concurrently, significantly reducing total analysis time
        when processing an entire report's worth of charts (typically 20-30 images).

        Concurrency Note:
            max_workers=3 by default to avoid overwhelming the Ollama server.
            Each vision model inference is GPU-bound, so more workers than
            available GPU capacity provides diminishing returns and may cause
            timeouts. Adjust based on hardware capabilities.

        Error Handling:
            If analysis fails for any individual chart (timeout, API error, etc.),
            the default insight is substituted for that chart. Other charts in
            the batch are not affected.

        Args:
            chart_paths: List of chart image paths to analyze
            max_workers: Number of parallel workers (default: 3)

        Returns:
            Dict mapping chart_path -> insight dict for every input path
        """
        results = {}

        # Submit all chart analysis tasks to the thread pool.
        # Only existing files are submitted (os.path.exists check filters out
        # paths to charts that were not generated, e.g., if data was missing).
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.analyze_chart, path): path
                for path in chart_paths if os.path.exists(path)
            }

            # Collect results as they complete (not necessarily in submission order).
            # as_completed() yields futures as they finish, allowing early processing.
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    # If a future raises an unexpected exception, fall back to defaults
                    logger.warning(f"Failed to analyze {path}: {e}")
                    results[path] = self._get_default_insight(path)

        return results

    def format_insight_for_dashboard(self, insight: Dict[str, Any]) -> str:
        """
        Format insight dict into a display string for the Dashboard Excel sheet.

        Combines the summary, key insight, and recommendation into a single
        pipe-delimited string suitable for embedding in an Excel cell above
        the corresponding chart image.

        Output Format Example:
            "Shows friction scores across categories. | Key: Top 3 categories
            account for 65% of friction. | Action: Focus improvement on
            Billing and Network categories."

        The trend field is intentionally excluded from the dashboard display
        to keep the text concise. It remains available in the full insight dict
        for other consumers.

        Args:
            insight: Insight dictionary from analyze_chart or _get_default_insight

        Returns:
            Formatted string for Excel cell, or "Analysis pending." if all fields empty
        """
        parts = []

        # Build the display string from available fields (skip empty ones)
        if insight.get("summary"):
            parts.append(insight["summary"])

        if insight.get("key_insight"):
            # Prefix with "Key:" to visually distinguish the finding
            parts.append(f"Key: {insight['key_insight']}")

        if insight.get("recommendation"):
            # Prefix with "Action:" to highlight the actionable recommendation
            parts.append(f"Action: {insight['recommendation']}")

        # Join with pipe delimiter for visual separation in Excel cells
        return " | ".join(parts) if parts else "Analysis pending."

    def is_available(self) -> bool:
        """
        Check if the configured Ollama vision model is available.

        Queries the Ollama /api/tags endpoint to list all locally installed models,
        then checks if the configured vision model (or its base name without tag)
        is present in the list.

        This is used as a pre-flight check before attempting chart analysis.
        If the model is not available, callers can skip the API call overhead
        and go straight to default insights.

        The check uses a short 5-second timeout to avoid blocking the report
        generation pipeline if Ollama is unresponsive.

        Returns:
            True if the vision model is installed and Ollama is reachable,
            False otherwise (including any connection or parsing errors).
        """
        try:
            import requests
            # Query Ollama's model listing endpoint
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model is available by matching the base model name
                # (before the colon/tag). E.g., "llama3.2-vision" matches
                # "llama3.2-vision:latest" or "llama3.2-vision:11b".
                return any(self.model.split(":")[0] in name for name in model_names)
            return False
        except:
            # Bare except: any failure (connection refused, timeout, JSON error,
            # missing requests library) means the model is not available.
            return False


# ---------------------------------------------------------------------------
# Module-Level Singleton and Convenience Functions
# ---------------------------------------------------------------------------

# Global analyzer instance for the singleton pattern.
# Initialized lazily by get_chart_analyzer() on first call.
_chart_analyzer: Optional[ChartInsightAnalyzer] = None


def get_chart_analyzer(model: str = None, timeout: int = None) -> ChartInsightAnalyzer:
    """
    Get or create the chart analyzer singleton.

    Implements a lazy singleton pattern: the ChartInsightAnalyzer is created on
    first call and reused for subsequent calls. If a different model is requested,
    the singleton is replaced with a new instance configured for that model.

    This singleton approach ensures:
        - The insights cache is shared across all chart analyses in a single run
        - Only one Ollama connection/model configuration exists at a time
        - Callers don't need to manage analyzer lifecycle

    Args:
        model: Vision model to use (default: from config, recommended: llama3.2-vision:latest)
        timeout: Request timeout in seconds (default: from config)

    Returns:
        ChartInsightAnalyzer instance (shared singleton)
    """
    global _chart_analyzer
    # Determine the target model, falling back to config default
    target_model = model or VISION_MODEL
    # Create a new instance if none exists or if the requested model differs
    # from the current singleton's model (allows runtime model switching)
    if _chart_analyzer is None or _chart_analyzer.model != target_model:
        _chart_analyzer = ChartInsightAnalyzer(model=model, timeout=timeout)
    return _chart_analyzer


def analyze_chart_image(chart_path: str, context: str = "") -> str:
    """
    Convenience function to analyze a single chart and return formatted insight.

    This is the simplest way to get a dashboard-ready insight string for a chart.
    It handles the full pipeline: singleton management, availability check,
    vision analysis (or fallback to defaults), and formatting.

    Called by report_generator.py's write_dashboard() method for each chart
    image that needs an insight annotation in the Excel report.

    Flow:
        1. Get (or create) the singleton ChartInsightAnalyzer
        2. Check if Ollama vision model is available
        3. If available: run full vision analysis
        4. If unavailable: use static default insights
        5. Format the insight dict into a pipe-delimited display string

    Args:
        chart_path: Path to chart image (PNG file)
        context: Optional context about the chart for the vision model

    Returns:
        Formatted insight string for display in Excel dashboard cell.
        Example: "Shows friction distribution. | Key: Top 3 = 70%. | Action: Focus on Billing."
    """
    analyzer = get_chart_analyzer()
    if not analyzer.is_available():
        # Skip the API call entirely if Ollama is not reachable or the model
        # is not installed. Use pre-written default insights instead.
        logger.info("Vision model not available, using default insights")
        insight = analyzer._get_default_insight(chart_path)
    else:
        # Full vision analysis: encode image, send to model, parse response
        insight = analyzer.analyze_chart(chart_path, context)

    # Format the structured insight into a single display string
    return analyzer.format_insight_for_dashboard(insight)
