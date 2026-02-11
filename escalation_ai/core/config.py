"""
Central Configuration Module for Escalation AI.

=== PURPOSE ===
This module is the single source of truth for every tunable constant, weight,
threshold, column mapping, and classification taxonomy used across the entire
Escalation AI pipeline.  Every other module imports from here rather than
defining its own magic numbers, which keeps the system easy to audit and
re-tune without scattered code changes.

=== DATA FLOW ===
  1. On import, this module auto-detects GPU VRAM and selects appropriate
     AI model tiers (embedding + generative) so downstream inference code
     can simply reference EMBED_MODEL / GEN_MODEL.
  2. The WEIGHTS dict drives the scoring engine (escalation_ai.scoring):
     Base_Severity x Type_Multiplier x Origin_Multiplier x Impact_Multiplier.
  3. ANCHORS / SUB_CATEGORIES / CATEGORY_KEYWORDS define the 8-category
     classification taxonomy with ~60 sub-categories.  The classification
     engine uses these for keyword/phrase/regex-based hybrid matching.
  4. ROOT_CAUSE_CATEGORIES power the root-cause classifier inside the
     scoring engine, mapping free-text root-cause descriptions to buckets.
  5. COL_* constants abstract away the raw spreadsheet column names so that
     if the input schema changes, only this file needs updating.
  6. Financial parameters (PRICE_CATALOG_FILE, DEFAULT_HOURLY_RATE) feed
     the price catalog and financial metrics modules.

=== KEY DESIGN DECISIONS ===
- McKinsey-style strategic framework: base severity x cascading multipliers
  produces a Strategic Friction Score that orders escalations by business risk.
- 8-category taxonomy (not free-form) ensures repeatable classification that
  maps cleanly to the price catalog for financial impact costing.
- Recidivism penalties (1.25x / 1.5x) boost scores for repeat issues so they
  surface in executive dashboards before becoming chronic.
- GPU tier selection is automatic; operators never need to pick a model
  manually -- just install the GPU and the system adapts.

Contains all constants, weights, thresholds, and column mappings.
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

# ==========================================
# GPU / RAPIDS CONFIGURATION
# ==========================================
# When True, the pipeline will attempt to use NVIDIA RAPIDS (cuML, cuDF)
# for GPU-accelerated data processing.  Falls back gracefully to CPU if
# no compatible GPU is detected.
USE_GPU = True  # Enable GPU acceleration when available (cuML, cuDF)

# Safety cap on VRAM consumption.  0.8 means "use up to 80% of total VRAM,"
# leaving headroom for the OS compositor, Ollama server, and other GPU
# workloads running concurrently.
GPU_MEMORY_LIMIT = 0.8  # Max fraction of GPU memory to use (0.0-1.0)

# ==========================================
# VRAM DETECTION & MODEL SELECTION
# ==========================================
def get_gpu_vram_gb():
    """Detect GPU VRAM in GB. Returns 0 if no NVIDIA GPU found.

    Shells out to ``nvidia-smi`` with a 5-second timeout.  If the binary
    is missing (no NVIDIA driver), the process times out, or the output
    cannot be parsed, the function silently returns 0 so the caller falls
    back to CPU-only / smallest-model mode.

    Returns:
        float: Total VRAM of the first GPU in gigabytes, or 0.0 on failure.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # nvidia-smi returns VRAM in MiB; convert to GiB for tier comparison
            # If multiple GPUs exist, only the first GPU's VRAM is considered
            vram_mb = int(result.stdout.strip().split('\n')[0])
            return vram_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        # FileNotFoundError  -> nvidia-smi not installed (no NVIDIA driver)
        # TimeoutExpired     -> driver hung or GPU inaccessible
        # ValueError         -> unexpected output format from nvidia-smi
        pass
    return 0

def select_models():
    """Select AI models based on available VRAM.

    The system uses a three-tier model strategy to maximise quality while
    staying within hardware constraints:

    - **High tier** (>=20 GB VRAM, e.g. RTX 3090/4090, A5000):
      Largest embedding (8B param) + largest generative (20B param) models.
    - **Medium tier** (>=12 GB VRAM, e.g. RTX 3060 12GB, RTX 4070):
      Mid-size embedding (4B param) + full generative model (20B still fits
      because embeddings are smaller).
    - **Low tier** (<12 GB or CPU-only):
      Smallest embedding (0.6B param) + small generative model (8B param).
      This tier is also used as the graceful fallback when no GPU is found.

    Returns:
        tuple: (embed_model_name, gen_model_name, detected_vram_gb)
    """
    vram_gb = get_gpu_vram_gb()

    # --------------- Model tiers based on VRAM ---------------
    if vram_gb >= 20:
        # High VRAM (24GB+): Use largest models for best classification accuracy
        embed_model = "qwen3-embedding:8b"
        gen_model = "gpt-oss:20b"
        tier = "high"
    elif vram_gb >= 12:
        # Medium VRAM (16GB): Use medium embedding but still large generative model
        # The 4B embedding fits in ~8GB, leaving room for the 20B gen model in offload mode
        embed_model = "qwen3-embedding:4b"
        gen_model = "gpt-oss:20b"
        tier = "medium"
    else:
        # Low VRAM (<12GB) or CPU only: Use smallest models to stay within limits
        embed_model = "qwen3-embedding:0.6b"
        gen_model = "qwen3:8b"
        tier = "low"

    logger.info(f"[Config] Detected {vram_gb:.1f}GB VRAM → {tier} tier models")
    return embed_model, gen_model, vram_gb

# --- Module-level auto-detection ---
# These three constants are resolved once at import time so every downstream
# module gets the same model selection without re-probing the GPU.
EMBED_MODEL, GEN_MODEL, DETECTED_VRAM_GB = select_models()

# ==========================================
# AI MODEL CONFIGURATION
# ==========================================
# Base URL for the local Ollama inference server.  Both embedding and
# generative models are served from this endpoint.
OLLAMA_BASE_URL = "http://localhost:11434"
# EMBED_MODEL and GEN_MODEL are set automatically above based on VRAM

# ==========================================
# STRATEGIC WEIGHTS (McKinsey Framework)
# ==========================================
# These weights implement a multiplicative scoring model inspired by
# McKinsey's risk-impact matrices.  The scoring formula is:
#
#   Strategic_Friction_Score = BASE_SEVERITY
#                              x TYPE_MULTIPLIER
#                              x ORIGIN_MULTIPLIER
#                              x IMPACT_MULTIPLIER
#
# Each dimension is independent, so a Critical + External + High-Impact
# escalation scores: 100 x 1.5 x 2.5 x 2.0 = 750 (top of scale),
# while a Minor + Internal + No-Impact concern: 10 x 1.0 x 1.0 x 1.0 = 10.
#
# This dynamic range (5 to 750) lets dashboards clearly separate the
# "burning platform" items from routine noise.
WEIGHTS = {
    # BASE_SEVERITY: Absolute starting score keyed on the severity field.
    # Critical items start at 100 (hard floor), Minor at 10, unknown at 5.
    'BASE_SEVERITY': {'Critical': 100, 'Major': 50, 'Minor': 10, 'Default': 5},

    # TYPE_MULTIPLIER: Escalations are 1.5x (customer-visible friction),
    # Concerns are 1.0x (internal-only), Lessons Learned are 0.0x (informational,
    # not risks -- they should not inflate the friction score).
    'TYPE_MULTIPLIER': {'Escalations': 1.5, 'Concerns': 1.0, 'Lessons Learned': 0.0},

    # ORIGIN_MULTIPLIER: External-origin issues carry 2.5x weight because they
    # are visible to the customer / vendor and carry reputational + SLA risk.
    # Internal issues are 1.0x (containable within the organisation).
    'ORIGIN_MULTIPLIER': {'External': 2.5, 'Internal': 1.0},

    # IMPACT_MULTIPLIER: High business impact doubles the score; Low adds a
    # small 10% uplift; None leaves the score unchanged.
    'IMPACT_MULTIPLIER': {'High': 2.0, 'Low': 1.1, 'None': 1.0}
}

# ==========================================
# SIMILARITY & RECIDIVISM THRESHOLDS
# ==========================================
# These thresholds govern the AI-driven similarity engine that detects
# whether a new escalation is a repeat of a historical one.

# Cosine-similarity score (0-1) from the embedding model.
# >= 0.60 is treated as a confirmed match (high confidence).
SIMILARITY_THRESHOLD_HIGH = 0.60   # High confidence match

# >= 0.50 is flagged as a possible match worth human review.
SIMILARITY_THRESHOLD_MEDIUM = 0.50  # Medium confidence - worth flagging

# Keyword-overlap ratio (Jaccard-style) between two issue summaries.
# 35% overlap triggers the "likely related" flag even if embedding similarity
# is below threshold -- catches cases where wording differs but topics match.
KEYWORD_OVERLAP_THRESHOLD = 0.35    # 35% keyword overlap = likely related

# ==========================================
# CLASSIFICATION SETTINGS
# ==========================================
# Minimum confidence score (0-1) from the hybrid classifier.  If the best-
# matching category scores below this, the ticket is labelled "Unclassified"
# rather than forcing a low-confidence assignment.
MIN_CLASSIFICATION_CONFIDENCE = 0.25  # Below this = "Unclassified"

# ==========================================
# RECIDIVISM PENALTY MULTIPLIERS
# ==========================================
# When the similarity engine confirms that an escalation is a repeat of a
# known issue, the Strategic Friction Score is multiplied by these penalties.
# This surfaces chronic problems in executive dashboards.

# Confirmed repeat (similarity >= SIMILARITY_THRESHOLD_HIGH): +50% score boost
RECIDIVISM_PENALTY_HIGH = 1.5      # 50% score increase for confirmed repeats

# Possible repeat (similarity >= SIMILARITY_THRESHOLD_MEDIUM): +25% score boost
RECIDIVISM_PENALTY_MEDIUM = 1.25   # 25% score increase for possible repeats

# ==========================================
# FEEDBACK/REINFORCEMENT LEARNING SETTINGS
# ==========================================
# The system supports a human-in-the-loop feedback loop:
# analysts can override AI classifications and resolution estimates,
# and those corrections are persisted in Excel files that the next
# pipeline run reads to re-weight its predictions.

# Stores analyst overrides of AI category/sub-category assignments.
FEEDBACK_FILE = "classification_feedback.xlsx"

# Stores analyst corrections to predicted resolution times.
RESOLUTION_FEEDBACK_FILE = "resolution_feedback.xlsx"  # Persistent human feedback for resolution times

# Blending weight for user feedback vs original model predictions.
# 0.4 means 40% feedback influence, 60% model -- prevents a single
# bad override from dominating while still rewarding consistent corrections.
FEEDBACK_WEIGHT = 0.4  # How much weight to give user feedback vs original anchors (0-1)

# JSON file storing analyst accept/reject decisions on similarity matches.
SIMILARITY_FEEDBACK_PATH = "similarity_feedback.json"

# ==========================================
# ML MODEL PATHS
# ==========================================
# Serialised scikit-learn / cuML models trained on historical escalation data.

# Predicts probability of an issue recurring (binary classifier).
RECURRENCE_MODEL_PATH = "recurrence_model.pkl"

# Label encoders fitted during recurrence model training.
RECURRENCE_ENCODERS_PATH = "recurrence_encoders.pkl"

# Predicts expected resolution time (regression model).
RESOLUTION_MODEL_PATH = "resolution_model.pkl"

# ==========================================
# VISION MODEL SETTINGS (for chart insights)
# ==========================================
# The pipeline can optionally feed generated charts (PNG) into a vision-
# language model to produce textual insights ("this chart shows a spike
# in Site Readiness issues in Q3...").

# Recommended: llama3.2-vision:latest (best compatibility with Ollama)
# Alternative: qwen3-vl:8b (may have compatibility issues)
VISION_MODEL = os.environ.get("VISION_MODEL", "llama3.2-vision:latest")

# Timeout in seconds for a single vision-model inference call.
# Vision models are slow (~30-120s per chart depending on resolution).
VISION_MODEL_TIMEOUT = int(os.environ.get("VISION_MODEL_TIMEOUT", "120"))

# Allow override via environment variable for containerised deployments.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ==========================================
# OUTPUT DIRECTORIES
# ==========================================
from pathlib import Path

# All generated chart PNGs (heatmaps, funnels, waterfalls, etc.) are
# written here.  The directory is created lazily by the plotting code.
PLOT_DIR = Path("plots")  # Directory for generated charts

# ==========================================
# PRICE CATALOG SETTINGS
# ==========================================
# The price catalog (price_catalog.xlsx) is the single source of truth for
# translating escalation categories into dollar-denominated financial impact.

# Path to the Excel-based price catalog workbook.
PRICE_CATALOG_FILE = "price_catalog.xlsx"

# Fallback hourly labour rate used if the price catalog does not specify one
# for a given category.  Represents a blended loaded cost rate ($/hr).
DEFAULT_HOURLY_RATE = 20.0  # $/hour for labor if not specified in price_catalog.xlsx

# ==========================================
# REPORT METADATA
# ==========================================
# Version and title stamped into the generated Strategic_Report.xlsx output.
REPORT_VERSION = "2.2"
REPORT_TITLE = "STRATEGIC FRICTION ANALYSIS"

# ==========================================
# ENGINEER & LOB THRESHOLDS
# ==========================================
# Used by the accountability tracker in the scoring engine to flag repeat
# offenders and high-risk lines of business.

# An engineer with >= this many escalations/concerns is flagged for review.
ENGINEER_REPEAT_THRESHOLD = 3  # Flag engineers with 3+ issues

# A Line of Business (LOB) with >= this many issues triggers a risk alert.
LOB_RISK_THRESHOLD = 5  # Flag LOBs with 5+ issues

# ==========================================
# UI COLORS
# ==========================================
# Corporate palette constant used by Matplotlib/Streamlit charting code.
MC_BLUE = '#004C97'

# ==========================================
# CATEGORIZATION ANCHORS
# 8-category system with sub-categories optimised for telecom escalation
# analysis.  Based on manual review of 300+ historical error samples.
#
# HOW ANCHORS WORK:
# Each category has a flat list of anchor keywords/phrases.  The hybrid
# classifier computes a weighted overlap score between the ticket's issue
# summary and these anchors.  The category with the highest score wins.
# Sub-category assignment (see SUB_CATEGORIES below) refines the match.
#
# WHY 8 CATEGORIES:
# Fewer categories (e.g. 4) were too broad for actionable insights.
# More categories (e.g. 15+) caused classification confusion because
# telecom escalation language is heavily overlapping.  8 was the sweet
# spot identified during iterative testing on real data.
# ==========================================
ANCHORS = {
    # --- Category 1: Scheduling & Planning ---
    # Covers issues where the field engineer (FE), site, or ticket was not
    # properly scheduled in the TI (Ticket/Integration) system, or the
    # schedule was not followed.
    "Scheduling & Planning": [
        "schedule", "TI", "Ti", "planned", "unplanned", "calendar",
        "date", "time", "weekend", "holiday", "closeout", "close out",
        "not schedule", "not scheduled", "schedule missing", "no schedule",
        "without schedule", "TI entry missing", "not in TI", "not in Ti",
        "FE on site", "FE logged", "logged for IX", "logged for support",
        "ticket in closeout", "moved to closeout", "wrong bucket"
    ],

    # --- Category 2: Documentation & Reporting ---
    # Missing or incorrect documentation artifacts: snapshots, CBN outputs,
    # E911 records, RTT reports, EOD summaries, email attachments, etc.
    "Documentation & Reporting": [
        "snapshot", "snap", "screenshot", "missing", "wrong", "incorrect",
        "attachment", "logs", "report", "email", "mail", "subject",
        "CBN", "E911", "RTT", "EOD", "summary", "table",
        "missing snapshot", "wrong site ID", "different site",
        "forgot to attach", "forgot to paste", "CBN output missing",
        "incomplete snapshot", "wrong attachment", "PSAP contact missing"
    ],

    # --- Category 3: Validation & QA ---
    # Quality-assurance failures: missed checks, incomplete validation,
    # wrong pass/fail determinations, undetected degradations, skipped
    # activation or call-test steps.
    "Validation & QA": [
        "missed", "miss", "check", "validation", "verify", "audit",
        "precheck", "postcheck", "VSWR", "RSSI", "RTWP", "KPI",
        "pass", "fail", "detected", "captured", "escalated",
        "missed to check", "missed to report", "missed to capture",
        "not detected", "wrong pass fail", "incomplete validation",
        "degradation not detected", "issue not escalated"
    ],

    # --- Category 4: Process Compliance ---
    # Violations of standard operating procedures (SOPs): skipped steps,
    # wrong escalation targets (e.g. NTAC), missing approvals, wrong
    # ticket buckets, proceeding without required pre-conditions.
    "Process Compliance": [
        "process", "SOP", "guideline", "escalate", "NTAC", "distro",
        "approval", "without", "skip", "bypass", "procedure", "step",
        "released without", "supported without", "without BH actualized",
        "escalated to NTAC", "wrong distro", "not following process",
        "skipped step", "bypassed validation", "improper escalation"
    ],

    # --- Category 5: Configuration & Data Mismatch ---
    # Mismatches between planning documents and physical site configuration:
    # port matrices, RET naming, TAC codes, CIQ/SCF/RFDS discrepancies,
    # One-T system data, IP addresses, spectrum allocation.
    "Configuration & Data Mismatch": [
        "mismatch", "port matrix", "PMX", "SCF", "CIQ", "RFDS", "TAC",
        "RET", "naming", "One-T", "OneT", "configuration", "config",
        "port matrix mismatch", "RET naming issue", "RET naming wrong",
        "TAC mismatch", "SCF mismatch", "CIQ mismatch", "RFDS mismatch",
        "need updated port matrix", "RET swap", "RIOT red", "not matching"
    ],

    # --- Category 6: Site Readiness ---
    # The physical site or its backhaul (BH) / microwave (MW) link was not
    # ready for integration: missing materials, BH not actualised, site
    # down, power issues, complexity (e.g. MW chain dependencies).
    "Site Readiness": [
        "BH", "backhaul", "actualized", "ready", "MW", "microwave",
        "material", "missing", "SFP", "cancelled", "down", "pending",
        "BH not actualized", "backhaul not ready", "BH not ready",
        "MW not ready", "microwave not ready", "site not ready",
        "material missing", "SFP missing", "site was down",
        "cancelled due to", "transmission not ready"
    ],

    # --- Category 7: Communication & Response ---
    # Delays in human communication: late replies to field engineers or
    # general contractors, missing proactive updates, repeated queries,
    # training/competency gaps in the communication chain.
    "Communication & Response": [
        "delay", "delayed", "reply", "response", "waiting", "waited",
        "hours", "follow-up", "reminder", "communicated", "query",
        "delayed reply", "delayed response", "delay in reply",
        "waited for hours", "FE waited", "GC query not replied",
        "no reply from", "follow-up required", "communication gap"
    ],

    # --- Category 8: Nesting & Tool Errors ---
    # Errors in the nesting workflow (NSA/SA/NSI/SI states) or tool-side
    # issues: wrong nest type for the market, nest extensions, RIOT/FCI
    # tool mismatches, post-on-air degradations, hardware faults,
    # premature cell unlocks.
    "Nesting & Tool Errors": [
        "nest", "nested", "nesting", "NSA", "SA", "NSI", "SI",
        "RIOT", "FCI", "TI", "tool", "updated", "market", "guideline",
        "nested as NSA", "nested as SA", "nested as NSI",
        "wrong nest type", "nest extended", "nest extension",
        "not allowed in market", "RIOT mismatch", "tool not updated",
        "FCI not updated", "site not unnested", "OSS mismatch"
    ]
}

# ==========================================
# SUB-CATEGORIES - Detailed sub-types from Embedding.md analysis
# Based on 300+ error sample classification
#
# PURPOSE:
# After the classifier picks one of the 8 top-level categories, it drills
# down into sub-categories for finer-grained labelling.  Sub-categories
# map 1:1 to rows in the price catalog's "Sub-Category Costs" sheet,
# enabling per-sub-type financial impact costing.
#
# STRUCTURE:
# {
#   "Top Category": {
#       "Sub-Category Name": [list of anchor phrases],
#       ...
#   },
#   ...
# }
# ==========================================
SUB_CATEGORIES = {
    # ---- Scheduling & Planning sub-categories ----
    "Scheduling & Planning": {
        # Site was never entered into the TI scheduling system
        "No TI Entry": [
            "not schedule in Ti", "TI entry missing", "not in TI", "not in Ti",
            "schedule missing in Ti", "no TI entry", "Ti entry missing",
            "Site not schedule in Ti", "not schedule for integration in Ti",
            "FE on site for integration", "site was not schedule"
        ],
        # FE logged on a different day or time than what was scheduled
        "Schedule Not Followed": [
            "schedule not followed", "Site schedule for", "but FE logged",
            "FE didn't login", "schedule for IX on", "but FE didn't",
            "Incorrect Schedule", "site schedule", "FE logged on"
        ],
        # Issues specific to weekend scheduling policies
        "Weekend Schedule Issue": [
            "weekend", "not site schedule on weekend", "over weekend",
            "not schedule on weekend", "site schedule on weekend"
        ],
        # Ticket found in wrong lifecycle bucket (e.g. closeout before work done)
        "Ticket Status Issue": [
            "closeout", "close out", "ticket in closeout", "Ticket was in closeout",
            "moved to closeout", "ticket moved", "Ticket is in closeout",
            "closeout bucket", "moved from closeout"
        ],
        # Site scheduled for integration before backhaul/microwave was ready
        "Premature Scheduling": [
            "without microwave ready", "without BH ready", "BH not ready",
            "schedule without", "MW not ready", "site not ready"
        ]
    },

    # ---- Documentation & Reporting sub-categories ----
    "Documentation & Reporting": {
        # CBN or validation screenshot not included in the report
        "Missing Snapshot": [
            "snapshot missing", "missing snapshot", "CBN snapshot missing",
            "snapshot is missing", "snap missing", "lemming snap missing",
            "validation snapshot missing", "CBN output snapshot missing"
        ],
        # Pre-check logs or other required files not attached to the ticket/email
        "Missing Attachment": [
            "missed to attach", "forgot to attach", "missing attachment",
            "not attached", "attachment missing", "Pre-check logs"
        ],
        # Data in the report is factually wrong (wrong RTT, wrong description)
        "Incorrect Reporting": [
            "incorrectly", "RTT from IX vendor was also wrong",
            "didn't provide a clear view", "wrong which", "reporting"
        ],
        # Log paste or manual log entry was omitted from the record
        "Missing Logs": [
            "forgets to paste", "manual logs", "missing logs",
            "no logs populated", "logs missing"
        ],
        # Correct document type but for the wrong site/sector
        "Wrong Attachment": [
            "wrong attachment", "wrongly attached", "different site",
            "another site", "Detailed site view snap"
        ],
        # Site ID in the email/ticket does not match the actual site worked
        "Wrong Site ID": [
            "wrong site ID", "different site id", "Wrong site id",
            "mistakenly placed a different site", "site ID missing",
            "Different site ID", "Different mail ID"
        ],
        # VSWR/pass/fail status marked opposite to actual measurement
        "Incorrect Status": [
            "marked VSWR as Fail", "actual status is", "incorrect information",
            "marked as Pass", "but actual", "incorrect status",
            "Live call test details", "mentioned in summary"
        ],
        # Snapshot exists but is missing sectors or incomplete coverage
        "Incomplete Snapshot": [
            "Incomplete lemming snapshot", "incomplete snapshot",
            "only 2 sectors", "sector was missing", "In-complete"
        ],
        # Numbers or identifiers in the report don't match the source data
        "Data Mismatch": [
            "mentioned different", "as compared to", "mismatch",
            "Count of sectors mentioned different"
        ],
        # Required data fields (PSAP contact, Live CT, summary table) absent
        "Missing Information": [
            "PSAP contact missing", "Table with PSAP", "missing",
            "not available", "missing in mail", "summary table missing",
            "Live CT missing", "Live CT CBN output"
        ],
        # Email subject line does not follow the required format
        "Wrong Subject": [
            "subject line", "Subject should have been", "wrong subject"
        ],
        # A required process step (e.g. E911 Go mail) was skipped before action
        "Process Skip": [
            "without sending", "E911 completion mail circulated",
            "without E911 Go mail", "skipped"
        ],
        # A required deliverable (RTT, EOD report) was not released or was late
        "Missing Deliverable": [
            "RTT wasn't released", "EOD released", "not released",
            "released later"
        ]
    },

    # ---- Validation & QA sub-categories ----
    "Validation & QA": {
        # General contractor (GC) was checked out before all validations complete
        "Premature Checkout": [
            "checked out GC", "Logged the GC out", "cells were locked",
            "premature checkout", "early checkout"
        ],
        # Validation steps performed but critical fields were not checked
        "Incomplete Validation": [
            "without checking", "IX precheck without", "not updated",
            "incomplete validation", "BH fields", "MagentaBuilt"
        ],
        # Change Request (CR) submitted with invalid or wrong data
        "Invalid CR": [
            "Invalid CR", "invalid CR manually", "wrong CR"
        ],
        # A flagged anomaly (e.g. max tilt) was seen but not acted upon
        "Ignored Anomaly": [
            "anomaly detection", "left max tilt", "even though it was captured",
            "ignored anomaly", "max tilt values"
        ],
        # An issue (fiber, SFP, etc.) was present but not identified/reported
        "Missed Issue": [
            "did not report", "Miss to capture", "missed to create",
            "Engineer did not identified", "missed to attach",
            "missed to report", "fiber issue", "SFP issue"
        ],
        # Support was incorrectly denied when it should have been approved
        "Wrong Denial": [
            "denied for Anchor site", "IX precheck denied",
            "wrong BH validation", "BH was already actualized"
        ],
        # A required activation (e.g. VoNR) was not performed
        "Missed Activation": [
            "VONR is not activated", "not activated", "missed activation"
        ],
        # A validation checkpoint (cell status, RSSI) was skipped entirely
        "Missed Check": [
            "Miss to check", "Cell Status Verification",
            "cells were in Unsync", "missed to check"
        ],
        # Issue was captured/detected but never escalated to the right team
        "No Escalation": [
            "not escalated", "but not escalated", "no escalation",
            "captured but not escalated"
        ],
        # KPIs used during validation did not match the agreed-upon set
        "Wrong KPIs": [
            "different set of KPIs", "wrong KPIs", "not aligned with the agreed"
        ],
        # A KPI degradation was present but not detected during checks
        "Missed Degradation": [
            "degradation was not detected", "locked and showing zero traffic",
            "missed degradation", "not detected during"
        ],
        # Moved to post-check phase without completing required validations
        "Skipped Validation": [
            "moved to Post check without", "skipped validation",
            "without Riot lemming validation"
        ],
        # Wrong tool or wrong tool output was used for validation
        "Wrong Tool Usage": [
            "AEHC swap report", "tool will only flag", "wrong tool",
            "Does Not Recognize Physical Swap"
        ],
        # Support denied prematurely despite valid conditions
        "Premature Denial": [
            "IX support was denied", "despite BH status", "citing the reason",
            "TI record was not updated", "denied support"
        ],
        # Testing (E911, VoNR, call test) was started but not completed
        "Incomplete Testing": [
            "missing L21 Gamma", "VoNR testing", "call test",
            "calls were completed and passed", "incomplete testing"
        ],
        # A required call test step was missed entirely
        "Missed Call Test": [
            "AWS3 E911 call test", "missed", "released full RTT",
            "caught during audit", "schedule FE for call test"
        ],
        # Configuration was changed but never reverted to original values
        "No Reversion": [
            "did not revert", "original values", "no reversion",
            "tilt settings", "not reverted"
        ],
        # A technology band (e.g. N6) was missed during multi-tech integration
        "Missed Tech": [
            "N6 was missed", "missed to add", "NSD site", "revisit for CT"
        ]
    },

    # ---- Process Compliance sub-categories ----
    "Process Compliance": {
        # A required process step was forgotten (unlock, share precheck, etc.)
        "Missed Step": [
            "missed to share", "missed to unlock", "Engineer missed",
            "missed step", "didn't lock the cells"
        ],
        # Issue was identified but never escalated to the responsible team
        "No Escalation": [
            "not escalated to concerned", "created the issue ticket but not",
            "no escalation", "not escalated"
        ],
        # Escalation sent to wrong vendor, wrong distribution list, or NTAC
        "Wrong Escalation": [
            "escalated to different vendor", "wrong distro",
            "escalated to NTAC", "which is raised by client",
            "wrong escalation", "should not reach NTAC"
        ],
        # Ticket placed in incorrect workflow bucket (e.g. design vs execution)
        "Wrong Bucket": [
            "not in correct bucket", "wrong bucket", "Preliminary design",
            "Ticket is not created", "ticket was in Design phase"
        ],
        # Work proceeded without required pre-conditions being met
        "Process Violation": [
            "IX supported without", "without Backhaul acceptance",
            "released by PAG BO without", "BH not actualized",
            "process violation", "without backhaul"
        ],
        # A required PAG or issue ticket was never created
        "Missing Ticket": [
            "ticket not created", "Ticket is not created", "PAG ticket",
            "not created for the site", "missing ticket"
        ],
        # Work started while the site was in wrong nesting state (e.g. NSI)
        "Wrong Nest State": [
            "sector swap started by GC with site nested in NSI",
            "wrong nest state", "nested in"
        ],
        # Integration used an outdated or incorrect configuration file
        "Wrong File Used": [
            "some other file was used", "wrong file", "correct SCF",
            "during integration some other"
        ],
        # Engineer could have proactively guided the FE but did not
        "Missed Guidance": [
            "could have informed FE", "if PSAP was busy",
            "whitelisted in lieu of", "missed guidance"
        ],
        # General non-compliance with published guidelines or SOPs
        "Process Non-Compliance": [
            "adhere to the guidelines", "inquiries from",
            "Tool Request Submission Process", "non-compliance"
        ]
    },

    # ---- Configuration & Data Mismatch sub-categories ----
    "Configuration & Data Mismatch": {
        # RET antenna naming has typos (extra character, missing letter)
        "RET Naming": [
            "RET naming issue", "RET naming", "extra '2'",
            "inadvertently removed I", "RET naming mismatch",
            "naming having an extra", "I missing", "naming is incorrect"
        ],
        # Alpha/Beta/Gamma RET identifiers are swapped between sectors
        "RET Swap": [
            "RET naming swap", "swapped", "RET swap",
            "controlling unit", "Beta & Gamma"
        ],
        # RET tilt parameter found at an invalid value (e.g. max tilt)
        "RET Parameter": [
            "RET found Max", "max tilt", "RET parameter"
        ],
        # Port matrix document does not match physical wiring/RET count
        "Port Matrix Mismatch": [
            "port matrix", "Port Matrix", "PMX", "Incorrect Port matrix",
            "port matrix mismatch", "as per port matrix", "wrong port matrix",
            "2 RET's defined per sector", "3 RET's per sector",
            "need updated port matrix", "Port Matrix is missing"
        ],
        # CIQ (Cell Information Questionnaire) and SCF (Site Config File) disagree
        "CIQ/SCF Mismatch": [
            "CIQ and SCF", "SCF mismatch", "CIQ mismatch",
            "Mismatch in Spectrum sheet", "SCF provided with",
            "CIQ provided not matching"
        ],
        # TAC (Tracking Area Code) does not match expected value; causes RIOT red
        "TAC Mismatch": [
            "TAC mismatch", "TAC showing mismatched", "RIOT red",
            "TAC was", "MB TAC", "RF Config and OSS"
        ],
        # RFDS (RF Design Specification) mismatches with other planning docs
        "RFDS Mismatch": [
            "RFDS", "Mismatch of SCF", "RFDS mismatch",
            "SCF and RFDS mismatch", "RFDS not mention"
        ],
        # One-T (unified network management) data not synchronised
        "One-T Mismatch": [
            "One-T", "OneT", "One-T not yet updated",
            "One-T mismatch", "IP exports", "non-SRAN"
        ],
        # Physical hardware on site does not match the design specification
        "Design Mismatch": [
            "Design change", "HW available on site", "ASIA",
            "RFDS with ASIB", "design mismatch"
        ],
        # Required planning documents (RFDS, Port Matrix) missing from TI
        "Missing Documents": [
            "RFDS & Port Matrix is missing", "Port Matrix is missing in TI",
            "missing in TI", "missing documents"
        ],
        # Software/configuration error: undefined PLMNs, eNB IDs, BWP, etc.
        "Config Error": [
            "Config Error", "configuration error", "error due to",
            "NRPLMNSET not define", "enbid", "not define in SCF"
        ],
        # A required configuration feature was not activated or is missing
        "Missing Config": [
            "BWP missing", "feature was not activated", "missing config",
            "MOCN", "Scripts were not implemented", "external neighbour"
        ],
        # IP address in One-T/SCF does not match what is configured on the node
        "IP Mismatch": [
            "IP mismatch", "Gateway IP different", "wrong ip",
            "IP different", "One-T and SCF"
        ],
        # ARFCN/bandwidth allocation does not match the spectrum plan
        "Spectrum Mismatch": [
            "ARFCN and bandwidth", "spectrum plan", "spectrum mismatch"
        ]
    },

    # ---- Site Readiness sub-categories ----
    "Site Readiness": {
        # Backhaul link not actualised (active) in the management system
        "BH Not Ready": [
            "BH not ready", "backhaul not ready", "BH was not ready",
            "backhaul acceptance", "BH readiness pending", "BH not actualized",
            "BH actualization", "backhaul readiness"
        ],
        # Microwave transport link not yet ready for traffic
        "MW Not Ready": [
            "MW not ready", "microwave not ready", "Microwave link was not ready",
            "MW link was not ready", "incomplete microwave"
        ],
        # Physical materials (SFP transceivers, AMIDs, etc.) not available on site
        "Material Missing": [
            "material missing", "SFP missing", "AMID not available",
            "missing material", "Material", "didn't had material"
        ],
        # Site was powered down or shut down during scheduled integration window
        "Site Down": [
            "site was down", "shut down", "Site was shut down",
            "couldn't lock the live cells", "site down"
        ],
        # Site has unusual complexity (MW chain, generator dependency, etc.)
        "Site Complexity": [
            "middle of a microwave chain", "downstream sites",
            "didn't have CRs", "generator", "complexity", "aborted"
        ],
        # iNTP (initial Network Time Protocol) synchronisation not completed
        "iNTP Not Ready": [
            "iNTP still not completed", "iNTP not ready"
        ],
        # BH readiness was checked but using the wrong method/criteria
        "BH Check Error": [
            "BH readiness", "checked wrongly", "BH Check Error",
            "wrong method", "mistake"
        ],
        # BH actualisation status field filled incorrectly in the system
        "BH Status Issue": [
            "BH Actualization filled as pending", "BH pending in field",
            "BH status", "status issue"
        ]
    },

    # ---- Communication & Response sub-categories ----
    "Communication & Response": {
        # Engineer or PM replied late to a GC/FE query
        "Delayed Response": [
            "Delay in reply", "Delayed reply", "delayed response",
            "GC query", "Replied back", "when PM asked",
            "delayed reply to GC", "delayed reply to FE"
        ],
        # A deliverable (EOD report, etc.) was delivered hours late
        "Delayed Deliverable": [
            "FE waited", "waiting for", "hrs to get EOD",
            "3hrs", "4hrs", "delayed deliverable"
        ],
        # Engineer did not proactively communicate status or delays
        "No Proactive Update": [
            "pro-actively", "questioned why", "such queries",
            "getting questioned over delays", "proactive update"
        ],
        # A known problem was not communicated early enough to prevent waste
        "No Proactive Communication": [
            "could have communicated", "not completed on Friday",
            "Had to cancel", "Else could have", "proactive communication"
        ],
        # The same question is asked repeatedly due to staff rotation or gaps
        "Repeated Queries": [
            "same queries", "new IX BO engg", "each day",
            "unnecessary delays", "repeated queries"
        ],
        # Schedule or status was never communicated to the field team
        "No Communication": [
            "No information", "schedule from PM to FE",
            "not communicated", "no communication"
        ],
        # FE lacks the knowledge or competency to perform the task
        "Training Issue": [
            "FE doesn't know", "Competency issue", "even after explaining",
            "training issue", "doesn't know how"
        ]
    },

    # ---- Nesting & Tool Errors sub-categories ----
    "Nesting & Tool Errors": {
        # Site nested in wrong state (e.g. NSA when only SA is allowed)
        "Wrong Nest Type": [
            "nested as NSA", "nested as SA", "nested as NSI",
            "wrong nest type", "which is not allowed", "NSA", "NSI",
            "only SA is allowed", "nested NSI from SI"
        ],
        # Nest was extended beyond the allowed window or during follow-up
        "Improper Extension": [
            "nest extended", "extended the nest", "nest extension",
            "during Follow Up", "improper extension"
        ],
        # Site was worked on without being nested in the management tool
        "Missing Nesting": [
            "not Nested", "was not added in Nesting tool",
            "without Nesting", "missing nesting"
        ],
        # Parameter changes caused measurable RF impact (e.g. UL traffic drop)
        "Parameter Impact": [
            "NR UL traffic volume reduction", "shifted to LTE",
            "post site modification", "parameter impact"
        ],
        # KPI degradation detected after nesting/integration activity
        "KPI Impact": [
            "TPUT decrease", "Cluster KPI", "KPI impact",
            "degraded", "post"
        ],
        # Neighbouring site performance affected by this activity
        "Neighbor Impact": [
            "starts to deviate", "MS2 site", "on-aired",
            "neighbor impact"
        ],
        # A requested feature activation (VoNR, etc.) was not performed
        "Missed Activation": [
            "VONR was not activated", "despite requested",
            "only NRPCI was shared", "missed activation"
        ],
        # KPI degradation occurring after the site went on-air
        "Post-OA Degradation": [
            "congestion", "low Throughput", "AFR", "DCR degraded",
            "HSI low speed", "after site replacement", "post-OA"
        ],
        # External alarm naming in the system is incorrect
        "Alarm Naming": [
            "External alarm", "naming is incorrect", "SURGE SUPPRESSOR",
            "alarm naming"
        ],
        # Post-integration audit was performed significantly late (e.g. 5 days)
        "Delayed Audit": [
            "Audit after 5days", "delay in Audit", "delayed audit"
        ],
        # Work had to be redone due to errors or unclear requirements
        "Rework": [
            "rework", "PM again requesting", "why", "was deleted",
            "no clarity", "SCF prep and IX work"
        ],
        # Physical hardware fault (GPS SFP, PTP sync, RET antenna controller)
        "HW Issue": [
            "GPS SFP", "PTP sync issue", "RET Antenna control failure",
            "hardware fault", "antenna will be replaced", "HW issue"
        ],
        # Cells were unlocked before all post-integration steps were complete
        "Premature Unlock": [
            "unlocked the cells", "change the cell status to UUU",
            "pending with call test", "premature unlock"
        ],
        # Integration cancelled by external party (TMO, holiday, etc.)
        "External Cancel": [
            "Site was cancelled", "TMO didnt want", "Holiday week",
            "external cancel"
        ]
    }
}

# ==========================================
# CATEGORY KEYWORDS - Extended taxonomy for hybrid classification
#
# STRUCTURE:
# Each category has three layers used by the hybrid classifier:
#   "primary"  -> Single keywords (fast initial filter)
#   "phrases"  -> Multi-word phrases (higher precision than single words)
#   "patterns" -> Python regex patterns (catches morphological variations)
#
# The classifier computes a weighted score across all three layers:
#   score = w1 * primary_matches + w2 * phrase_matches + w3 * pattern_matches
# This hybrid approach is more robust than pure embeddings for short,
# jargon-heavy telecom text while still being deterministic and auditable.
# ==========================================
CATEGORY_KEYWORDS = {
    "Scheduling & Planning": {
        "primary": [
            "schedule", "TI", "Ti", "planned", "unplanned", "calendar",
            "date", "time", "weekend", "holiday", "closeout", "close out"
        ],
        "phrases": [
            "not schedule", "not scheduled", "schedule missing", "no schedule",
            "without schedule", "TI entry missing", "not in TI", "not in Ti",
            "schedule not followed", "incorrect schedule", "FE on site",
            "FE logged", "logged for IX", "logged for support", "logged for TS",
            "ticket in closeout", "moved to closeout", "wrong bucket",
            "site not schedule for", "FE logged in for integration",
            "not schedule on weekend", "schedule after FE logged",
            "site was not schedule in Ti for integration",
            "ticket was in closeout FE logged"
        ],
        "patterns": [
            r"site.*not.*schedul",
            r"FE.*logged.*not.*schedul",
            r"Ti.*entry.*missing",
            r"ticket.*closeout",
            r"not.*schedul.*Ti"
        ]
    },
    "Documentation & Reporting": {
        "primary": [
            "snapshot", "snap", "screenshot", "missing", "wrong", "incorrect",
            "attachment", "logs", "report", "email", "mail", "subject",
            "CBN", "E911", "RTT", "EOD", "summary", "table"
        ],
        "phrases": [
            "missing snapshot", "snapshot missing", "wrong site ID",
            "different site", "forgot to attach", "forgot to paste",
            "missing in mail", "CBN output missing", "incomplete snapshot",
            "wrong attachment", "missing logs", "PSAP contact missing",
            "summary table missing", "Live CT missing", "E911 Go mail",
            "E911 complete mail", "detailed site view", "lemming snap",
            "engineer forgot to paste", "Table with PSAP missing"
        ],
        "patterns": [
            r"missing.*snapshot",
            r"wrong.*site.*ID",
            r"snapshot.*missing",
            r"E911.*missing",
            r"CBN.*missing"
        ]
    },
    "Validation & QA": {
        "primary": [
            "missed", "miss", "check", "validation", "verify", "audit",
            "precheck", "postcheck", "VSWR", "RSSI", "RTWP", "KPI",
            "pass", "fail", "detected", "captured", "escalated"
        ],
        "phrases": [
            "missed to check", "missed to report", "missed to capture",
            "missed to create", "missed to escalate", "not detected",
            "not identified", "wrong pass fail", "incomplete validation",
            "degradation not detected", "issue not escalated",
            "alarm captured but not", "cells in unsync", "RSSI imbalance",
            "engineer missed", "overlooked during", "not verified",
            "VSWR marked as Fail but actual Pass"
        ],
        "patterns": [
            r"missed.*to.*check",
            r"not.*detected",
            r"missed.*report",
            r"incomplete.*validation",
            r"wrong.*pass.*fail"
        ]
    },
    "Process Compliance": {
        "primary": [
            "process", "SOP", "guideline", "escalate", "NTAC", "distro",
            "approval", "without", "skip", "bypass", "procedure", "step"
        ],
        "phrases": [
            "released without", "supported without", "without BH actualized",
            "without backhaul", "escalated to NTAC", "wrong distro",
            "not following process", "skipped step", "bypassed validation",
            "proceeded without", "wrong bucket", "improper escalation",
            "not supposed to", "against guideline", "ticket in wrong",
            "escalated to NTAC while not supposed to",
            "escalated to different vendor", "not following SOP"
        ],
        "patterns": [
            r"without.*BH.*actual",
            r"without.*backhaul",
            r"escalat.*NTAC",
            r"wrong.*distro",
            r"releas.*without"
        ]
    },
    "Configuration & Data Mismatch": {
        "primary": [
            "mismatch", "port matrix", "PMX", "SCF", "CIQ", "RFDS", "TAC",
            "RET", "naming", "One-T", "OneT", "configuration", "config"
        ],
        "phrases": [
            "port matrix mismatch", "RET naming issue", "RET naming wrong",
            "TAC mismatch", "SCF mismatch", "CIQ mismatch", "RFDS mismatch",
            "need updated port matrix", "as per site configuration",
            "as per port matrix", "detected on site but", "RET swap",
            "extra 2 in naming", "I missing", "N66 not present",
            "RIOT red", "not matching", "doesn't match",
            "SCF and RFDS mismatch", "CIQ and SCF not matching"
        ],
        "patterns": [
            r"port.*matrix.*mismatch",
            r"RET.*naming",
            r"TAC.*mismatch",
            r"SCF.*CIQ.*mismatch",
            r"need.*updated.*port"
        ]
    },
    "Site Readiness": {
        "primary": [
            "BH", "backhaul", "actualized", "ready", "MW", "microwave",
            "material", "missing", "SFP", "cancelled", "down", "pending"
        ],
        "phrases": [
            "BH not actualized", "backhaul not ready", "BH not ready",
            "MW not ready", "microwave not ready", "site not ready",
            "material missing", "SFP missing", "site was down",
            "cancelled due to", "could not be integrated", "FE on site but BH",
            "pending actualization", "BH acceptance pending", "transmission not ready",
            "BH not ready in MB", "MW link was not ready"
        ],
        "patterns": [
            r"BH.*not.*actual",
            r"backhaul.*not.*ready",
            r"MW.*not.*ready",
            r"material.*missing",
            r"site.*not.*ready"
        ]
    },
    "Communication & Response": {
        "primary": [
            "delay", "delayed", "reply", "response", "waiting", "waited",
            "hours", "follow-up", "reminder", "communicated", "query"
        ],
        "phrases": [
            "delayed reply", "delayed response", "delay in reply",
            "waited for hours", "FE waited", "GC query not replied",
            "replied when PM asked", "no reply from", "follow-up required",
            "multiple follow-ups", "not communicated", "communication gap",
            "wrong distro", "proactive update missing", "questioned over delays",
            "replied back when PM asked", "reminder sent"
        ],
        "patterns": [
            r"delay.*reply",
            r"delay.*response",
            r"waited.*hours",
            r"no.*reply",
            r"communication.*gap"
        ]
    },
    "Nesting & Tool Errors": {
        "primary": [
            "nest", "nested", "nesting", "NSA", "SA", "NSI", "SI",
            "RIOT", "FCI", "TI", "tool", "updated", "market", "guideline"
        ],
        "phrases": [
            "nested as NSA", "nested as SA", "nested as NSI",
            "wrong nest type", "nest extended", "nest extension",
            "not allowed in market", "market guideline", "RIOT red",
            "RIOT mismatch", "tool not updated", "FCI not updated",
            "site not unnested", "lemming validation", "without RIOT",
            "TAC mismatch RIOT", "RF config mismatch", "OSS mismatch"
        ],
        "patterns": [
            r"nested.*as.*NSA",
            r"nested.*as.*NSI",
            r"RIOT.*red",
            r"nest.*extend",
            r"not.*allow.*market"
        ]
    }
}

# ==========================================
# ROOT CAUSE CATEGORIES
#
# Maps free-text root-cause descriptions from the source spreadsheet to
# one of 7 standardised buckets.  Used by _add_root_cause_classification()
# in the scoring engine.  Each bucket has a list of lowercase keyword
# fragments; if any fragment appears in the root-cause text, that bucket
# is assigned.  First match wins (ordered dict).
# ==========================================
ROOT_CAUSE_CATEGORIES = {
    'Human Error': ['human error', 'operator error', 'manual error', 'user error', 'mistake'],
    'External Party': ['non amdocs', 'external', 'vendor', 'third party', '3rd party', 'customer caused'],
    'Process Gap': ['process gap', 'process issue', 'sop missing', 'procedure gap', 'workflow issue'],
    'System/Technical': ['system error', 'technical issue', 'software bug', 'hardware failure', 'system failure'],
    'Training Gap': ['training', 'knowledge gap', 'skill gap', 'lack of training'],
    'Communication': ['communication', 'miscommunication', 'information gap', 'handoff issue'],
    'Resource': ['resource', 'understaffed', 'capacity', 'bandwidth'],
}

# ==========================================
# COLUMN NAME CONSTANTS
#
# These constants map logical field names used throughout the codebase to
# the actual column headers in the source Strategic_Report.xlsx / input
# spreadsheet.  By centralising them here, a schema change in the input
# file only requires updating these strings -- no grep-and-replace across
# dozens of modules.
# ==========================================
COL_SEVERITY = 'tickets_data_severity'             # Severity level: Critical / Major / Minor
COL_TYPE = 'tickets_data_type'                      # Record type: Escalations / Concerns / Lessons Learned
COL_ORIGIN = 'tickets_data_escalation_origin'       # Origin: External / Internal
COL_IMPACT = 'tickets_data_business_impact'         # Business impact: High / Low / None
COL_SUMMARY = 'tickets_data_issue_summary'          # Free-text issue description (main classifier input)
COL_CATEGORY = 'tickets_data_issue_category'        # Original human-assigned category (if any)
COL_DATETIME = 'tickets_data_issue_datetime'        # Timestamp when the issue was raised
COL_CLOSE_DATE = 'tickets_data_resolution_datetime' # Timestamp when the issue was resolved
COL_RESOLUTION_DATE = 'tickets_data_resolution_datetime'  # Alias for COL_CLOSE_DATE (used by metrics)
COL_RESOLUTION_NOTES = 'tickets_data_commentsresolution_plan'  # Free-text resolution notes / plan
COL_ENGINEER = 'tickets_data_engineer_name'         # Name of the assigned engineer
COL_LOB = 'tickets_data_lob'                        # Line of Business (organisational unit)
COL_LESSON_TITLE = 'tickets_data_lessons_learned_preventive_actions'  # Lesson learned text
COL_LESSON_STATUS = 'tickets_data_current_status'   # Current status of the lesson/action
COL_ROOT_CAUSE = 'tickets_data_root_cause'          # Free-text root cause description
COL_RECURRENCE_RISK = 'tickets_data_risk_for_recurrence'  # PM-assessed recurrence risk

# Minimum columns required for the scoring engine to function.
# If any of these are missing from the input DataFrame, validate_columns()
# will raise an error before processing begins.
REQUIRED_COLUMNS = [COL_SEVERITY, COL_TYPE, COL_ORIGIN]
