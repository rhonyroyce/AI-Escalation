"""
Configuration settings for Escalation AI.

Contains all constants, weights, thresholds, and column mappings.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)

# ==========================================
# GPU / RAPIDS CONFIGURATION
# ==========================================
USE_GPU = True  # Enable GPU acceleration when available (cuML, cuDF)
GPU_MEMORY_LIMIT = 0.8  # Max fraction of GPU memory to use (0.0-1.0)

# ==========================================
# VRAM DETECTION & MODEL SELECTION
# ==========================================
def get_gpu_vram_gb():
    """Detect GPU VRAM in GB. Returns 0 if no NVIDIA GPU found."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Get first GPU's VRAM in MB, convert to GB
            vram_mb = int(result.stdout.strip().split('\n')[0])
            return vram_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0

def select_models():
    """Select AI models based on available VRAM."""
    vram_gb = get_gpu_vram_gb()
    
    # Model tiers based on VRAM
    if vram_gb >= 20:
        # High VRAM (24GB+): Use largest models
        embed_model = "qwen3-embedding:8b"
        gen_model = "qwen3:30b"
        tier = "high"
    elif vram_gb >= 12:
        # Medium VRAM (16GB): Use medium models
        embed_model = "qwen3-embedding:4b"
        gen_model = "qwen3:14b"
        tier = "medium"
    else:
        # Low VRAM (<12GB) or CPU only: Use smallest models
        embed_model = "qwen3-embedding:0.6b"
        gen_model = "qwen3:8b"
        tier = "low"
    
    logger.info(f"[Config] Detected {vram_gb:.1f}GB VRAM → {tier} tier models")
    return embed_model, gen_model, vram_gb

# Auto-detect and set models
EMBED_MODEL, GEN_MODEL, DETECTED_VRAM_GB = select_models()

# ==========================================
# AI MODEL CONFIGURATION
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434"
# EMBED_MODEL and GEN_MODEL are set automatically above based on VRAM

# ==========================================
# STRATEGIC WEIGHTS (McKinsey Framework)
# ==========================================
WEIGHTS = {
    'BASE_SEVERITY': {'Critical': 100, 'Major': 50, 'Minor': 10, 'Default': 5},
    'TYPE_MULTIPLIER': {'Escalations': 1.5, 'Concerns': 1.0, 'Lessons Learned': 0.0},
    'ORIGIN_MULTIPLIER': {'External': 2.5, 'Internal': 1.0},
    'IMPACT_MULTIPLIER': {'High': 2.0, 'Low': 1.1, 'None': 1.0}
}

# ==========================================
# SIMILARITY & RECIDIVISM THRESHOLDS
# ==========================================
SIMILARITY_THRESHOLD_HIGH = 0.60   # High confidence match
SIMILARITY_THRESHOLD_MEDIUM = 0.50  # Medium confidence - worth flagging
KEYWORD_OVERLAP_THRESHOLD = 0.35    # 35% keyword overlap = likely related

# ==========================================
# CLASSIFICATION SETTINGS
# ==========================================
MIN_CLASSIFICATION_CONFIDENCE = 0.25  # Below this = "Unclassified"

# ==========================================
# RECIDIVISM PENALTY MULTIPLIERS
# ==========================================
RECIDIVISM_PENALTY_HIGH = 1.5      # 50% score increase for confirmed repeats
RECIDIVISM_PENALTY_MEDIUM = 1.25   # 25% score increase for possible repeats

# ==========================================
# FEEDBACK/REINFORCEMENT LEARNING SETTINGS
# ==========================================
FEEDBACK_FILE = "classification_feedback.xlsx"
FEEDBACK_WEIGHT = 0.4  # How much weight to give user feedback vs original anchors (0-1)
SIMILARITY_FEEDBACK_PATH = "similarity_feedback.json"

# ==========================================
# ML MODEL PATHS
# ==========================================
RECURRENCE_MODEL_PATH = "recurrence_model.pkl"
RECURRENCE_ENCODERS_PATH = "recurrence_encoders.pkl"
RESOLUTION_MODEL_PATH = "resolution_model.pkl"

# ==========================================
# OUTPUT DIRECTORIES
# ==========================================
from pathlib import Path
PLOT_DIR = Path("plots")  # Directory for generated charts

# ==========================================
# PRICE CATALOG SETTINGS
# ==========================================
PRICE_CATALOG_FILE = "price_catalog.xlsx"
DEFAULT_HOURLY_RATE = 20.0  # $/hour for labor if not specified
DEFAULT_DELAY_COST = 500.0  # $/hour of delay (SLA penalty, revenue loss)

# ==========================================
# REPORT METADATA
# ==========================================
REPORT_VERSION = "2.2"
REPORT_TITLE = "STRATEGIC FRICTION ANALYSIS"

# ==========================================
# ENGINEER & LOB THRESHOLDS
# ==========================================
ENGINEER_REPEAT_THRESHOLD = 3  # Flag engineers with 3+ issues
LOB_RISK_THRESHOLD = 5  # Flag LOBs with 5+ issues

# ==========================================
# UI COLORS
# ==========================================
MC_BLUE = '#004C97'

# ==========================================
# CATEGORIZATION ANCHORS
# 8-category system optimized for telecom escalation analysis
# Based on 300+ error sample analysis
# ==========================================
ANCHORS = {
    "Scheduling & Planning": [
        "schedule", "TI", "Ti", "planned", "unplanned", "calendar",
        "date", "time", "weekend", "holiday", "closeout", "close out",
        "not schedule", "not scheduled", "schedule missing", "no schedule",
        "without schedule", "TI entry missing", "not in TI", "not in Ti",
        "FE on site", "FE logged", "logged for IX", "logged for support",
        "ticket in closeout", "moved to closeout", "wrong bucket"
    ],
    "Documentation & Reporting": [
        "snapshot", "snap", "screenshot", "missing", "wrong", "incorrect",
        "attachment", "logs", "report", "email", "mail", "subject",
        "CBN", "E911", "RTT", "EOD", "summary", "table",
        "missing snapshot", "wrong site ID", "different site",
        "forgot to attach", "forgot to paste", "CBN output missing",
        "incomplete snapshot", "wrong attachment", "PSAP contact missing"
    ],
    "Validation & QA": [
        "missed", "miss", "check", "validation", "verify", "audit",
        "precheck", "postcheck", "VSWR", "RSSI", "RTWP", "KPI",
        "pass", "fail", "detected", "captured", "escalated",
        "missed to check", "missed to report", "missed to capture",
        "not detected", "wrong pass fail", "incomplete validation",
        "degradation not detected", "issue not escalated"
    ],
    "Process Compliance": [
        "process", "SOP", "guideline", "escalate", "NTAC", "distro",
        "approval", "without", "skip", "bypass", "procedure", "step",
        "released without", "supported without", "without BH actualized",
        "escalated to NTAC", "wrong distro", "not following process",
        "skipped step", "bypassed validation", "improper escalation"
    ],
    "Configuration & Data Mismatch": [
        "mismatch", "port matrix", "PMX", "SCF", "CIQ", "RFDS", "TAC",
        "RET", "naming", "One-T", "OneT", "configuration", "config",
        "port matrix mismatch", "RET naming issue", "RET naming wrong",
        "TAC mismatch", "SCF mismatch", "CIQ mismatch", "RFDS mismatch",
        "need updated port matrix", "RET swap", "RIOT red", "not matching"
    ],
    "Site Readiness": [
        "BH", "backhaul", "actualized", "ready", "MW", "microwave",
        "material", "missing", "SFP", "cancelled", "down", "pending",
        "BH not actualized", "backhaul not ready", "BH not ready",
        "MW not ready", "microwave not ready", "site not ready",
        "material missing", "SFP missing", "site was down",
        "cancelled due to", "transmission not ready"
    ],
    "Communication & Response": [
        "delay", "delayed", "reply", "response", "waiting", "waited",
        "hours", "follow-up", "reminder", "communicated", "query",
        "delayed reply", "delayed response", "delay in reply",
        "waited for hours", "FE waited", "GC query not replied",
        "no reply from", "follow-up required", "communication gap"
    ],
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
# CATEGORY KEYWORDS - Extended taxonomy for hybrid classification
# Includes primary keywords, semantic phrases, and regex patterns
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
# ==========================================
COL_SEVERITY = 'tickets_data_severity'
COL_TYPE = 'tickets_data_type_1'
COL_ORIGIN = 'tickets_data_escalation_origin'
COL_IMPACT = 'tickets_data_business_impact_pm'
COL_SUMMARY = 'tickets_data_issue_summary'
COL_CATEGORY = 'tickets_data_issue_category_1'
COL_DATETIME = 'tickets_data_issue_datetime'
COL_CLOSE_DATE = 'tickets_data_close_datetime'
COL_RESOLUTION_DATE = 'tickets_data_resolution_datetime'  # Actual resolution timestamp
COL_RESOLUTION_NOTES = 'tickets_data_resolution_notes'
COL_ENGINEER = 'tickets_data_engineer_name'
COL_LOB = 'tickets_data_lob'
COL_LESSON_TITLE = 'tickets_data_lessons_learned_title'
COL_LESSON_STATUS = 'tickets_data_lessons_learned_status'
COL_ROOT_CAUSE = 'tickets_data_root_cause'
COL_RECURRENCE_RISK = 'tickets_data_risk_for_recurrence_pm'

REQUIRED_COLUMNS = [COL_SEVERITY, COL_TYPE, COL_ORIGIN]
