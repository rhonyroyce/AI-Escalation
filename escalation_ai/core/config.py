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
RESOLUTION_FEEDBACK_FILE = "resolution_feedback.xlsx"  # Persistent human feedback for resolution times
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
# 8-category system with sub-categories optimized for telecom escalation analysis
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
# SUB-CATEGORIES - Detailed sub-types from Embedding.md analysis
# Based on 300+ error sample classification
# ==========================================
SUB_CATEGORIES = {
    "Scheduling & Planning": {
        "No TI Entry": [
            "not schedule in Ti", "TI entry missing", "not in TI", "not in Ti",
            "schedule missing in Ti", "no TI entry", "Ti entry missing",
            "Site not schedule in Ti", "not schedule for integration in Ti",
            "FE on site for integration", "site was not schedule"
        ],
        "Schedule Not Followed": [
            "schedule not followed", "Site schedule for", "but FE logged",
            "FE didn't login", "schedule for IX on", "but FE didn't",
            "Incorrect Schedule", "site schedule", "FE logged on"
        ],
        "Weekend Schedule Issue": [
            "weekend", "not site schedule on weekend", "over weekend",
            "not schedule on weekend", "site schedule on weekend"
        ],
        "Ticket Status Issue": [
            "closeout", "close out", "ticket in closeout", "Ticket was in closeout",
            "moved to closeout", "ticket moved", "Ticket is in closeout",
            "closeout bucket", "moved from closeout"
        ],
        "Premature Scheduling": [
            "without microwave ready", "without BH ready", "BH not ready",
            "schedule without", "MW not ready", "site not ready"
        ]
    },
    "Documentation & Reporting": {
        "Missing Snapshot": [
            "snapshot missing", "missing snapshot", "CBN snapshot missing",
            "snapshot is missing", "snap missing", "lemming snap missing",
            "validation snapshot missing", "CBN output snapshot missing"
        ],
        "Missing Attachment": [
            "missed to attach", "forgot to attach", "missing attachment",
            "not attached", "attachment missing", "Pre-check logs"
        ],
        "Incorrect Reporting": [
            "incorrectly", "RTT from IX vendor was also wrong",
            "didn't provide a clear view", "wrong which", "reporting"
        ],
        "Missing Logs": [
            "forgets to paste", "manual logs", "missing logs",
            "no logs populated", "logs missing"
        ],
        "Wrong Attachment": [
            "wrong attachment", "wrongly attached", "different site",
            "another site", "Detailed site view snap"
        ],
        "Wrong Site ID": [
            "wrong site ID", "different site id", "Wrong site id",
            "mistakenly placed a different site", "site ID missing",
            "Different site ID", "Different mail ID"
        ],
        "Incorrect Status": [
            "marked VSWR as Fail", "actual status is", "incorrect information",
            "marked as Pass", "but actual", "incorrect status",
            "Live call test details", "mentioned in summary"
        ],
        "Incomplete Snapshot": [
            "Incomplete lemming snapshot", "incomplete snapshot",
            "only 2 sectors", "sector was missing", "In-complete"
        ],
        "Data Mismatch": [
            "mentioned different", "as compared to", "mismatch",
            "Count of sectors mentioned different"
        ],
        "Missing Information": [
            "PSAP contact missing", "Table with PSAP", "missing",
            "not available", "missing in mail", "summary table missing",
            "Live CT missing", "Live CT CBN output"
        ],
        "Wrong Subject": [
            "subject line", "Subject should have been", "wrong subject"
        ],
        "Process Skip": [
            "without sending", "E911 completion mail circulated",
            "without E911 Go mail", "skipped"
        ],
        "Missing Deliverable": [
            "RTT wasn't released", "EOD released", "not released",
            "released later"
        ]
    },
    "Validation & QA": {
        "Premature Checkout": [
            "checked out GC", "Logged the GC out", "cells were locked",
            "premature checkout", "early checkout"
        ],
        "Incomplete Validation": [
            "without checking", "IX precheck without", "not updated",
            "incomplete validation", "BH fields", "MagentaBuilt"
        ],
        "Invalid CR": [
            "Invalid CR", "invalid CR manually", "wrong CR"
        ],
        "Ignored Anomaly": [
            "anomaly detection", "left max tilt", "even though it was captured",
            "ignored anomaly", "max tilt values"
        ],
        "Missed Issue": [
            "did not report", "Miss to capture", "missed to create",
            "Engineer did not identified", "missed to attach",
            "missed to report", "fiber issue", "SFP issue"
        ],
        "Wrong Denial": [
            "denied for Anchor site", "IX precheck denied",
            "wrong BH validation", "BH was already actualized"
        ],
        "Missed Activation": [
            "VONR is not activated", "not activated", "missed activation"
        ],
        "Missed Check": [
            "Miss to check", "Cell Status Verification",
            "cells were in Unsync", "missed to check"
        ],
        "No Escalation": [
            "not escalated", "but not escalated", "no escalation",
            "captured but not escalated"
        ],
        "Wrong KPIs": [
            "different set of KPIs", "wrong KPIs", "not aligned with the agreed"
        ],
        "Missed Degradation": [
            "degradation was not detected", "locked and showing zero traffic",
            "missed degradation", "not detected during"
        ],
        "Skipped Validation": [
            "moved to Post check without", "skipped validation",
            "without Riot lemming validation"
        ],
        "Wrong Tool Usage": [
            "AEHC swap report", "tool will only flag", "wrong tool",
            "Does Not Recognize Physical Swap"
        ],
        "Premature Denial": [
            "IX support was denied", "despite BH status", "citing the reason",
            "TI record was not updated", "denied support"
        ],
        "Incomplete Testing": [
            "missing L21 Gamma", "VoNR testing", "call test",
            "calls were completed and passed", "incomplete testing"
        ],
        "Missed Call Test": [
            "AWS3 E911 call test", "missed", "released full RTT",
            "caught during audit", "schedule FE for call test"
        ],
        "No Reversion": [
            "did not revert", "original values", "no reversion",
            "tilt settings", "not reverted"
        ],
        "Missed Tech": [
            "N6 was missed", "missed to add", "NSD site", "revisit for CT"
        ]
    },
    "Process Compliance": {
        "Missed Step": [
            "missed to share", "missed to unlock", "Engineer missed",
            "missed step", "didn't lock the cells"
        ],
        "No Escalation": [
            "not escalated to concerned", "created the issue ticket but not",
            "no escalation", "not escalated"
        ],
        "Wrong Escalation": [
            "escalated to different vendor", "wrong distro",
            "escalated to NTAC", "which is raised by client",
            "wrong escalation", "should not reach NTAC"
        ],
        "Wrong Bucket": [
            "not in correct bucket", "wrong bucket", "Preliminary design",
            "Ticket is not created", "ticket was in Design phase"
        ],
        "Process Violation": [
            "IX supported without", "without Backhaul acceptance",
            "released by PAG BO without", "BH not actualized",
            "process violation", "without backhaul"
        ],
        "Missing Ticket": [
            "ticket not created", "Ticket is not created", "PAG ticket",
            "not created for the site", "missing ticket"
        ],
        "Wrong Nest State": [
            "sector swap started by GC with site nested in NSI",
            "wrong nest state", "nested in"
        ],
        "Wrong File Used": [
            "some other file was used", "wrong file", "correct SCF",
            "during integration some other"
        ],
        "Missed Guidance": [
            "could have informed FE", "if PSAP was busy",
            "whitelisted in lieu of", "missed guidance"
        ],
        "Process Non-Compliance": [
            "adhere to the guidelines", "inquiries from",
            "Tool Request Submission Process", "non-compliance"
        ]
    },
    "Configuration & Data Mismatch": {
        "RET Naming": [
            "RET naming issue", "RET naming", "extra '2'",
            "inadvertently removed I", "RET naming mismatch",
            "naming having an extra", "I missing", "naming is incorrect"
        ],
        "RET Swap": [
            "RET naming swap", "swapped", "RET swap",
            "controlling unit", "Beta & Gamma"
        ],
        "RET Parameter": [
            "RET found Max", "max tilt", "RET parameter"
        ],
        "Port Matrix Mismatch": [
            "port matrix", "Port Matrix", "PMX", "Incorrect Port matrix",
            "port matrix mismatch", "as per port matrix", "wrong port matrix",
            "2 RET's defined per sector", "3 RET's per sector",
            "need updated port matrix", "Port Matrix is missing"
        ],
        "CIQ/SCF Mismatch": [
            "CIQ and SCF", "SCF mismatch", "CIQ mismatch",
            "Mismatch in Spectrum sheet", "SCF provided with",
            "CIQ provided not matching"
        ],
        "TAC Mismatch": [
            "TAC mismatch", "TAC showing mismatched", "RIOT red",
            "TAC was", "MB TAC", "RF Config and OSS"
        ],
        "RFDS Mismatch": [
            "RFDS", "Mismatch of SCF", "RFDS mismatch",
            "SCF and RFDS mismatch", "RFDS not mention"
        ],
        "One-T Mismatch": [
            "One-T", "OneT", "One-T not yet updated",
            "One-T mismatch", "IP exports", "non-SRAN"
        ],
        "Design Mismatch": [
            "Design change", "HW available on site", "ASIA",
            "RFDS with ASIB", "design mismatch"
        ],
        "Missing Documents": [
            "RFDS & Port Matrix is missing", "Port Matrix is missing in TI",
            "missing in TI", "missing documents"
        ],
        "Config Error": [
            "Config Error", "configuration error", "error due to",
            "NRPLMNSET not define", "enbid", "not define in SCF"
        ],
        "Missing Config": [
            "BWP missing", "feature was not activated", "missing config",
            "MOCN", "Scripts were not implemented", "external neighbour"
        ],
        "IP Mismatch": [
            "IP mismatch", "Gateway IP different", "wrong ip",
            "IP different", "One-T and SCF"
        ],
        "Spectrum Mismatch": [
            "ARFCN and bandwidth", "spectrum plan", "spectrum mismatch"
        ]
    },
    "Site Readiness": {
        "BH Not Ready": [
            "BH not ready", "backhaul not ready", "BH was not ready",
            "backhaul acceptance", "BH readiness pending", "BH not actualized",
            "BH actualization", "backhaul readiness"
        ],
        "MW Not Ready": [
            "MW not ready", "microwave not ready", "Microwave link was not ready",
            "MW link was not ready", "incomplete microwave"
        ],
        "Material Missing": [
            "material missing", "SFP missing", "AMID not available",
            "missing material", "Material", "didn't had material"
        ],
        "Site Down": [
            "site was down", "shut down", "Site was shut down",
            "couldn't lock the live cells", "site down"
        ],
        "Site Complexity": [
            "middle of a microwave chain", "downstream sites",
            "didn't have CRs", "generator", "complexity", "aborted"
        ],
        "iNTP Not Ready": [
            "iNTP still not completed", "iNTP not ready"
        ],
        "BH Check Error": [
            "BH readiness", "checked wrongly", "BH Check Error",
            "wrong method", "mistake"
        ],
        "BH Status Issue": [
            "BH Actualization filled as pending", "BH pending in field",
            "BH status", "status issue"
        ]
    },
    "Communication & Response": {
        "Delayed Response": [
            "Delay in reply", "Delayed reply", "delayed response",
            "GC query", "Replied back", "when PM asked",
            "delayed reply to GC", "delayed reply to FE"
        ],
        "Delayed Deliverable": [
            "FE waited", "waiting for", "hrs to get EOD",
            "3hrs", "4hrs", "delayed deliverable"
        ],
        "No Proactive Update": [
            "pro-actively", "questioned why", "such queries",
            "getting questioned over delays", "proactive update"
        ],
        "No Proactive Communication": [
            "could have communicated", "not completed on Friday",
            "Had to cancel", "Else could have", "proactive communication"
        ],
        "Repeated Queries": [
            "same queries", "new IX BO engg", "each day",
            "unnecessary delays", "repeated queries"
        ],
        "No Communication": [
            "No information", "schedule from PM to FE",
            "not communicated", "no communication"
        ],
        "Training Issue": [
            "FE doesn't know", "Competency issue", "even after explaining",
            "training issue", "doesn't know how"
        ]
    },
    "Nesting & Tool Errors": {
        "Wrong Nest Type": [
            "nested as NSA", "nested as SA", "nested as NSI",
            "wrong nest type", "which is not allowed", "NSA", "NSI",
            "only SA is allowed", "nested NSI from SI"
        ],
        "Improper Extension": [
            "nest extended", "extended the nest", "nest extension",
            "during Follow Up", "improper extension"
        ],
        "Missing Nesting": [
            "not Nested", "was not added in Nesting tool",
            "without Nesting", "missing nesting"
        ],
        "Parameter Impact": [
            "NR UL traffic volume reduction", "shifted to LTE",
            "post site modification", "parameter impact"
        ],
        "KPI Impact": [
            "TPUT decrease", "Cluster KPI", "KPI impact",
            "degraded", "post"
        ],
        "Neighbor Impact": [
            "starts to deviate", "MS2 site", "on-aired",
            "neighbor impact"
        ],
        "Missed Activation": [
            "VONR was not activated", "despite requested",
            "only NRPCI was shared", "missed activation"
        ],
        "Post-OA Degradation": [
            "congestion", "low Throughput", "AFR", "DCR degraded",
            "HSI low speed", "after site replacement", "post-OA"
        ],
        "Alarm Naming": [
            "External alarm", "naming is incorrect", "SURGE SUPPRESSOR",
            "alarm naming"
        ],
        "Delayed Audit": [
            "Audit after 5days", "delay in Audit", "delayed audit"
        ],
        "Rework": [
            "rework", "PM again requesting", "why", "was deleted",
            "no clarity", "SCF prep and IX work"
        ],
        "HW Issue": [
            "GPS SFP", "PTP sync issue", "RET Antenna control failure",
            "hardware fault", "antenna will be replaced", "HW issue"
        ],
        "Premature Unlock": [
            "unlocked the cells", "change the cell status to UUU",
            "pending with call test", "premature unlock"
        ],
        "External Cancel": [
            "Site was cancelled", "TMO didnt want", "Holiday week",
            "external cancel"
        ]
    }
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
