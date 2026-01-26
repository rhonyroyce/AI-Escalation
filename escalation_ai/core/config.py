"""
Configuration settings for Escalation AI.

Contains all constants, weights, thresholds, and column mappings.
"""

# ==========================================
# AI MODEL CONFIGURATION
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:8b"  # Best semantic understanding for classification
GEN_MODEL = "gemma3:27b"  # Clean professional output, no thinking tags

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
# Comprehensive 11-category system for telecom escalations
# ==========================================
ANCHORS = {
    # Equipment & Infrastructure
    "RF & Antenna Issues": [
        "antenna misalignment", "vswr alarm", "rru fault", "radio failure", 
        "sector down", "rf interference", "antenna swap", "feeder cable",
        "bbu fault", "baseband", "carrier down", "cell outage"
    ],
    "Transmission & Backhaul": [
        "fiber cut", "microwave link down", "transmission failure", "mw fade",
        "ethernet fault", "capacity exhaust", "latency", "packet loss",
        "backhaul", "transport", "ipsec", "vpn down", "circuit down"
    ],
    "Power & Environment": [
        "power outage", "battery failure", "rectifier fault", "generator issue",
        "ac failure", "high temperature", "equipment smoke", "cooling failure",
        "ups fault", "breaker trip", "fuel empty", "solar panel", "cabinet alarm"
    ],
    
    # Access & Field Operations
    "Site Access & Logistics": [
        "keys missing", "gate locked", "access denied", "landlord issue",
        "permit expired", "security clearance", "escort required", "site unsafe",
        "site inaccessible", "road blocked", "no access", "tower climb"
    ],
    "Contractor & Vendor Issues": [
        "crew no show", "wrong crew", "incomplete work", "material shortage",
        "vendor delay", "subcontractor issue", "quality defect", "rework required",
        "parts missing", "tool shortage", "training gap", "crew late"
    ],
    
    # Technical & Software
    "Configuration & Integration": [
        "parameter mismatch", "wrong ip", "integration error", "script failed",
        "alarm suppressed", "neighbor list", "handover failure", "config rollback",
        "software bug", "feature activation", "license issue", "template error"
    ],
    "OSS/NMS & Systems": [
        "oss fault", "nms unreachable", "provisioning error", "database sync",
        "element manager", "snmp trap", "discovery failed", "inventory mismatch",
        "mediation", "ticketing system", "monitoring gap", "correlation failure",
        "nesting", "nest extension", "nsi", "si nesting", "cell planning", 
        "network planning", "pci conflict", "antenna tilt", "coverage optimization"
    ],
    
    # Process & Communication
    "Process & Documentation": [
        "paperwork missing", "approval delay", "incorrect data", "process gap",
        "sow mismatch", "change window", "notification failure", "handoff issue",
        "documentation error", "method statement", "safety violation", "audit finding"
    ],
    "Communication & Coordination": [
        "miscommunication", "escalation delay", "wrong contact", "no response",
        "scheduling conflict", "timezone issue", "language barrier", "email missed",
        "handover gap", "shift change", "notification delay", "stakeholder"
    ],
    
    # External Factors
    "Weather & Natural Events": [
        "flood", "hurricane", "storm", "lightning", "extreme heat", "ice", 
        "wind damage", "earthquake", "wildfire", "snow", "fog", "monsoon"
    ],
    "Third-Party & External": [
        "theft", "vandalism", "fiber cut by third party", "construction damage",
        "utility outage", "road closure", "civil unrest", "regulatory hold",
        "permit rejection", "zoning issue", "public complaint", "legal dispute"
    ]
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
COL_RESOLUTION_DATE = 'tickets_data_close_datetime'  # Alias for resolution date
COL_RESOLUTION_NOTES = 'tickets_data_resolution_notes'
COL_ENGINEER = 'tickets_data_engineer_name'
COL_LOB = 'tickets_data_lob'
COL_LESSON_TITLE = 'tickets_data_lessons_learned_title'
COL_LESSON_STATUS = 'tickets_data_lessons_learned_status'
COL_ROOT_CAUSE = 'tickets_data_root_cause'
COL_RECURRENCE_RISK = 'tickets_data_risk_for_recurrence_pm'

REQUIRED_COLUMNS = [COL_SEVERITY, COL_TYPE, COL_ORIGIN]
