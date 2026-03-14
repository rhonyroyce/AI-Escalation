"""
Synthetic ticket data generator for integration testing.

Generates a 50-row Excel file with all required COL_* columns from config.py,
realistic values, and deliberate patterns for testing classification,
recidivism detection, and scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


# The 8 classification categories from config.py
CATEGORIES = [
    "Scheduling & Planning",
    "Documentation & Reporting",
    "Validation & QA",
    "Process Compliance",
    "Configuration & Data Mismatch",
    "Site Readiness",
    "Communication & Response",
    "Nesting & Tool Errors",
]

# Ticket summaries designed to match each category (at least 2 per category)
# Plus 3 near-duplicate pairs for recidivism detection
TICKET_SUMMARIES = [
    # Scheduling & Planning (indices 0-5)
    ("Site was not scheduled in TI for integration on planned date",
     "Scheduling & Planning"),
    ("FE logged for IX on wrong date, schedule not followed per TI entry",
     "Scheduling & Planning"),
    ("Calendar conflict - site not scheduled, TI entry missing for weekend work",
     "Scheduling & Planning"),
    ("Unplanned site visit due to scheduling error in closeout bucket",
     "Scheduling & Planning"),
    # DUPLICATE of index 0 (recidivism trigger)
    ("Site was not scheduled in TI system for integration on the planned date",
     "Scheduling & Planning"),
    # DUPLICATE of index 1 (recidivism trigger)
    ("FE logged for IX on the wrong date schedule was not followed per TI",
     "Scheduling & Planning"),

    # Documentation & Reporting (indices 6-8)
    ("Missing CBN snapshots and documentation not uploaded to SharePoint",
     "Documentation & Reporting"),
    ("Pre-integration snapshots not captured, CBN output report missing",
     "Documentation & Reporting"),
    ("Incomplete documentation package - no RF snapshots or test results attached",
     "Documentation & Reporting"),

    # Validation & QA (indices 9-11)
    ("Post-integration validation failed due to missing KPI checks on sectors",
     "Validation & QA"),
    ("QA audit revealed failed acceptance criteria on antenna alignment test",
     "Validation & QA"),
    ("Validation step skipped during integration, KPI thresholds not met",
     "Validation & QA"),

    # Process Compliance (indices 12-15)
    ("SOP not followed during site access - vendor arrived without proper authorization",
     "Process Compliance"),
    ("Process compliance violation: work performed without approved MOC permit",
     "Process Compliance"),
    ("Third party contractor did not follow safety protocol at tower site",
     "Process Compliance"),
    # DUPLICATE of index 12 (recidivism trigger)
    ("SOP not followed during site access vendor arrived without authorization",
     "Process Compliance"),

    # Configuration & Data Mismatch (indices 16-19)
    ("RF antenna configuration mismatch between design and as-built parameters",
     "Configuration & Data Mismatch"),
    ("NMS configuration error causing false alarm alerts on sector B",
     "Configuration & Data Mismatch"),
    ("Data mismatch between CIQ spreadsheet and actual site configuration values",
     "Configuration & Data Mismatch"),
    ("Integration parameter mismatch - wrong software version loaded on RRU",
     "Configuration & Data Mismatch"),

    # Site Readiness (indices 20-23)
    ("Power supply unit failure at cell tower site causing outage",
     "Site Readiness"),
    ("Site not ready for integration - missing ground bar and surge protector",
     "Site Readiness"),
    ("Fiber optic cable cut by contractor during excavation at site ABC",
     "Site Readiness"),
    ("Weather damage to outdoor equipment cabinet, site access blocked by debris",
     "Site Readiness"),

    # Communication & Response (indices 24-27)
    ("Communication breakdown between engineering team and field operations crew",
     "Communication & Response"),
    ("No response from site contact after multiple escalation attempts",
     "Communication & Response"),
    ("Handoff miscommunication between day and night shift engineers on critical issue",
     "Communication & Response"),
    ("Delayed response from vendor support team causing extended site downtime",
     "Communication & Response"),

    # Nesting & Tool Errors (indices 28-31)
    ("TEMS tool error during drive test - software crash lost measurement data",
     "Nesting & Tool Errors"),
    ("Nesting tool failed to import neighbor list correctly from CIQ file",
     "Nesting & Tool Errors"),
    ("RAN tool configuration error caused incorrect PCI assignment to sectors",
     "Nesting & Tool Errors"),
    ("Drive test tool malfunction during optimization, data collection interrupted",
     "Nesting & Tool Errors"),

    # Additional mixed tickets to reach 50 rows (indices 32-49)
    ("Antenna swap delayed due to missing equipment from vendor shipment",
     "Process Compliance"),
    ("Backhaul transmission link failure causing site isolation event",
     "Site Readiness"),
    ("Integration checklist incomplete - steps 5-8 not signed off by lead",
     "Validation & QA"),
    ("Wrong antenna model installed at site per design specification mismatch",
     "Configuration & Data Mismatch"),
    ("Site access denied by building management due to expired access permit",
     "Process Compliance"),
    ("RF optimization not completed before site acceptance sign-off deadline",
     "Scheduling & Planning"),
    ("Test results not documented in standard CBN report format template",
     "Documentation & Reporting"),
    ("Network management system alarm correlation failure for microwave link",
     "Nesting & Tool Errors"),
    ("Crew dispatch coordination failure between NOC and field engineering team",
     "Communication & Response"),
    ("Pre-integration checklist validation items not completed before cutover",
     "Validation & QA"),
    ("Site power backup generator test failed during routine maintenance check",
     "Site Readiness"),
    ("Scheduling system did not reflect updated timeline after change request",
     "Scheduling & Planning"),
    ("Post-integration drive test data not uploaded to centralized dashboard",
     "Documentation & Reporting"),
    ("CIQ parameter entry error in azimuth and tilt values for sector C",
     "Configuration & Data Mismatch"),
    ("Safety compliance issue - harness inspection certificate expired for tower crew",
     "Process Compliance"),
    ("Remote monitoring tool connectivity loss to site management interface",
     "Nesting & Tool Errors"),
    ("Escalation email not sent to regional manager within SLA response window",
     "Communication & Response"),
    ("Site grounding verification incomplete before equipment energization",
     "Site Readiness"),
]


def generate_synthetic_tickets(output_path, n_rows=50, seed=42):
    """Generate a synthetic Excel file with n_rows tickets.

    Args:
        output_path: Path to write the Excel file.
        n_rows: Number of rows (default 50, uses first n_rows from TICKET_SUMMARIES).
        seed: Random seed for reproducibility.

    Returns:
        Path to the created Excel file.
    """
    rng = np.random.RandomState(seed)
    output_path = Path(output_path)

    # Use the pre-defined summaries (50 entries)
    summaries_and_cats = TICKET_SUMMARIES[:n_rows]

    # Base date for generating dates within last 90 days
    base_date = datetime(2026, 3, 1)

    severities = ['Critical', 'Major', 'Minor']
    severity_weights = [0.2, 0.5, 0.3]

    types = ['Escalations', 'Concerns', 'Lessons Learned']
    type_weights = [0.5, 0.3, 0.2]

    origins = ['External', 'Internal']
    impacts = ['High', 'Low', 'None']

    engineers = [
        'John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Williams',
        'Carlos Martinez', 'Sarah Chen', 'Mike O\'Brien', 'Priya Patel',
    ]

    lobs = ['Wireless', 'Fiber', 'IT Systems', 'Transport', 'Core Network']

    issue_categories = [
        'Network Integration', 'Site Maintenance', 'Equipment Failure',
        'Process Issue', 'Documentation Gap', 'RF Optimization',
        'Configuration Change', 'Vendor Management',
    ]

    statuses = ['Open', 'In Progress', 'Resolved', 'Closed']
    status_weights = [0.15, 0.15, 0.4, 0.3]

    root_causes = [
        'Human error during configuration',
        'Vendor equipment defect',
        'Process gap in handoff procedure',
        'System error in management tool',
        'Training gap for new procedure',
        'Communication breakdown between teams',
        'Resource shortage on field crew',
        'External party scheduling conflict',
    ]

    recurrence_risks = ['High', 'Medium', 'Low']

    rows = []
    for i in range(n_rows):
        summary, _expected_cat = summaries_and_cats[i]

        # Random date within last 90 days
        days_ago = rng.randint(1, 91)
        issue_date = base_date - timedelta(days=days_ago)

        # Resolution date: some tickets are resolved, some are not
        status = rng.choice(statuses, p=status_weights)
        if status in ('Resolved', 'Closed'):
            resolution_hours = rng.uniform(4, 120)
            resolution_date = issue_date + timedelta(hours=resolution_hours)
        else:
            resolution_hours = None
            resolution_date = None

        severity = rng.choice(severities, p=severity_weights)

        row = {
            'tickets_data_id': f'TKT-{2024000 + i}',
            'tickets_data_issue_summary': summary,
            'tickets_data_issue_category': rng.choice(issue_categories),
            'tickets_data_severity': severity,
            'tickets_data_type': rng.choice(types, p=type_weights),
            'tickets_data_escalation_origin': rng.choice(origins),
            'tickets_data_business_impact': rng.choice(impacts),
            'tickets_data_issue_datetime': issue_date,
            'tickets_data_resolution_datetime': resolution_date,
            'tickets_data_engineer_name': rng.choice(engineers),
            'tickets_data_lob': rng.choice(lobs),
            'tickets_data_commentsresolution_plan': (
                f'Resolved by addressing root cause: {rng.choice(root_causes)}'
                if status in ('Resolved', 'Closed') else ''
            ),
            'tickets_data_lessons_learned_preventive_actions': (
                f'Implement checklist step for {summary[:40].lower()}'
                if rng.random() > 0.5 else ''
            ),
            'tickets_data_current_status': status,
            'tickets_data_root_cause': rng.choice(root_causes),
            'tickets_data_risk_for_recurrence': rng.choice(recurrence_risks),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, sheet_name='Raw Data')
    return output_path


def get_expected_categories():
    """Return the expected category for each synthetic ticket.

    Returns:
        List of (index, expected_category) tuples.
    """
    return [(i, cat) for i, (_, cat) in enumerate(TICKET_SUMMARIES)]
