"""
Sample data fixtures for testing

This module provides sample data that mimics real escalation data
for use in unit tests.
"""

import pandas as pd
from pathlib import Path
import tempfile


def create_sample_escalation_data():
    """
    Create sample escalation data for testing.

    Returns:
        pd.DataFrame: Sample escalation data with typical columns
    """
    data = {
        'tickets_data_id': [1, 2, 3, 4, 5],
        'tickets_data_summary': [
            'RF antenna alignment issue causing signal degradation',
            'Power supply failure at cell site',
            'Fiber optic cable cut by contractor',
            'Configuration error in network management system',
            'Weather damage to outdoor equipment',
        ],
        'tickets_data_severity': ['Critical', 'Major', 'Critical', 'Minor', 'Major'],
        'tickets_data_type_1': ['Escalations', 'Escalations', 'Concerns', 'Escalations', 'Lessons'],
        'tickets_data_origin': ['External', 'Internal', 'External', 'Internal', 'External'],
        'tickets_data_impact': ['High', 'High', 'Low', 'None', 'High'],
        'tickets_data_engineer': ['John Smith', 'Jane Doe', 'John Smith', 'Bob Johnson', 'Jane Doe'],
        'tickets_data_lob': ['Wireless', 'Wireless', 'Fiber', 'IT Systems', 'Wireless'],
        'tickets_data_date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'tickets_data_resolution_hours': [24, 48, 72, 12, 36],
    }

    return pd.DataFrame(data)


def create_sample_excel_file(output_path=None):
    """
    Create a sample Excel file for testing file loading.

    Args:
        output_path: Optional path to save the file. If None, creates a temp file.

    Returns:
        Path: Path to the created Excel file
    """
    df = create_sample_escalation_data()

    if output_path is None:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xlsx', delete=False
        )
        output_path = temp_file.name
        temp_file.close()

    df.to_excel(output_path, index=False, sheet_name='Escalations')

    return Path(output_path)


def get_sample_ticket_text():
    """
    Get sample ticket descriptions for testing text processing.

    Returns:
        list: List of sample ticket descriptions
    """
    return [
        "RF antenna alignment issue causing signal degradation at site ABC123",
        "Power supply unit failed during routine maintenance at tower site",
        "Fiber optic cable was accidentally cut by third-party contractor",
        "Network management system configuration error causing alerts",
        "Severe weather caused physical damage to outdoor equipment cabinet",
        "Backhaul connectivity lost due to transmission line failure",
        "Site access denied by property owner delaying scheduled maintenance",
        "Vendor delivered wrong equipment model causing project delay",
        "Integration testing failed due to software version mismatch",
        "Communication breakdown between engineering and operations teams",
    ]


def get_sample_categories():
    """
    Get the expected category classifications for sample tickets.

    Uses the new 8-category system:
    - Scheduling & Planning
    - Documentation & Reporting
    - Validation & QA
    - Process Compliance
    - Configuration & Data Mismatch
    - Site Readiness
    - Communication & Response
    - Nesting & Tool Errors

    Returns:
        dict: Mapping of ticket index to expected category
    """
    return {
        0: "Configuration & Data Mismatch",  # RF antenna alignment
        1: "Site Readiness",                 # Power supply failure
        2: "Site Readiness",                 # Fiber cut - site issue
        3: "Configuration & Data Mismatch",  # NMS config error
        4: "Site Readiness",                 # Weather damage
        5: "Site Readiness",                 # Backhaul/transmission
        6: "Process Compliance",             # Site access issue
        7: "Process Compliance",             # Vendor issue
        8: "Configuration & Data Mismatch",  # Integration mismatch
        9: "Communication & Response",       # Communication breakdown
    }
