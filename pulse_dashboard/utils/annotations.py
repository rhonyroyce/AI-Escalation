"""
Pulse Dashboard - Annotation Persistence
==========================================
JSON-backed annotation system for project-level notes and comments.
Used by: 6_Project_Details.py
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

ANNOTATIONS_FILE = Path(__file__).parent.parent / "annotations.json"

ANNOTATION_TYPES = ["Note", "Action Item", "Risk Flag", "Follow-up"]


def load_annotations() -> list[dict]:
    """Load all annotations from the JSON file."""
    if not ANNOTATIONS_FILE.exists():
        return []
    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_annotations(annotations: list[dict]) -> None:
    """Persist annotations list to the JSON file."""
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=2, default=str)


def get_project_annotations(
    project: str,
    week: Optional[int] = None,
    year: Optional[int] = None,
) -> list[dict]:
    """Get annotations for a specific project, optionally filtered by week/year."""
    all_annots = load_annotations()
    filtered = [a for a in all_annots if a.get("project") == project]
    if week is not None:
        filtered = [a for a in filtered if a.get("week") == week]
    if year is not None:
        filtered = [a for a in filtered if a.get("year") == year]
    return sorted(filtered, key=lambda a: a.get("timestamp", ""), reverse=True)


def add_annotation(
    project: str,
    week: int,
    year: int,
    text: str,
    author: str = "Analyst",
    annotation_type: str = "Note",
) -> dict:
    """Add a new annotation and persist to disk. Returns the new annotation."""
    annot = {
        "id": str(uuid.uuid4())[:8],
        "project": project,
        "week": week,
        "year": year,
        "author": author,
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "annotation_type": annotation_type,
    }
    annotations = load_annotations()
    annotations.append(annot)
    save_annotations(annotations)
    return annot


def delete_annotation(annot_id: str) -> bool:
    """Delete an annotation by ID. Returns True if found and removed."""
    annotations = load_annotations()
    before = len(annotations)
    annotations = [a for a in annotations if a.get("id") != annot_id]
    if len(annotations) < before:
        save_annotations(annotations)
        return True
    return False
