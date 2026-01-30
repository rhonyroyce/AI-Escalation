"""
Price Catalog System for financial impact calculation.
Excel-based pricing catalog for calculating financial impact of escalations.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill

from ..core.config import PRICE_CATALOG_FILE, DEFAULT_HOURLY_RATE, DEFAULT_DELAY_COST

logger = logging.getLogger(__name__)


class PriceCatalog:
    """
    Excel-based pricing catalog for calculating financial impact of escalations.
    Supports keyword patterns, category-based pricing, and severity multipliers.
    """
    
    def __init__(self, catalog_path: str = PRICE_CATALOG_FILE):
        self.catalog_path = catalog_path
        self.category_costs: Dict[str, Dict] = {}
        self.keyword_costs: List[Dict] = []
        self.severity_multipliers: Dict[str, float] = {}
        self.origin_premiums: Dict[str, float] = {}
        self.is_loaded = False
    
    def create_template(self) -> str:
        """Create a blank price catalog Excel template with sample data."""
        try:
            wb = Workbook()
            
            # Sheet 1: Instructions
            ws_instructions = wb.active
            ws_instructions.title = "Instructions"
            
            instructions = [
                ["PRICE CATALOG - INSTRUCTIONS"],
                [""],
                ["This workbook defines the financial impact calculations for escalation analysis."],
                [""],
                ["SHEETS:"],
                ["1. Category Costs - Base costs per escalation category"],
                ["2. Keyword Patterns - Regex patterns for specific cost overrides"],
                ["3. Severity Multipliers - Cost multipliers based on severity level"],
                ["4. Origin Premiums - Additional percentage costs based on origin type"],
                [""],
                ["FORMULA:"],
                ["Total_Impact = (Material + Labor × Rate + Delay) × Severity_Mult × (1 + Origin_Premium)"],
            ]
            
            for row in instructions:
                ws_instructions.append(row)
            
            ws_instructions['A1'].font = Font(bold=True, size=14)
            ws_instructions.column_dimensions['A'].width = 80
            
            # Sheet 2: Category Costs
            ws_category = wb.create_sheet("Category Costs")
            category_headers = ["Category", "Material_Cost", "Labor_Hours", "Hourly_Rate", "Delay_Cost_Per_Hour", "Notes"]
            ws_category.append(category_headers)
            
            # 8-category system optimized for telecom escalation analysis
            category_data = [
                ["Scheduling & Planning", 300, 2, 100, 150, "TI scheduling, FE coordination"],
                ["Documentation & Reporting", 200, 3, 100, 100, "Snapshots, E911, CBN reports"],
                ["Validation & QA", 500, 4, 125, 250, "Precheck, postcheck, VSWR validation"],
                ["Process Compliance", 400, 3, 125, 200, "SOP adherence, escalation process"],
                ["Configuration & Data Mismatch", 800, 6, 150, 400, "Port matrix, RET, TAC mismatch"],
                ["Site Readiness", 1500, 8, 150, 500, "BH actualization, MW readiness"],
                ["Communication & Response", 200, 2, 100, 150, "Delayed replies, follow-ups"],
                ["Nesting & Tool Errors", 600, 5, 125, 300, "NSA/NSI nesting, RIOT, FCI tools"],
                ["Unclassified", 500, 4, 125, 200, "Default for unknown"],
            ]
            
            for row in category_data:
                ws_category.append(row)
            
            self._style_header(ws_category, len(category_headers))
            ws_category.column_dimensions['A'].width = 30
            ws_category.column_dimensions['F'].width = 35
            
            # Sheet 3: Keyword Patterns
            ws_keywords = wb.create_sheet("Keyword Patterns")
            keyword_headers = ["Keyword_Pattern", "Category_Override", "Material_Cost", "Labor_Hours", "Priority", "Notes"]
            ws_keywords.append(keyword_headers)
            
            # Keyword patterns for cost overrides based on 8-category system
            keyword_data = [
                [".*BH.*not.*actual.*", "Site Readiness", 2000, 12, 1, "Backhaul not actualized"],
                [".*port.*matrix.*mismatch.*", "Configuration & Data Mismatch", 1500, 8, 1, "Port matrix issues"],
                [".*RET.*naming.*", "Configuration & Data Mismatch", 1200, 6, 2, "RET naming errors"],
                [".*not.*schedul.*", "Scheduling & Planning", 500, 4, 2, "Scheduling issues"],
                [".*snapshot.*missing.*", "Documentation & Reporting", 300, 2, 3, "Missing documentation"],
                [".*RIOT.*red.*", "Nesting & Tool Errors", 800, 5, 2, "RIOT validation failure"],
            ]
            
            for row in keyword_data:
                ws_keywords.append(row)
            
            self._style_header(ws_keywords, len(keyword_headers))
            
            # Sheet 4: Severity Multipliers
            ws_severity = wb.create_sheet("Severity Multipliers")
            severity_headers = ["Severity_Level", "Cost_Multiplier", "Description"]
            ws_severity.append(severity_headers)
            
            severity_data = [
                ["Critical", 2.5, "Network down, major outage"],
                ["High", 1.75, "Significant degradation"],
                ["Medium", 1.25, "Moderate impact"],
                ["Low", 1.0, "Minor issue, no multiplier"],
            ]
            
            for row in severity_data:
                ws_severity.append(row)
            
            self._style_header(ws_severity, len(severity_headers))
            
            # Sheet 5: Origin Premiums
            ws_origin = wb.create_sheet("Origin Premiums")
            origin_headers = ["Origin_Type", "Premium_Percentage", "Description"]
            ws_origin.append(origin_headers)
            
            origin_data = [
                ["Vendor", 0.15, "15% premium for vendor-caused issues"],
                ["Process", 0.05, "5% premium for internal process failures"],
                ["External", 0.20, "20% premium for external/uncontrollable factors"],
                ["Customer", 0.10, "10% premium for customer-initiated issues"],
                ["Technical", 0.0, "No premium for standard technical issues"],
            ]
            
            for row in origin_data:
                ws_origin.append(row)
            
            self._style_header(ws_origin, len(origin_headers))
            
            wb.save(self.catalog_path)
            logger.info(f"✓ Price catalog template created: {self.catalog_path}")
            return self.catalog_path
            
        except Exception as e:
            logger.error(f"Failed to create price catalog template: {e}")
            raise
    
    def _style_header(self, ws, col_count):
        """Apply header styling to a worksheet."""
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        for col in range(1, col_count + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = header_font
    
    def load_catalog(self) -> bool:
        """Load pricing data from Excel catalog."""
        if not os.path.exists(self.catalog_path):
            logger.warning(f"Price catalog not found: {self.catalog_path}")
            logger.info("Creating template price catalog...")
            self.create_template()
        
        try:
            wb = load_workbook(self.catalog_path, data_only=True)
            
            # Load Category Costs
            if "Category Costs" in wb.sheetnames:
                ws = wb["Category Costs"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        category = str(row[0]).strip()
                        self.category_costs[category] = {
                            'material_cost': float(row[1] or 0),
                            'labor_hours': float(row[2] or 0),
                            'hourly_rate': float(row[3] or DEFAULT_HOURLY_RATE),
                            'delay_cost_per_hour': float(row[4] or 0),
                        }
                logger.info(f"  → Loaded {len(self.category_costs)} category cost entries")
            
            # Load Keyword Patterns
            if "Keyword Patterns" in wb.sheetnames:
                ws = wb["Keyword Patterns"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        self.keyword_costs.append({
                            'pattern': str(row[0]).strip(),
                            'category_override': str(row[1] or '').strip(),
                            'material_cost': float(row[2] or 0),
                            'labor_hours': float(row[3] or 0),
                            'priority': int(row[4] or 99),
                        })
                self.keyword_costs.sort(key=lambda x: x['priority'])
                logger.info(f"  → Loaded {len(self.keyword_costs)} keyword pattern rules")
            
            # Load Severity Multipliers
            if "Severity Multipliers" in wb.sheetnames:
                ws = wb["Severity Multipliers"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        severity = str(row[0]).strip().lower()
                        self.severity_multipliers[severity] = float(row[1] or 1.0)
                logger.info(f"  → Loaded {len(self.severity_multipliers)} severity multipliers")
            
            # Load Origin Premiums
            if "Origin Premiums" in wb.sheetnames:
                ws = wb["Origin Premiums"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        origin = str(row[0]).strip().lower()
                        self.origin_premiums[origin] = float(row[1] or 0)
                logger.info(f"  → Loaded {len(self.origin_premiums)} origin premium rules")
            
            wb.close()
            self.is_loaded = True
            logger.info(f"✓ Price catalog loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load price catalog: {e}")
            return False
    
    def _match_keyword_pattern(self, text: str) -> Optional[Dict]:
        """Check if text matches any keyword pattern."""
        if not text:
            return None
        
        text_lower = text.lower()
        for rule in self.keyword_costs:
            try:
                if re.search(rule['pattern'], text_lower, re.IGNORECASE):
                    return rule
            except re.error:
                logger.warning(f"Invalid regex pattern: {rule['pattern']}")
        return None
    
    def calculate_financial_impact(
        self,
        category: str,
        severity: str = "Medium",
        origin: str = "Technical",
        description: str = "",
        delay_hours: float = None
    ) -> Dict[str, float]:
        """Calculate total financial impact for an escalation.

        Note: delay_hours is deprecated and ignored. Delay cost is now calculated
        as labor_hours × delay_cost_per_hour for consistency.
        """
        if not self.is_loaded:
            self.load_catalog()

        keyword_match = self._match_keyword_pattern(description)

        if keyword_match:
            material_cost = keyword_match['material_cost']
            labor_hours = keyword_match['labor_hours']
            hourly_rate = DEFAULT_HOURLY_RATE
            delay_cost_per_hour = self.category_costs.get(
                keyword_match.get('category_override', category), {}
            ).get('delay_cost_per_hour', 300)
        else:
            cat_costs = self.category_costs.get(category, self.category_costs.get('Unclassified', {}))
            material_cost = cat_costs.get('material_cost', 1000)
            labor_hours = cat_costs.get('labor_hours', 6)
            hourly_rate = cat_costs.get('hourly_rate', DEFAULT_HOURLY_RATE)
            delay_cost_per_hour = cat_costs.get('delay_cost_per_hour', 300)

        labor_cost = labor_hours * hourly_rate
        delay_cost = labor_hours * delay_cost_per_hour  # Use labor_hours, not delay_hours
        base_cost = material_cost + labor_cost + delay_cost
        
        severity_mult = self.severity_multipliers.get(severity.lower(), 1.0)
        origin_premium = self.origin_premiums.get(origin.lower(), 0.0)
        
        total_impact = base_cost * severity_mult * (1 + origin_premium)
        
        return {
            'material_cost': round(material_cost, 2),
            'labor_cost': round(labor_cost, 2),
            'delay_cost': round(delay_cost, 2),
            'base_cost': round(base_cost, 2),
            'severity_multiplier': severity_mult,
            'origin_premium': origin_premium,
            'total_impact': round(total_impact, 2),
            'keyword_match': keyword_match['pattern'] if keyword_match else None,
        }
    
    def get_catalog_summary(self) -> str:
        """Get summary of loaded pricing data for AI context."""
        if not self.is_loaded:
            return "Price catalog not loaded."
        
        summary_lines = [
            f"PRICE CATALOG SUMMARY:",
            f"- {len(self.category_costs)} category cost entries",
            f"- {len(self.keyword_costs)} keyword pattern rules",
            f"- {len(self.severity_multipliers)} severity multipliers",
            f"- {len(self.origin_premiums)} origin premium rules",
        ]
        
        return "\n".join(summary_lines)


# Global price catalog instance
_price_catalog: Optional[PriceCatalog] = None

def get_price_catalog() -> PriceCatalog:
    """Get or create the price catalog singleton."""
    global _price_catalog
    if _price_catalog is None:
        _price_catalog = PriceCatalog()
    return _price_catalog
