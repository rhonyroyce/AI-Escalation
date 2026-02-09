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

from ..core.config import PRICE_CATALOG_FILE, DEFAULT_HOURLY_RATE, SUB_CATEGORIES

logger = logging.getLogger(__name__)


class PriceCatalog:
    """
    Excel-based pricing catalog for calculating financial impact of escalations.
    Single source of truth for all financial calculations.
    Supports keyword patterns, category/sub-category pricing, severity multipliers,
    and business multipliers - all loaded from price_catalog.xlsx.
    """

    def __init__(self, catalog_path: str = PRICE_CATALOG_FILE):
        self.catalog_path = catalog_path
        self.category_costs: Dict[str, Dict] = {}
        self.sub_category_costs: Dict[str, Dict[str, Dict]] = {}  # {category: {sub_cat: {...}}}
        self.keyword_costs: List[Dict] = []
        self.severity_multipliers: Dict[str, float] = {}
        self.origin_premiums: Dict[str, float] = {}
        self.business_multipliers: Dict[str, float] = {}
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
                ["2. Sub-Category Costs - Granular costs per sub-category"],
                ["3. Keyword Patterns - Regex patterns for specific cost overrides"],
                ["4. Severity Multipliers - Cost multipliers based on severity level"],
                ["5. Origin Premiums - Additional percentage costs based on origin type"],
                ["6. Business Multipliers - Configurable multipliers for business metrics"],
                [""],
                ["FORMULA:"],
                ["Total_Impact = (Material + Labor_Hours × Hourly_Rate) × Severity_Mult × (1 + Origin_Premium)"],
            ]
            
            for row in instructions:
                ws_instructions.append(row)
            
            ws_instructions['A1'].font = Font(bold=True, size=14)
            ws_instructions.column_dimensions['A'].width = 80
            
            # Sheet 2: Category Costs
            ws_category = wb.create_sheet("Category Costs")
            category_headers = ["Category", "Material_Cost", "Labor_Hours", "Hourly_Rate", "Notes"]
            ws_category.append(category_headers)

            # 8-category system - values match actual price_catalog.xlsx
            # $20/hr rate, $0 material (rework labor cost only)
            category_data = [
                ["Scheduling & Planning", 0, 2.0, 20, "TI scheduling, FE coordination"],
                ["Documentation & Reporting", 0, 1.5, 20, "Snapshots, E911, CBN reports"],
                ["Validation & QA", 0, 3.0, 20, "Precheck, postcheck, VSWR validation"],
                ["Process Compliance", 0, 2.5, 20, "SOP adherence, escalation process"],
                ["Configuration & Data Mismatch", 0, 4.0, 20, "Port matrix, RET, TAC mismatch"],
                ["Site Readiness", 0, 8.0, 20, "BH actualization, MW readiness"],
                ["Communication & Response", 0, 0.5, 20, "Delayed replies, follow-ups"],
                ["Nesting & Tool Errors", 0, 3.0, 20, "NSA/NSI nesting, RIOT, FCI tools"],
                ["Unclassified", 0, 0.2, 20, "Default for unknown"],
            ]
            
            for row in category_data:
                ws_category.append(row)
            
            self._style_header(ws_category, len(category_headers))
            ws_category.column_dimensions['A'].width = 30
            ws_category.column_dimensions['F'].width = 35

            # Sheet 3: Sub-Category Costs
            ws_subcat = wb.create_sheet("Sub-Category Costs")
            subcat_headers = ["Category", "Sub_Category", "Material_Cost", "Labor_Hours", "Hourly_Rate", "Notes"]
            ws_subcat.append(subcat_headers)

            # Sub-category costs - granular pricing per sub-type
            # Values use $20/hr base rate; labor hours vary by complexity
            subcat_data = [
                # Scheduling & Planning
                ["Scheduling & Planning", "No TI Entry", 0, 3.0, 20, "Site not scheduled in TI"],
                ["Scheduling & Planning", "Schedule Not Followed", 0, 2.0, 20, "FE logged on wrong day"],
                ["Scheduling & Planning", "Weekend Schedule Issue", 0, 3.0, 20, "Weekend scheduling error"],
                ["Scheduling & Planning", "Ticket Status Issue", 0, 1.5, 20, "Ticket in wrong bucket/closeout"],
                ["Scheduling & Planning", "Premature Scheduling", 0, 4.0, 20, "Scheduled without BH/MW ready"],
                # Documentation & Reporting
                ["Documentation & Reporting", "Missing Snapshot", 0, 1.5, 20, "CBN/validation snapshot missing"],
                ["Documentation & Reporting", "Missing Attachment", 0, 1.0, 20, "Pre-check logs not attached"],
                ["Documentation & Reporting", "Incorrect Reporting", 0, 2.0, 20, "RTT/data incorrectly reported"],
                ["Documentation & Reporting", "Wrong Site ID", 0, 2.0, 20, "Different site ID in mail"],
                ["Documentation & Reporting", "Incomplete Snapshot", 0, 1.5, 20, "Lemming snap missing sectors"],
                ["Documentation & Reporting", "Missing Information", 0, 1.5, 20, "PSAP/Live CT details missing"],
                ["Documentation & Reporting", "Wrong Attachment", 0, 1.0, 20, "Wrong file attached"],
                ["Documentation & Reporting", "Incorrect Status", 0, 2.5, 20, "Pass/Fail status wrong"],
                # Validation & QA
                ["Validation & QA", "Incomplete Validation", 0, 3.5, 20, "BH fields not checked"],
                ["Validation & QA", "Missed Issue", 0, 4.0, 20, "SFP/fiber issue not identified"],
                ["Validation & QA", "Missed Check", 0, 3.0, 20, "Cell status not verified"],
                ["Validation & QA", "No Escalation", 0, 2.5, 20, "Issue captured but not escalated"],
                ["Validation & QA", "Missed Degradation", 0, 5.0, 20, "KPI degradation not detected"],
                ["Validation & QA", "Wrong Tool Usage", 0, 2.5, 20, "AEHC swap report misused"],
                ["Validation & QA", "Incomplete Testing", 0, 3.5, 20, "E911/VoNR testing incomplete"],
                # Process Compliance
                ["Process Compliance", "Process Violation", 0, 4.0, 20, "IX without BH actualized"],
                ["Process Compliance", "Wrong Escalation", 0, 3.0, 20, "Escalated to wrong vendor/NTAC"],
                ["Process Compliance", "Wrong Bucket", 0, 2.5, 20, "Ticket in preliminary design"],
                ["Process Compliance", "Missed Step", 0, 3.0, 20, "Forgot to unlock/share precheck"],
                ["Process Compliance", "Missing Ticket", 0, 2.5, 20, "PAG ticket not created"],
                ["Process Compliance", "Process Non-Compliance", 0, 3.5, 20, "Guidelines not followed"],
                # Configuration & Data Mismatch
                ["Configuration & Data Mismatch", "Port Matrix Mismatch", 0, 5.0, 20, "RET count mismatch with PMX"],
                ["Configuration & Data Mismatch", "RET Naming", 0, 4.0, 20, "Extra/missing letter in naming"],
                ["Configuration & Data Mismatch", "RET Swap", 0, 5.0, 20, "Alpha/Beta RET swapped"],
                ["Configuration & Data Mismatch", "TAC Mismatch", 0, 4.0, 20, "TAC causing RIOT red"],
                ["Configuration & Data Mismatch", "CIQ/SCF Mismatch", 0, 5.0, 20, "CIQ and SCF not matching"],
                ["Configuration & Data Mismatch", "RFDS Mismatch", 0, 5.0, 20, "RFDS and SCF mismatch"],
                ["Configuration & Data Mismatch", "Missing Documents", 0, 3.0, 20, "RFDS/Port Matrix missing in TI"],
                ["Configuration & Data Mismatch", "Config Error", 0, 4.0, 20, "NRPLMNSET/BWP not defined"],
                # Site Readiness
                ["Site Readiness", "BH Not Ready", 0, 10.0, 20, "Backhaul not actualized in MB"],
                ["Site Readiness", "MW Not Ready", 0, 8.0, 20, "Microwave link not ready"],
                ["Site Readiness", "Material Missing", 0, 6.0, 20, "SFP/AMID not available"],
                ["Site Readiness", "Site Down", 0, 8.0, 20, "Site shut down during IX"],
                ["Site Readiness", "BH Status Issue", 0, 5.0, 20, "BH status incorrectly filled"],
                ["Site Readiness", "Site Complexity", 0, 8.0, 20, "MW chain/generator issues"],
                # Communication & Response
                ["Communication & Response", "Delayed Response", 0, 0.5, 20, "Late reply to GC/FE query"],
                ["Communication & Response", "Delayed Deliverable", 0, 1.0, 20, "FE waited 3-4hrs for EOD"],
                ["Communication & Response", "No Proactive Update", 0, 0.5, 20, "Not answering delay queries"],
                ["Communication & Response", "No Communication", 0, 0.5, 20, "Schedule not communicated to FE"],
                ["Communication & Response", "Training Issue", 0, 2.0, 20, "FE competency/knowledge gap"],
                # Nesting & Tool Errors
                ["Nesting & Tool Errors", "Wrong Nest Type", 0, 4.0, 20, "Nested as NSA when SA required"],
                ["Nesting & Tool Errors", "Improper Extension", 0, 3.0, 20, "Nest extended during follow-up"],
                ["Nesting & Tool Errors", "Missing Nesting", 0, 3.0, 20, "Site not nested before activity"],
                ["Nesting & Tool Errors", "HW Issue", 0, 5.0, 20, "GPS SFP/RET antenna failure"],
                ["Nesting & Tool Errors", "Rework", 0, 5.0, 20, "SCF prep and IX rework needed"],
                ["Nesting & Tool Errors", "Post-OA Degradation", 0, 6.0, 20, "Congestion/DCR after on-air"],
                ["Nesting & Tool Errors", "Delayed Audit", 0, 2.0, 20, "Audit done 5 days late"],
            ]

            for row in subcat_data:
                ws_subcat.append(row)

            self._style_header(ws_subcat, len(subcat_headers))
            ws_subcat.column_dimensions['A'].width = 30
            ws_subcat.column_dimensions['B'].width = 25
            ws_subcat.column_dimensions['F'].width = 30

            # Sheet 4: Keyword Patterns
            ws_keywords = wb.create_sheet("Keyword Patterns")
            keyword_headers = ["Keyword_Pattern", "Category_Override", "Material_Cost", "Labor_Hours", "Priority", "Notes"]
            ws_keywords.append(keyword_headers)

            # Keyword patterns for cost overrides - $20/hr, $0 material
            keyword_data = [
                [".*BH.*not.*actual.*", "Site Readiness", 0, 10, 1, "Backhaul not actualized"],
                [".*port.*matrix.*mismatch.*", "Configuration & Data Mismatch", 0, 6, 1, "Port matrix issues"],
                [".*RET.*naming.*", "Configuration & Data Mismatch", 0, 4, 2, "RET naming errors"],
                [".*not.*schedul.*", "Scheduling & Planning", 0, 3, 2, "Scheduling issues"],
                [".*snapshot.*missing.*", "Documentation & Reporting", 0, 1.5, 3, "Missing documentation"],
                [".*RIOT.*red.*", "Nesting & Tool Errors", 0, 4, 2, "RIOT validation failure"],
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

            # Sheet 7: Business Multipliers
            ws_biz = wb.create_sheet("Business Multipliers")
            biz_headers = ["Multiplier_Name", "Value", "Description"]
            ws_biz.append(biz_headers)

            biz_data = [
                ["revenue_at_risk", 2.5, "Multiplier for downstream revenue impact (total_cost × this)"],
                ["opportunity_cost", 0.35, "Opportunity cost as fraction of total cost"],
                ["customer_impact", 1.5, "Multiplier for customer-facing issue costs"],
                ["sla_penalty", 0.2, "SLA penalty rate for critical issues"],
                ["prevention_rate", 0.8, "Expected prevention rate for ROI calculation (0-1)"],
                ["cost_avoidance_rate", 0.7, "Expected cost avoidance rate for repeat issues (0-1)"],
            ]

            for row in biz_data:
                ws_biz.append(row)

            self._style_header(ws_biz, len(biz_headers))
            ws_biz.column_dimensions['A'].width = 25
            ws_biz.column_dimensions['C'].width = 55

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
            
            # Load Sub-Category Costs
            if "Sub-Category Costs" in wb.sheetnames:
                ws = wb["Sub-Category Costs"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0] and row[1]:
                        category = str(row[0]).strip()
                        sub_cat = str(row[1]).strip()
                        if category not in self.sub_category_costs:
                            self.sub_category_costs[category] = {}
                        self.sub_category_costs[category][sub_cat] = {
                            'material_cost': float(row[2] or 0),
                            'labor_hours': float(row[3] or 0),
                            'hourly_rate': float(row[4] or DEFAULT_HOURLY_RATE),
                        }
                total_sub = sum(len(v) for v in self.sub_category_costs.values())
                logger.info(f"  → Loaded {total_sub} sub-category cost entries across {len(self.sub_category_costs)} categories")

            # Load Origin Premiums
            if "Origin Premiums" in wb.sheetnames:
                ws = wb["Origin Premiums"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        origin = str(row[0]).strip().lower()
                        self.origin_premiums[origin] = float(row[1] or 0)
                logger.info(f"  → Loaded {len(self.origin_premiums)} origin premium rules")

            # Load Business Multipliers (if sheet exists)
            if "Business Multipliers" in wb.sheetnames:
                ws = wb["Business Multipliers"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        name = str(row[0]).strip().lower()
                        self.business_multipliers[name] = float(row[1] or 1.0)
                logger.info(f"  → Loaded {len(self.business_multipliers)} business multipliers")

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
        sub_category: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate total financial impact (rework cost) for an escalation.

        Formula: (Material + Labor_Hours × Hourly_Rate) × Severity_Mult × (1 + Origin_Premium)
        Based entirely on price_catalog.xlsx values. No delay cost - no data to support it.

        Lookup priority:
        1. Keyword pattern match (highest priority)
        2. Sub-category cost (if sub_category provided and found)
        3. Category cost (fallback)
        4. 'Unclassified' category (last resort)

        Returns dict with cost breakdown and audit trail (source field).
        """
        if not self.is_loaded:
            self.load_catalog()

        source = 'unknown'
        keyword_match = self._match_keyword_pattern(description)

        if keyword_match:
            material_cost = keyword_match['material_cost']
            labor_hours = keyword_match['labor_hours']
            hourly_rate = DEFAULT_HOURLY_RATE
            source = f"keyword:{keyword_match['pattern']}"
        elif (sub_category and category in self.sub_category_costs
              and sub_category in self.sub_category_costs[category]):
            sub_costs = self.sub_category_costs[category][sub_category]
            material_cost = sub_costs.get('material_cost', 0)
            labor_hours = sub_costs.get('labor_hours', 2)
            hourly_rate = sub_costs.get('hourly_rate', DEFAULT_HOURLY_RATE)
            source = f"sub_category:{category}/{sub_category}"
        elif category in self.category_costs:
            cat_costs = self.category_costs[category]
            material_cost = cat_costs.get('material_cost', 0)
            labor_hours = cat_costs.get('labor_hours', 2)
            hourly_rate = cat_costs.get('hourly_rate', DEFAULT_HOURLY_RATE)
            source = f"category:{category}"
        elif 'Unclassified' in self.category_costs:
            cat_costs = self.category_costs['Unclassified']
            material_cost = cat_costs.get('material_cost', 0)
            labor_hours = cat_costs.get('labor_hours', 2)
            hourly_rate = cat_costs.get('hourly_rate', DEFAULT_HOURLY_RATE)
            source = f"fallback:Unclassified (requested:{category})"
        else:
            # Catalog loaded but category not found and no Unclassified - fail loud
            logger.warning(f"No pricing data for category '{category}' and no Unclassified fallback")
            material_cost = 0
            labor_hours = 0
            hourly_rate = DEFAULT_HOURLY_RATE
            source = f"no_match:{category}"

        # Rework cost only: material + labor (no delay - no data to support it)
        labor_cost = labor_hours * hourly_rate
        base_cost = material_cost + labor_cost

        severity_key = severity.lower() if severity else 'medium'
        severity_mult = self.severity_multipliers.get(severity_key, 1.0)
        if severity_key not in self.severity_multipliers:
            logger.warning(f"Unknown severity '{severity}', using multiplier 1.0")

        origin_key = origin.lower() if origin else 'technical'
        origin_premium = self.origin_premiums.get(origin_key, 0.0)

        total_impact = base_cost * severity_mult * (1 + origin_premium)

        # Validation: sanity check the output
        if total_impact < 0:
            logger.error(f"Negative financial impact calculated: {total_impact} for {category}/{sub_category}")
            total_impact = 0

        return {
            'material_cost': round(material_cost, 2),
            'labor_cost': round(labor_cost, 2),
            'labor_hours': labor_hours,
            'hourly_rate': hourly_rate,
            'base_cost': round(base_cost, 2),
            'severity_multiplier': severity_mult,
            'origin_premium': origin_premium,
            'total_impact': round(total_impact, 2),
            'source': source,
            'keyword_match': keyword_match['pattern'] if keyword_match else None,
        }
    
    def get_business_multiplier(self, name: str, default: float = 1.0) -> float:
        """Get a business multiplier from price_catalog.xlsx.

        Falls back to default if the multiplier isn't configured.
        """
        if not self.is_loaded:
            self.load_catalog()
        return self.business_multipliers.get(name.lower(), default)

    def get_benchmark_costs(self) -> Dict[str, float]:
        """Calculate benchmark cost statistics from the loaded catalog.

        Returns dict with avg_per_ticket, best_in_class, industry_avg, laggard.
        All derived from price_catalog.xlsx data, not hardcoded.
        """
        if not self.is_loaded:
            self.load_catalog()

        if not self.category_costs:
            return {'avg_per_ticket': 0, 'best_in_class': 0, 'industry_avg': 0, 'laggard': 0, 'hourly_rate': DEFAULT_HOURLY_RATE}

        base_costs = []
        for cat_data in self.category_costs.values():
            material = cat_data.get('material_cost', 0)
            labor = cat_data.get('labor_hours', 0) * cat_data.get('hourly_rate', DEFAULT_HOURLY_RATE)
            base_costs.append(material + labor)

        avg_base = sum(base_costs) / len(base_costs) if base_costs else 0
        avg_sev = sum(self.severity_multipliers.values()) / len(self.severity_multipliers) if self.severity_multipliers else 1.0

        return {
            'avg_per_ticket': round(avg_base * avg_sev, 2),
            'best_in_class': round(min(base_costs), 2),
            'industry_avg': round(avg_base * avg_sev, 2),
            'laggard': round(max(base_costs) * max(self.severity_multipliers.values(), default=1.0), 2),
            'hourly_rate': next(iter(self.category_costs.values()), {}).get('hourly_rate', DEFAULT_HOURLY_RATE),
        }

    def get_catalog_summary(self) -> str:
        """Get summary of loaded pricing data for AI context."""
        if not self.is_loaded:
            return "Price catalog not loaded."

        total_sub = sum(len(v) for v in self.sub_category_costs.values())
        summary_lines = [
            f"PRICE CATALOG SUMMARY:",
            f"- {len(self.category_costs)} category cost entries",
            f"- {total_sub} sub-category cost entries",
            f"- {len(self.keyword_costs)} keyword pattern rules",
            f"- {len(self.severity_multipliers)} severity multipliers",
            f"- {len(self.origin_premiums)} origin premium rules",
            f"- {len(self.business_multipliers)} business multipliers",
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
