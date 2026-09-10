import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


class FrontendContractTests(unittest.TestCase):
    def test_every_by_id_selector_exists_once(self):
        selectors = set(re.findall(r'byId\("([^\"]+)"\)', APP_JS))
        ids = re.findall(r'id="([^\"]+)"', INDEX_HTML)
        self.assertFalse(selectors - set(ids), f"Missing DOM IDs: {sorted(selectors - set(ids))}")
        duplicates = {item for item in ids if ids.count(item) > 1}
        self.assertFalse(duplicates, f"Duplicate DOM IDs: {sorted(duplicates)}")

    def test_analyze_event_panel_contract(self):
        required = {
            "analyzeBtn", "analyzePanel", "analyzePanelTitle", "analyzeSummary",
            "anlzTempClim", "anlzPrecipClim", "anlzWindClim", "anlzClimMethod",
            "anlzZTemp", "anlzZPrecip", "anlzZWind", "anlzZMax", "anlzEfi",
            "anlzEfiInterp", "anlzEfiProv", "anlzSevBand", "anlzSevScore",
            "anlzSevExceed", "anlzSubClim", "anlzSubAnomaly", "anlzSubEfi",
            "anlzSubGnn", "anlzSubDiff", "anlzSubPhys",
        }
        ids = set(re.findall(r'id="([^\"]+)"', INDEX_HTML))
        self.assertFalse(required - ids, f"Missing analysis IDs: {sorted(required - ids)}")
        self.assertIn("/api/analyze-event", APP_JS)
        self.assertIn("handleAnalyzeEvent", APP_JS)


if __name__ == "__main__":
    unittest.main()