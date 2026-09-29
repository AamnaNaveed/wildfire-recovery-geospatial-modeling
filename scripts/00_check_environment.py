import sys 
import json 
import platform 
from pathlib import Path 
from datetime import datetime 
 
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
REPORT_PATH = PROJECT_ROOT / "reports" / "environment_check.json" 
 
def main(): 
    report = { 
        "timestamp": datetime.now().isoformat(), 
        "python_version": sys.version, 
        "platform": platform.platform(), 
        "packages": {}, 
    } 
    for pkg in ["numpy", "pandas", "matplotlib", "sklearn", "xarray", "rasterio", "geopandas"]: 
        try: 
            mod = __import__(pkg) 
            report["packages"][pkg] = getattr(mod, "__version__", "unknown") 
        except ImportError: 
            report["packages"][pkg] = "NOT INSTALLED" 
 
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True) 
    REPORT_PATH.write_text(json.dumps(report, indent=2)) 
    print(json.dumps(report, indent=2)) 
 
if __name__ == "__main__": 
    main() 
