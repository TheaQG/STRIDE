from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.unit_conversion import apply_unit_conversion

pr = np.array([[0.0, 1.0e-5], [-1.0e-6, 2.0e-5]], dtype=np.float32)
tas = np.array([[273.15, 280.15], [260.15, 300.15]], dtype=np.float32)
topo = np.array([[10.0, 100.0]], dtype=np.float32)
lsm = np.array([[-0.2, 0.4, 1.2]], dtype=np.float32)

print("Converted precip:")
print(apply_unit_conversion(pr, variable="prcp", source="NORCP_HR"))

print("\nConverted temp:")
print(apply_unit_conversion(tas, variable="temp", source="NORCP_LR"))

print("\nConverted topo:")
print(apply_unit_conversion(topo, variable="topo", source="NORCP_STATIC"))

print("\nConverted lsm:")
print(apply_unit_conversion(lsm, variable="lsm", source="NORCP_STATIC"))