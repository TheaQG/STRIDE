from pathlib import Path
import sys
# Allow direct execution via: python test_scripts/norcp/variable_test.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.variable_registry import (
    list_registered_variables,
    get_source_spec,
    get_default_transform,
    canonicalize_variable_name,
)

print(list_registered_variables())
print(get_source_spec("prcp", "NORCP_HR"))
print(get_default_transform("prcp"))
print(canonicalize_variable_name("pr"))
print(canonicalize_variable_name("orog"))

print(list_registered_variables())
print(get_source_spec("prcp", "NORCP_HR"))
print(get_source_spec("hus1000", "NORCP_LR"))
print(get_source_spec("zg500", "NORCP_LR"))
print(canonicalize_variable_name("ta850"))
print(canonicalize_variable_name("orog"))