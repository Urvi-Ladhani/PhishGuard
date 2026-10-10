import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

notebooks_to_test = [
    "notebooks/05_logistic_regression.ipynb",
    "notebooks/06_decision_tree.ipynb",
    "notebooks/07_model_comparison_and_test_eval.ipynb"
]

for nb_path in notebooks_to_test:
    print(f"\n--- Testing Notebook Execution: {nb_path} ---")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    
    # Create an execution namespace
    exec_globals = {}
    
    for idx, cell in enumerate(nb_json["cells"]):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            print(f"Executing Code Cell {idx + 1}...")
            try:
                exec(code, exec_globals)
                print(f"Cell {idx + 1} executed successfully.")
            except Exception as e:
                print(f"ERROR in Cell {idx + 1}: {e}")
                raise e

print("\nAll notebooks executed with 0 errors!")
