import io
import sys
import json
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

notebook_paths = [
    "notebooks/01_problem_data_audit.ipynb",
    "notebooks/02_eda.ipynb",
    "notebooks/03_feature_engineering.ipynb",
    "notebooks/04_data_split_and_baseline.ipynb",
    "notebooks/05_logistic_regression.ipynb",
    "notebooks/06_decision_tree.ipynb",
    "notebooks/07_model_comparison_and_test_eval.ipynb"
]

print("==================================================")
print("RUNNING AND SAVING OUTPUTS FOR ALL NOTEBOOKS 01-07")
print("==================================================")

for rel_path in notebook_paths:
    nb_file = PROJECT_ROOT / rel_path
    print(f"\n>>> Executing notebook: {rel_path} ...")
    
    with open(nb_file, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
        
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(nb_file)
    }
    
    exec_count = 1
    for idx, cell in enumerate(nb_json.get("cells", [])):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            code = "".join(source_lines)
            
            if not code.strip():
                continue
                
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            plt.close('all')
            
            try:
                sys.stdout = stdout_capture
                exec(code, exec_globals)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"FAILED on cell {idx + 1} of {rel_path}!")
                print(f"Error: {e}")
                raise e
            finally:
                sys.stdout = old_stdout
                
            outputs = []
            captured_text = stdout_capture.getvalue()
            if captured_text:
                # Store text split with newlines
                split_lines = captured_text.splitlines(keepends=True)
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": split_lines
                })
                
            # Collect any figures created during cell execution
            for f_num in plt.get_fignums():
                fig = plt.figure(f_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode('utf-8')
                outputs.append({
                    "data": {
                        "image/png": b64,
                        "text/plain": [f"<Figure size at {id(fig):#x}>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)
                
            cell["outputs"] = outputs
            cell["execution_count"] = exec_count
            exec_count += 1
            print(f"  [Cell {idx+1}] executed successfully ({len(outputs)} output items).")
            
    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
        
    print(f"[OK] Notebook saved with outputs: {rel_path}")

print("\n==================================================")
print("ALL NOTEBOOKS (01 to 07) EXECUTED AND SAVED WITH 0 ERRORS!")
print("==================================================")
