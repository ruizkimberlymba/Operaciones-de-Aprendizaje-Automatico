"""Siete ejecuciones reales: cinco configuraciones, repetición y ablación."""
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = ["--C","1","--class-weight","balanced","--solver","lbfgs","--max-iter","1000","--seed","42","--run-name","baseline"]
CONFIGS = [BASE,
 ["--C","0.01","--run-name","C_0.01"],
 ["--C","100","--run-name","C_100"],
 ["--class-weight","none","--run-name","sin_pesos"],
 ["--solver","newton-cg","--run-name","newton_cg"],
 BASE,
 ["--features","decision","--run-name","sin_grado_tasa"]]
if __name__ == "__main__":
    commands = []
    for args in CONFIGS:
        command = [sys.executable, "train.py", *args]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        commands.append({"argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        (ROOT/"evidence/execution_log.json").write_text(json.dumps(commands,indent=2),encoding="utf-8")
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
