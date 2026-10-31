"""Delhivery Last-Mile Delay Intelligence Pipeline — Master Orchestrator.

Executes the complete capstone intelligence pipeline in sequential order:
  Step 1: Data Generation (generate_data.py)
  Step 2: Cleaning & Exploratory Data Analysis (part_a_clean_eda.py)
  Step 3: Transit Hours Regression Modeling (part_b_regression.py)
  Step 4: Delay SLA Breach Classification (part_c_classification.py)
  Step 5: Regional Hub Clustering & Playbook Segmentation (part_d_segments.py)
  Step 6: Production Model Packaging & Parity Verification (package_model.py)
  Step 7: Live Batch Scoring & Operational Actioning (predict_delay.py)

Usage:
  python run_pipeline.py
"""

import os
import subprocess
import sys
import time

PIPELINE_STEPS = [
    ("generate_data.py", "Generating synthetic shipment & hub records"),
    ("part_a_clean_eda.py", "Data cleaning, feature building & baseline delay EDA"),
    ("part_b_regression.py", "Delivery transit hours regression & tree depth study"),
    ("part_c_classification.py", "Feature selection, classifier tuning & threshold calibration"),
    ("part_d_segments.py", "K-Means hub clustering (K=3) & tier mismatch audit"),
    ("package_model.py", "Serializing production model bundle & verifying reload parity"),
    ("predict_delay.py", "Scoring 48 incoming parcels & routing operational playbooks"),
]


def run_command(script_name, description, step_num, total_steps):
    header = f"[{step_num}/{total_steps}] Running {script_name} — {description}"
    line = "=" * len(header)
    print(f"\n{line}\n{header}\n{line}")

    if not os.path.exists(script_name):
        print(f"ERROR: Script '{script_name}' not found in current directory.", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    result = subprocess.run([sys.executable, script_name])
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\nFAILED: {script_name} exited with code {result.returncode} after {elapsed:.2f}s", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"COMPLETED: {script_name} in {elapsed:.2f} seconds.")


def main():
    total_steps = len(PIPELINE_STEPS)
    print("\n" + "=" * 70)
    print("DELHIVERY LAST-MILE DELAY INTELLIGENCE PIPELINE: FULL RUNNER")
    print("=" * 70)
    overall_start = time.time()

    for idx, (script, desc) in enumerate(PIPELINE_STEPS, start=1):
        run_command(script, desc, idx, total_steps)

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {overall_elapsed:.2f} SECONDS!")
    print("All outputs, models, and tables have been generated and verified.")
    print("=" * 70)
    print("\nNext step: Explore the visual dashboard artifact at:")
    print(r"C:\Users\s_dev\.gemini\antigravity\brain\081fa200-3475-4c1a-b080-2ef6ba412c54\delhivery_dashboard.html")
    print("\nOr open directly in PowerShell via:")
    print(r'Start-Process "C:\Users\s_dev\.gemini\antigravity\brain\081fa200-3475-4c1a-b080-2ef6ba412c54\delhivery_dashboard.html"')
    print()


if __name__ == "__main__":
    main()

