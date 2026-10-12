"""Everything in one command, on YOUR OWN WHOOP data:

    python run.py                  log in if needed -> sync -> analysis -> open the dashboard
    python run.py --skip-sync      re-run the analysis on the data you already have
    python run.py --no-dashboard   stop after the analysis

Needs a .env file with WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET (see README). Everything runs
locally; data and tokens stay in this folder and are git-ignored.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

ANALYSIS = [
    ("Fitting the models (cross-validation)", ["src/train_baselines.py"]),
    ("Scoring the held-out test period", ["src/train_baselines.py", "--evaluate-test"]),
    ("Splitting every score into points per input", ["src/attribution.py", "--quick"]),
    ("Finding surprising mornings", ["src/m3_anomalies.py"]),
    ("Testing whether an irregular bedtime costs points", ["src/m3_irregularity.py"]),
    ("Grouping days into kinds", ["src/m3_archetypes.py"]),
]


def step(label, args):
    print(f"\n==> {label}", flush=True)
    t = time.time()
    rc = subprocess.run([PY, *args], cwd=ROOT).returncode
    if rc != 0:
        sys.exit(f"\nStopped at: {label}. The message above says why.")
    print(f"    done in {time.time() - t:.0f} s")


def preflight():
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is needed.")
    if not (ROOT / ".git").exists():
        sys.exit("Please get the project with `git clone` (not a ZIP download): the pre-registered "
                 "bedtime test checks that its hypothesis file is committed and unchanged.")
    env = ROOT / ".env"
    text = env.read_text() if env.exists() else ""
    if "WHOOP_CLIENT_ID" not in text or "WHOOP_CLIENT_SECRET" not in text:
        sys.exit("First create a file called .env in this folder containing:\n\n"
                 "    WHOOP_CLIENT_ID=your-client-id\n    WHOOP_CLIENT_SECRET=your-client-secret\n\n"
                 "Both come from your app on the WHOOP developer portal (see README, step 1).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sync", action="store_true", help="use the data already downloaded")
    ap.add_argument("--no-dashboard", action="store_true", help="don't open the dashboard at the end")
    args = ap.parse_args()
    preflight()

    if not args.skip_sync:
        if not (ROOT / ".whoop_tokens.json").exists():
            step("Logging in to WHOOP (a browser tab will open)", ["src/login.py"])
        first = not (ROOT / "data" / "raw" / "cycles.json").exists()
        step("Downloading your full WHOOP history (first run: a few minutes)" if first
             else "Downloading new WHOOP data", ["src/sync.py", "--no-rebuild"] + (["--full"] if first else []))
    elif not (ROOT / "data" / "raw" / "cycles.json").exists():
        sys.exit("No data downloaded yet; run without --skip-sync first.")

    step("Cleaning the data", ["src/normalize_data.py"])
    step("Building features and personal baselines", ["src/features.py"])
    for label, a in ANALYSIS:
        step(label, a)

    print("\nAll done. Results: results/ (summaries) and data/processed/ (private, per day).")
    if not args.no_dashboard:
        print("Opening the dashboard; press Ctrl+C here to stop it.\n")
        subprocess.run([PY, "-m", "streamlit", "run", "app.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
