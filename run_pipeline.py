"""Run the whole offline pipeline end to end:
    ingest -> spatial split + imbalance -> samples -> RF baseline -> CNN experiments -> champion export

    python run_pipeline.py                 # everything
    python run_pipeline.py --from train    # skip stages already done
    python run_pipeline.py --only baseline

Each stage is also runnable on its own (python -m pipeline.<stage>).
"""
import argparse
import subprocess
import sys

# The experiment grid answering the research question. Every CNN run uses flip/rotation augmentation.
EXPERIMENTS = [
    # custom CNN, RGB vs all 13 bands, with and without class weighting
    ["--model", "cnn", "--bands", "rgb", "--loss", "ce"],
    ["--model", "cnn", "--bands", "rgb", "--loss", "weighted_ce"],
    ["--model", "cnn", "--bands", "ms", "--loss", "ce"],
    ["--model", "cnn", "--bands", "ms", "--loss", "weighted_ce"],
    # EfficientNet-B0 transfer learning (frozen, then partially fine-tuned)
    ["--model", "effnet", "--bands", "rgb", "--loss", "ce"],
    ["--model", "effnet", "--bands", "rgb", "--loss", "weighted_ce"],
    ["--model", "effnet", "--bands", "rgb", "--loss", "focal"],
    # balanced reference: how much of the gap is due to imbalance at all?
    ["--model", "cnn", "--bands", "ms", "--loss", "ce", "--train-set", "full"],
]

STAGES = ["ingest", "split", "samples", "baseline", "train", "champion"]
MODULES = {"ingest": "pipeline.ingest", "split": "pipeline.split", "samples": "pipeline.make_samples",
           "baseline": "pipeline.baseline", "champion": "pipeline.select_champion"}


def run(args):
    print(f"\n>>> python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True)


def run_name(exp):
    """Same naming rule as pipeline/train.py."""
    o = dict(zip(exp[::2], exp[1::2]))
    return f"{o['--model']}_{o.get('--bands', 'rgb')}_{o.get('--loss', 'ce')}" + \
        ("_fulltrain" if o.get("--train-set") == "full" else "")


def finished_runs():
    import mlflow
    from pipeline.config import EXPERIMENT, TRACKING
    mlflow.set_tracking_uri(TRACKING)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        return set()
    df = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="attributes.status = 'FINISHED'")
    return set(df.get("tags.mlflow.runName", []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", choices=STAGES, default="ingest")
    p.add_argument("--only", choices=STAGES)
    p.add_argument("--force", action="store_true", help="re-train experiments that already finished")
    a = p.parse_args()
    stages = [a.only] if a.only else STAGES[STAGES.index(a.start):]

    for stage in stages:
        if stage == "train":
            done = set() if a.force else finished_runs()
            failed = []
            for exp in EXPERIMENTS:
                if run_name(exp) in done:
                    print(f"\n--- skipping {run_name(exp)} (already in MLflow; use --force to re-train)")
                    continue
                try:
                    run(["-m", "pipeline.train", *exp])
                except subprocess.CalledProcessError:      # one failed experiment should not stop the grid
                    failed.append(run_name(exp))
                    print(f"\n!!! {run_name(exp)} FAILED - continuing; re-run later with run_pipeline.py --from train")
            if failed:
                sys.exit(f"Training failed for: {', '.join(failed)}")
        else:
            run(["-m", MODULES[stage]])


if __name__ == "__main__":
    main()
