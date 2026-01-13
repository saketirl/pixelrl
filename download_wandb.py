import wandb
import pandas as pd
import argparse

def download_run(run_path: str, output_prefix: str = None):
    """Download all data from a W&B run.

    Args:
        run_path: W&B run path (e.g., "saketirl/benchmark/tciu2l4f")
        output_prefix: Prefix for output files (default: run ID)
    """
    api = wandb.Api()
    run = api.run(run_path)

    run_id = run_path.split("/")[-1]
    prefix = output_prefix or run_id

    # Download all logged metrics as a DataFrame
    print(f"Downloading history for {run_path}...")
    history = run.history(samples=100000)  # High sample count to get all data
    history_file = f"{prefix}_history.csv"
    history.to_csv(history_file, index=False)
    print(f"Saved {len(history)} rows to {history_file}")

    # Get summary metrics
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    summary_df = pd.DataFrame([summary])
    summary_file = f"{prefix}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    # Get config
    config = dict(run.config)
    config_df = pd.DataFrame([config])
    config_file = f"{prefix}_config.csv"
    config_df.to_csv(config_file, index=False)
    print(f"Saved config to {config_file}")

    # Print summary
    print("\n" + "="*50)
    print("Summary metrics:")
    print("="*50)
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")

    print("\n" + "="*50)
    print("Config:")
    print("="*50)
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")

    # Download any files (like model checkpoints)
    files = list(run.files())
    if files:
        print("\n" + "="*50)
        print("Downloading files:")
        print("="*50)
        for file in files:
            print(f"  Downloading {file.name}...")
            file.download(replace=True)

    return history, summary, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download W&B run data")
    parser.add_argument("run_path", help="W&B run path (e.g., saketirl/benchmark/tciu2l4f)")
    parser.add_argument("--output-prefix", "-o", help="Prefix for output files")
    args = parser.parse_args()

    download_run(args.run_path, args.output_prefix)
