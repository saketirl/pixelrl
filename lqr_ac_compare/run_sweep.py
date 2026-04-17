"""
Sweep over noise seeds with fixed LQR problem and policy initialization.
Tests robustness of learning across different environment noise realizations.
"""

import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(SCRIPT_DIR, "compare_upstairs_downstairs.py")


def run_experiment(seed: int, init_seed: int, noise_seed: int, config: dict, downstairs_only: bool, use_sgd: bool) -> dict:
    """Run a single experiment and return result info."""
    n_steps = config["n_steps"]
    csv_path = os.path.join(SCRIPT_DIR, f"results_nsteps{n_steps}_noise{noise_seed}.csv")
    plot_path = os.path.join(SCRIPT_DIR, f"plot_nsteps{n_steps}_noise{noise_seed}.png")

    cmd = [
        "python3", SCRIPT,
        "--seed", str(seed),
        "--init_seed", str(init_seed),
        "--noise_seed", str(noise_seed),
        "--ds", str(config["ds"]),
        "--da", str(config["da"]),
        "--d", str(config["d"]),
        "--n_steps", str(config["n_steps"]),
        "--alpha", str(config["alpha"]),
        "--eta_actor", str(config["eta_actor"]),
        "--eta_critic", str(config["eta_critic"]),
        "--iters", str(config["iters"]),
        "--save_csv", csv_path,
        "--save_plot", plot_path,
    ]

    if downstairs_only:
        cmd.append("--downstairs_only")

    if use_sgd:
        cmd.append("--use_sgd")

    print(f"[START] noise_seed={noise_seed}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=86400,  # 24 hours
        )
        success = result.returncode == 0
        if not success:
            print(f"[FAIL] noise_seed={noise_seed}")
        else:
            print(f"[DONE] noise_seed={noise_seed} -> {csv_path}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] noise_seed={noise_seed}")
        success = False

    return {
        "noise_seed": noise_seed,
        "success": success,
        "csv_path": csv_path if success else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep over noise seeds with fixed LQR and init")

    # Fixed parameters
    parser.add_argument("--seed", type=int, default=0, help="LQR problem seed (fixed)")
    parser.add_argument("--init_seed", type=int, default=None, help="Policy init seed (default: seed+1)")

    # Sweep parameters
    parser.add_argument("--noise_seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                        help="List of noise seeds to sweep over")
    parser.add_argument("--max_parallel", type=int, default=4, help="Max parallel processes")

    # Config parameters (working config from experiments)
    parser.add_argument("--iters", type=int, default=500)  # 500 episodes
    parser.add_argument("--n_steps", type=int, default=30, help="Multi-step horizon for bootstrapping")
    parser.add_argument("--ds", type=int, default=8)
    parser.add_argument("--da", type=int, default=2)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--eta_actor", type=float, default=0.01)
    parser.add_argument("--eta_critic", type=float, default=0.05)

    parser.add_argument("--downstairs_only", action="store_true", help="Only run downstairs")
    parser.add_argument("--use_sgd", action="store_true", help="Use standard SGD instead of Cayley retraction")

    args = parser.parse_args()

    init_seed = args.init_seed if args.init_seed is not None else args.seed + 1

    config = {
        "ds": args.ds,
        "da": args.da,
        "d": args.d,
        "n_steps": args.n_steps,
        "alpha": args.alpha,
        "eta_actor": args.eta_actor,
        "eta_critic": args.eta_critic,
        "iters": args.iters,
    }

    print("=" * 60)
    print("SWEEP: Fixed LQR + Init, Varying Noise Seeds")
    print("=" * 60)
    print(f"LQR seed: {args.seed}")
    print(f"Init seed: {init_seed}")
    print(f"Noise seeds: {args.noise_seeds}")
    print(f"Config: ds={args.ds}, da={args.da}, d={args.d}, n_steps={args.n_steps}, alpha={args.alpha}")
    print(f"Learning rates: eta_actor={args.eta_actor}, eta_critic={args.eta_critic}")
    print(f"Iterations: {args.iters}")
    print(f"Downstairs only: {args.downstairs_only}")
    print(f"Use SGD: {args.use_sgd}")
    print(f"Max parallel: {args.max_parallel}")
    print("=" * 60)

    results = []
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                run_experiment, args.seed, init_seed, noise_seed, config, args.downstairs_only, args.use_sgd
            ): noise_seed
            for noise_seed in args.noise_seeds
        }
        for future in as_completed(futures):
            noise_seed = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] noise_seed={noise_seed}: {e}")
                results.append({"noise_seed": noise_seed, "success": False})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_success = sum(1 for r in results if r["success"])
    print(f"Completed: {n_success}/{len(args.noise_seeds)}")
    if n_success > 0:
        csv_files = [r["csv_path"] for r in results if r["success"]]
        print(f"CSVs: {csv_files}")


if __name__ == "__main__":
    main()
