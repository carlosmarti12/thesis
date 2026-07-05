import argparse
import subprocess
import sys


def run_command(command):
    print("\n" + "=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    args = parser.parse_args()

    limit_args = []
    if args.limit is not None:
        limit_args = ["--limit", str(args.limit)]

    run_command([
        sys.executable,
        "src/run_experiments.py",
        "--method",
        "same_term",
        *limit_args,
    ])

    run_command([
        sys.executable,
        "src/run_experiments.py",
        "--method",
        "embedding",
        *limit_args,
    ])

    run_command([
        sys.executable,
        "src/run_experiments.py",
        "--method",
        "llm",
        *limit_args,
        "--model",
        args.model,
    ])

    run_command([
        sys.executable,
        "src/run_mas_langgraph.py",
        *limit_args,
        "--model",
        args.model,
    ])

    run_command([
        sys.executable,
        "src/run_mas_v2_langgraph.py",
        *limit_args,
        "--model",
        args.model,
    ])

    run_command([
        sys.executable,
        "src/run_mas_v3_safe_hybrid.py",
        *limit_args,
        "--model",
        args.model,
    ])

    run_command([
        sys.executable,
        "src/compare_results.py",
    ])


if __name__ == "__main__":
    main()


# python src/run_comparison.py --limit 100 --model llama3.2:3b