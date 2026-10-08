"""Push this folder to the Hugging Face Hub.

    pip install huggingface_hub
    huggingface-cli login            # paste a token with *write* access
    python upload_to_hf.py --repo <your-hf-username>/ValidButVacuous
"""
import argparse
import os

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="e.g. koushik/ValidButVacuous")
ap.add_argument("--type", default="model", choices=["model", "dataset", "space"])
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

api = HfApi()
api.create_repo(args.repo, repo_type=args.type, private=args.private, exist_ok=True)
api.upload_folder(
    folder_path=os.path.dirname(os.path.abspath(__file__)),
    repo_id=args.repo, repo_type=args.type,
    commit_message="Valid but Vacuous: paper, code, figures",
    ignore_patterns=["__pycache__/*", "**/__pycache__/*", ".pytest_cache/*", "*.pyc", "results/*"],
)
print(f"done: https://huggingface.co/{'' if args.type == 'model' else args.type + 's/'}{args.repo}")
