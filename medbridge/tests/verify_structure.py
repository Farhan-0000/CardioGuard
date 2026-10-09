import os
import sys


def verify():
    required_dirs = [
        "medbridge/api",
        "medbridge/core",
        "medbridge/state",
        "medbridge/ai",
        "medbridge/retrieval",
        "medbridge/db",
        "medbridge/ingestion",
        "medbridge/frontend",
        "medbridge/tests"
    ]

    missing = []
    for d in required_dirs:
        if not os.path.isdir(d):
            missing.append(d)

    if missing:
        print(f"FAILED: Missing directories: {missing}")
        sys.exit(1)

    required_files = [
        ".gitignore",
        ".env.example"
    ]

    for f in required_files:
        if not os.path.isfile(f):
            missing.append(f)

    if missing:
        print(f"FAILED: Missing files: {missing}")
        sys.exit(1)

    print("SUCCESS: Directory structure and base files verified.")

if __name__ == "__main__":
    verify()
