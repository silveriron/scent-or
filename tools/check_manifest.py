import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, "README.md")
CODE_DIRECTORIES = ("preprocessing", "training", "evaluation", "analysis")
EXCLUDED = {"analysis/stats_utils.py"}
MAPPING_HEADING = "## Mapping to the Manuscript"
TREE_HEADING = "## Repository Structure"


def tracked_python_files():
    output = subprocess.run(["git", "-C", REPO_ROOT, "ls-files", "*.py"],
                            capture_output=True, text=True, check=True).stdout
    return {line for line in output.splitlines()
            if line.startswith(CODE_DIRECTORIES) and line not in EXCLUDED}


def section(text, heading):
    start = text.find(heading)
    if start < 0:
        return ""
    end = text.find("\n## ", start + len(heading))
    return text[start:end if end > 0 else len(text)]


def mapping_paths(text):
    body = section(text, MAPPING_HEADING)
    return [match.group(1) for match in re.finditer(r"^\|\s*`([^`]+)`\s*\|", body,
                                                    flags=re.MULTILINE)]


def tree_paths(text):
    body = section(text, TREE_HEADING)
    found = []
    for line in body.splitlines():
        match = re.search(r"[│├└─\s]*([A-Za-z0-9_./-]+\.(?:py|txt|md|csv|pkl))", line)
        if match and not line.strip().startswith("#"):
            found.append(match.group(1))
    return found


def main():
    readme = open(README, encoding="utf-8").read()
    tracked = tracked_python_files()
    failures = []

    listed = mapping_paths(readme)
    for path in listed:
        if not os.path.exists(os.path.join(REPO_ROOT, path)):
            failures.append(f"mapping table lists a missing path: {path}")
    duplicates = {path for path in listed if listed.count(path) > 1}
    for path in sorted(duplicates):
        failures.append(f"mapping table lists {path} more than once")
    for path in sorted(tracked - set(listed)):
        failures.append(f"tracked module absent from the mapping table: {path}")
    for path in sorted(set(listed) - tracked):
        if path.endswith(".py") and path.startswith(CODE_DIRECTORIES):
            failures.append(f"mapping table lists an untracked module: {path}")

    for name in tree_paths(readme):
        candidate = name if os.path.sep in name else None
        if candidate and not os.path.exists(os.path.join(REPO_ROOT, candidate)):
            failures.append(f"structure tree lists a missing path: {name}")

    numbers = []
    for path in sorted(tracked):
        if not path.startswith("analysis/"):
            continue
        match = re.match(r"(\d+)_", os.path.basename(path))
        if match:
            numbers.append(int(match.group(1)))
    for value in sorted({n for n in numbers if numbers.count(n) > 1}):
        failures.append(f"duplicate number in analysis/: {value:02d}")
    if numbers:
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        for value in missing:
            failures.append(f"gap in the analysis/ numbering: {value:02d}")

    print(f"tracked modules {len(tracked)}   mapping rows {len(listed)}   "
          f"analysis/ numbers {min(numbers) if numbers else 0}"
          f"-{max(numbers) if numbers else 0}")
    if failures:
        for message in failures:
            print(f"  FAIL  {message}")
        return 1
    print("  manifest consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
