"""
SZZ labeling module for JIT Defect Prediction.

Implements the SZZ (Śliwerski-Zimmermann-Zeller) algorithm to identify
bug-inducing commits by tracing bug-fix commits back to the commits that
introduced the buggy lines.

Process:
1. Identify fix commits by message pattern matching
2. Filter out refactoring-only fixes (>80% whitespace/comment changes)
3. Run git blame to find inducing commits for each fix
4. Join labels back to the full commit dataset

Key principle: SZZ has known false-positive issues with cosmetic changes;
the refactoring filter reduces (but does not eliminate) this noise.
"""

import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def identify_fix_commits(commits_df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Identify bug-fix commits by message pattern matching.

    Args:
        commits_df: DataFrame with 'msg' column
        pattern: Regex pattern to match fix-related keywords (case-insensitive)

    Returns:
        Subset of commits_df where msg matches the pattern

    Example patterns:
        r"\b(fix|fixes|fixed|bug|defect|resolve[sd]?)\b|LANG-\d+"
        Matches "fix", "bug", "defect", "resolve", or Jira-style "LANG-1234"
    """
    # Case-insensitive regex search
    regex = re.compile(pattern, re.IGNORECASE)
    mask = commits_df["msg"].apply(lambda msg: bool(regex.search(str(msg))))
    
    fix_commits = commits_df[mask].copy()
    
    return fix_commits


def apply_refactoring_filter(
    fix_commits_df: pd.DataFrame, 
    repo_path: str, 
    threshold: float
) -> pd.DataFrame:
    """
    Filter out cosmetic-only fix commits to reduce SZZ false positives.

    A fix commit is excluded if (whitespace_lines + comment_only_lines) / 
    total_changed_lines exceeds the threshold.

    Args:
        fix_commits_df: DataFrame with 'commit_hash' column
        repo_path: Path to the Git repository
        threshold: Fraction above which a fix is considered cosmetic (e.g., 0.8)

    Returns:
        Filtered DataFrame with cosmetic fixes removed
    """
    filtered_hashes = []
    
    for commit_hash in fix_commits_df["commit_hash"]:
        cosmetic_ratio = _compute_cosmetic_ratio(commit_hash, repo_path)
        
        # Keep commits below the cosmetic threshold
        if cosmetic_ratio < threshold:
            filtered_hashes.append(commit_hash)
    
    # Return rows matching the kept hashes
    result = fix_commits_df[fix_commits_df["commit_hash"].isin(filtered_hashes)].copy()
    
    return result


def _compute_cosmetic_ratio(commit_hash: str, repo_path: str) -> float:
    """
    Compute the ratio of cosmetic changes (whitespace/comments) in a commit.

    Uses git diff to analyze the commit's changes and count lines that are
    purely whitespace or comment-related.

    Args:
        commit_hash: Full 40-character commit SHA
        repo_path: Path to the Git repository

    Returns:
        Float between 0.0 and 1.0 representing cosmetic ratio
        Returns 0.0 if diff cannot be parsed (commit is considered substantive)
    """
    try:
        # Get the diff for this commit
        result = subprocess.run(
            ["git", "diff", f"{commit_hash}^", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            shell=False
        )
        
        diff_lines = result.stdout.split("\n")
        
        total_changes = 0
        cosmetic_changes = 0
        
        for line in diff_lines:
            # Count added/removed lines (start with + or -, but not +++ or ---)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                total_changes += 1
                
                # Check if line is purely whitespace or looks like a comment
                content = line[1:].strip()  # Remove +/- prefix
                
                if not content:  # Empty or whitespace-only
                    cosmetic_changes += 1
                elif _looks_like_comment(content):
                    cosmetic_changes += 1
        
        if total_changes == 0:
            return 0.0
        
        return cosmetic_changes / total_changes
    
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception):
        # If git command fails, assume commit is substantive (don't filter)
        return 0.0


def _looks_like_comment(line: str) -> bool:
    """
    Heuristic check if a line looks like a comment.

    Checks common comment patterns for multiple languages:
    - // (C++, Java, JavaScript)
    - # (Python, Ruby, Shell)
    - /* */ (C, Java, CSS)
    - <!-- --> (HTML, XML)
    - * (Javadoc/block comment continuation)

    Args:
        line: Stripped line content

    Returns:
        True if line appears to be a comment
    """
    comment_patterns = [
        "//", "#", "/*", "*/", "*", "<!--", "-->", "\"\"\"", "'''"
    ]
    
    return any(line.startswith(pattern) for pattern in comment_patterns)


def run_szz(fix_commits_df: pd.DataFrame, repo_path: str) -> pd.DataFrame:
    """
    Run SZZ algorithm to find bug-inducing commits.

    For each fix commit, uses git blame to trace each changed line back to
    the commit that last modified it before the fix. Those commits are the
    "inducing" commits.

    Args:
        fix_commits_df: DataFrame with 'commit_hash' column
        repo_path: Path to the Git repository

    Returns:
        DataFrame with columns [inducing_commit_hash, fix_commit_hash]
        Deduplicated on (inducing_commit_hash, fix_commit_hash) pairs
    """
    inducing_pairs = []
    
    for fix_hash in fix_commits_df["commit_hash"]:
        # Get list of files changed in this fix
        changed_files = _get_changed_files(fix_hash, repo_path)
        
        for file_path, changed_lines in changed_files:
            # Run git blame for each changed line
            for line_num in changed_lines:
                inducing_hash = _blame_line(fix_hash, file_path, line_num, repo_path)
                
                if inducing_hash and inducing_hash != fix_hash:
                    inducing_pairs.append({
                        "inducing_commit_hash": inducing_hash,
                        "fix_commit_hash": fix_hash
                    })
    
    # Create DataFrame and deduplicate
    if inducing_pairs:
        result_df = pd.DataFrame(inducing_pairs)
        result_df = result_df.drop_duplicates(
            subset=["inducing_commit_hash", "fix_commit_hash"]
        ).reset_index(drop=True)
    else:
        # No inducing commits found (empty result)
        result_df = pd.DataFrame(columns=["inducing_commit_hash", "fix_commit_hash"])
    
    return result_df


def _get_changed_files(commit_hash: str, repo_path: str) -> list[tuple[str, list[int]]]:
    """
    Get list of files changed in a commit and their changed line numbers.

    Args:
        commit_hash: Full 40-character commit SHA
        repo_path: Path to the Git repository

    Returns:
        List of (file_path, [line_numbers]) tuples
        Example: [("src/main.py", [10, 11, 15]), ("src/util.py", [42])]
    """
    try:
        # Get the diff with line numbers
        result = subprocess.run(
            ["git", "diff", "-U0", f"{commit_hash}^", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            shell=False
        )
        
        files_and_lines = []
        current_file = None
        
        for line in result.stdout.split("\n"):
            # File marker: +++ b/path/to/file
            if line.startswith("+++ b/"):
                current_file = line[6:]  # Remove "+++ b/"
            
            # Hunk header: @@ -start,count +start,count @@
            elif line.startswith("@@") and current_file:
                # Parse the hunk header to get changed line range
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if match:
                    start_line = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    
                    # Generate list of changed line numbers
                    changed_lines = list(range(start_line, start_line + count))
                    
                    if changed_lines and current_file:
                        files_and_lines.append((current_file, changed_lines))
        
        return files_and_lines
    
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


def _blame_line(
    fix_hash: str, 
    file_path: str, 
    line_num: int, 
    repo_path: str
) -> str | None:
    """
    Use git blame to find which commit last modified a line before a fix.

    Args:
        fix_hash: The fix commit hash
        file_path: Path to file relative to repo root
        line_num: Line number to blame (1-indexed)
        repo_path: Path to the Git repository

    Returns:
        The inducing commit hash (40 chars), or None if blame fails
    """
    try:
        # Blame the line at fix_hash^ (parent of fix)
        # Use --follow to track through file renames
        result = subprocess.run(
            [
                "git", "blame", 
                "-L", f"{line_num},{line_num}",
                "--follow",
                f"{fix_hash}^",
                "--",
                file_path
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            shell=False
        )
        
        # Parse blame output: first 40 characters are the commit hash
        blame_output = result.stdout.strip()
        if len(blame_output) >= 40:
            inducing_hash = blame_output[:40]
            
            # Validate it's a valid hex hash
            if re.match(r"^[a-f0-9]{40}$", inducing_hash):
                return inducing_hash
        
        return None
    
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Blame failed (file didn't exist, line out of range, etc.)
        return None


def label_commits(
    commits_df: pd.DataFrame, 
    inducing_pairs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join bug-inducing labels onto the full commit dataset.

    Args:
        commits_df: Full commit DataFrame with 'commit_hash' column
        inducing_pairs_df: DataFrame with [inducing_commit_hash, fix_commit_hash]

    Returns:
        commits_df with two new columns:
        - is_bug_inducing (int8): 0 or 1
        - fix_commit_hash (str, nullable): The fix that identified this as inducing
    """
    # Create a mapping from inducing hash to fix hash
    # If a commit induced multiple bugs, we keep only one fix (arbitrary choice)
    if not inducing_pairs_df.empty:
        inducing_map = inducing_pairs_df.groupby("inducing_commit_hash")["fix_commit_hash"].first()
    else:
        inducing_map = pd.Series(dtype=str)
    
    # Left join to add labels
    result = commits_df.copy()
    result["fix_commit_hash"] = result["commit_hash"].map(inducing_map)
    result["is_bug_inducing"] = result["fix_commit_hash"].notna().astype("int8")
    
    return result


if __name__ == "__main__":
    """
    Standalone execution: label commits from mined dataset.
    
    Usage: python -m pipeline.label_szz
    Input: data/commits.parquet
    Output: data/commits_labeled.parquet
    """
    from pipeline.config import (
        REPO_LOCAL_PATH, 
        FIX_PATTERN_REGEX,
        REFACTOR_FILTER_THRESHOLD,
        MIN_COMMITS,
        MIN_POSITIVE_RATE,
        MAX_POSITIVE_RATE
    )
    
    print("Loading commits from data/commits.parquet...")
    commits_df = pd.read_parquet("data/commits.parquet")
    print(f"  Total commits: {len(commits_df)}")
    
    print(f"\nIdentifying fix commits (pattern: {FIX_PATTERN_REGEX})...")
    fix_commits = identify_fix_commits(commits_df, FIX_PATTERN_REGEX)
    print(f"  Fix commits found: {len(fix_commits)}")
    
    print(f"\nApplying refactoring filter (threshold: {REFACTOR_FILTER_THRESHOLD})...")
    filtered_fixes = apply_refactoring_filter(
        fix_commits, 
        REPO_LOCAL_PATH, 
        REFACTOR_FILTER_THRESHOLD
    )
    print(f"  Fix commits after filter: {len(filtered_fixes)}")
    print(f"  Filtered out: {len(fix_commits) - len(filtered_fixes)}")
    
    print(f"\nRunning SZZ on {len(filtered_fixes)} fix commits...")
    print("  (This may take several minutes depending on repo size)")
    inducing_pairs = run_szz(filtered_fixes, REPO_LOCAL_PATH)
    print(f"  Inducing commit pairs found: {len(inducing_pairs)}")
    
    print("\nLabeling commits...")
    labeled_commits = label_commits(commits_df, inducing_pairs)
    
    # Validation
    positive_count = labeled_commits["is_bug_inducing"].sum()
    positive_rate = positive_count / len(labeled_commits)
    
    print(f"\n✅ Labeling complete!")
    print(f"   Total commits: {len(labeled_commits)}")
    print(f"   Bug-inducing: {positive_count} ({positive_rate:.1%})")
    print(f"   Clean: {len(labeled_commits) - positive_count}")
    
    # Success criteria checks (SC2)
    print(f"\n🔍 Validation against Success Criteria (SC2):")
    print(f"   ≥{MIN_COMMITS} commits: {'✅ PASS' if len(labeled_commits) >= MIN_COMMITS else '❌ FAIL'}")
    print(f"   {MIN_POSITIVE_RATE:.0%}-{MAX_POSITIVE_RATE:.0%} positive rate: ", end="")
    if MIN_POSITIVE_RATE <= positive_rate <= MAX_POSITIVE_RATE:
        print("✅ PASS")
    else:
        print(f"❌ FAIL (actual: {positive_rate:.1%})")
        print(f"\n⚠️  Positive rate out of range. Review FIX_PATTERN_REGEX in config.py")
    
    # Save
    output_path = "data/commits_labeled.parquet"
    labeled_commits.to_parquet(output_path, index=False)
    print(f"\n💾 Output: {output_path}")
