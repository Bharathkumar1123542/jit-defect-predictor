"""
Commit mining module for JIT Defect Prediction.

Extracts commit-level features from a Git repository using PyDriller.
Computes the 17 JIT features defined in §6.1 of implementation.md.

Key principle: Features are computed using only information available AS OF
each commit (no look-ahead) to prevent temporal leakage.

This module's `compute_commit_features` function is imported by
`action/extract_pr_features.py` to guarantee train/serve feature parity (§10.2).
"""

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydriller import Repository


def mine_commits(repo_path: str, since: datetime, to: datetime) -> pd.DataFrame:
    """
    Walk commit history and compute JIT features for each commit.

    Args:
        repo_path: Local filesystem path to a full (non-shallow) Git clone
        since: Start date (inclusive, timezone-aware UTC)
        to: End date (inclusive, timezone-aware UTC)

    Returns:
        DataFrame with columns: commit_hash, author_date, author_email, msg,
        plus all 17 feature columns from config.FEATURE_COLUMNS

    Raises:
        FileNotFoundError: If repo_path does not contain a .git directory
        ValueError: If since >= to
    """
    # Validation
    git_dir = Path(repo_path) / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(f"No .git directory found at {repo_path}")
    
    if since >= to:
        raise ValueError(f"Invalid date range: since ({since}) must be before to ({to})")

    # Initialize running state for stateful features
    author_state = defaultdict(lambda: {
        "commit_count": 0,
        "fix_count": 0,
        "subsystem_commits": defaultdict(int),
        "commit_dates": []
    })
    
    file_state = defaultdict(lambda: {
        "last_modified": None,
        "developers": set(),
        "change_count": 0
    })

    rows = []
    
    # Traverse commits chronologically (oldest first)
    for commit in Repository(repo_path, since=since, to=to, order='date-order').traverse_commits():
        # Compute features for this commit BEFORE updating state
        features = compute_commit_features(commit, author_state, file_state)
        
        # Build row with metadata + features
        row = {
            "commit_hash": commit.hash,
            "author_date": commit.author_date,
            "author_email": commit.author.email,
            "msg": commit.msg,
            **features
        }
        rows.append(row)
        
        # Update state for next iteration
        _update_state(commit, author_state, file_state)
    
    df = pd.DataFrame(rows)
    
    # Ensure chronological order (should already be, but explicit is better)
    df = df.sort_values("author_date").reset_index(drop=True)
    
    return df


def compute_commit_features(commit: Any, author_state: dict, file_state: dict) -> dict:
    """
    Compute all 17 JIT features for a single commit.

    This function is PURE with respect to the state dicts (reads only, no writes).
    The caller is responsible for updating state AFTER this function returns.

    This is the function imported by action/extract_pr_features.py to guarantee
    train/serve parity (§10.2). DO NOT duplicate this logic elsewhere.

    Args:
        commit: PyDriller Commit object
        author_state: Dict mapping author_email -> {commit_count, fix_count, 
                      subsystem_commits, commit_dates}
        file_state: Dict mapping file_path -> {last_modified, developers, change_count}

    Returns:
        Dict with keys matching config.FEATURE_COLUMNS (17 features)
    """
    author_email = commit.author.email
    author_info = author_state[author_email]
    
    # Basic diff metrics
    la = commit.lines
    ld = commit.deletions
    lt = sum(mf.nloc or 0 for mf in commit.modified_files if mf.nloc is not None)
    
    # Scope metrics
    nf = len(commit.modified_files)
    
    # Get touched file paths and subsystems
    touched_files = [mf.new_path or mf.old_path for mf in commit.modified_files]
    subsystems = set()
    directories = set()
    
    for file_path in touched_files:
        if file_path:
            parts = Path(file_path).parts
            if parts:
                subsystems.add(parts[0])  # Top-level directory
                if len(parts) > 1:
                    directories.add(str(Path(*parts[:-1])))  # All parent dirs
                else:
                    directories.add(parts[0])
    
    ns = len(subsystems)
    nd = len(directories)
    
    # Entropy: distribution of changes across files
    if nf > 1:
        change_sizes = []
        for mf in commit.modified_files:
            size = (mf.added_lines or 0) + (mf.deleted_lines or 0)
            if size > 0:
                change_sizes.append(size)
        
        if change_sizes:
            total = sum(change_sizes)
            probabilities = [s / total for s in change_sizes]
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        else:
            entropy = 0.0
    else:
        entropy = 0.0
    
    # File history metrics (looking back at state before this commit)
    ndev_list = []
    age_list = []
    nuc_list = []
    
    for file_path in touched_files:
        if file_path and file_path in file_state:
            fstate = file_state[file_path]
            ndev_list.append(len(fstate["developers"]))
            
            if fstate["last_modified"]:
                age_days = (commit.author_date - fstate["last_modified"]).total_seconds() / 86400
                age_list.append(max(0.0, age_days))
            
            nuc_list.append(fstate["change_count"])
    
    ndev = sum(ndev_list) if ndev_list else 0
    age = sum(age_list) / len(age_list) if age_list else 0.0
    nuc = sum(nuc_list) if nuc_list else 0
    
    # Author experience metrics
    exp = author_info["commit_count"]
    
    # Recency-weighted experience (exponential decay over 180 days)
    rexp = 0.0
    for past_date in author_info["commit_dates"]:
        days_ago = (commit.author_date - past_date).total_seconds() / 86400
        rexp += math.exp(-days_ago / 180.0)
    
    # Subsystem experience
    current_subsystem = list(subsystems)[0] if subsystems else ""
    sexp = float(author_info["subsystem_commits"].get(current_subsystem, 0))
    
    # Fix ratio
    total_commits = author_info["commit_count"]
    if total_commits > 0:
        fix_ratio_author = author_info["fix_count"] / total_commits
    else:
        fix_ratio_author = 0.0
    
    # Temporal features
    hour_of_day = commit.author_date.hour
    is_weekend = commit.author_date.weekday() >= 5  # 5=Saturday, 6=Sunday
    
    # Message features
    msg = commit.msg or ""
    
    if msg:
        char_counts = Counter(msg)
        total_chars = len(msg)
        probabilities = [count / total_chars for count in char_counts.values()]
        msg_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    else:
        msg_entropy = 0.0
    
    msg_length = len(msg)
    
    return {
        "la": la,
        "ld": ld,
        "lt": lt,
        "nf": nf,
        "ns": ns,
        "nd": nd,
        "entropy": entropy,
        "ndev": ndev,
        "age": age,
        "nuc": nuc,
        "exp": exp,
        "rexp": rexp,
        "sexp": sexp,
        "fix_ratio_author": fix_ratio_author,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "msg_entropy": msg_entropy,
        "msg_length": msg_length,
    }


def _update_state(commit: Any, author_state: dict, file_state: dict) -> None:
    """
    Update running state after processing a commit.

    This mutates author_state and file_state in place.
    Called by mine_commits() AFTER compute_commit_features() returns.

    Args:
        commit: PyDriller Commit object
        author_state: Author tracking dict (mutated in place)
        file_state: File tracking dict (mutated in place)
    """
    author_email = commit.author.email
    
    # Update author state
    author_state[author_email]["commit_count"] += 1
    author_state[author_email]["commit_dates"].append(commit.author_date)
    
    # Check if this is a fix commit (using simple keyword heuristic)
    msg_lower = (commit.msg or "").lower()
    if any(keyword in msg_lower for keyword in ["fix", "bug", "defect"]):
        author_state[author_email]["fix_count"] += 1
    
    # Update subsystem experience
    touched_files = [mf.new_path or mf.old_path for mf in commit.modified_files]
    subsystems = set()
    for file_path in touched_files:
        if file_path:
            parts = Path(file_path).parts
            if parts:
                subsystems.add(parts[0])
    
    for subsystem in subsystems:
        author_state[author_email]["subsystem_commits"][subsystem] += 1
    
    # Update file state
    for file_path in touched_files:
        if file_path:
            file_state[file_path]["last_modified"] = commit.author_date
            file_state[file_path]["developers"].add(author_email)
            file_state[file_path]["change_count"] += 1


if __name__ == "__main__":
    """
    Standalone execution: mine commits from configured repository.
    
    Usage: python -m pipeline.mine
    Output: data/commits.parquet
    """
    from pipeline.config import REPO_LOCAL_PATH, DATE_START, DATE_END
    
    print(f"Mining commits from {REPO_LOCAL_PATH}")
    print(f"Date range: {DATE_START} to {DATE_END}")
    
    df = mine_commits(REPO_LOCAL_PATH, DATE_START, DATE_END)
    
    output_path = "data/commits.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Mining complete!")
    print(f"   Rows: {len(df)}")
    print(f"   Date range: {df['author_date'].min()} to {df['author_date'].max()}")
    print(f"   Output: {output_path}")
    print(f"\nSample features from first commit:")
    feature_cols = ["la", "ld", "nf", "ns", "entropy", "exp", "msg_length"]
    print(df[feature_cols].head(1).to_string(index=False))
