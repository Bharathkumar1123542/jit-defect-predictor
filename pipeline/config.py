"""
Configuration constants for the JIT defect prediction pipeline.

Single source of truth for repository target, date range, and labeling patterns.
All constants are module-level (no functions) and must be set before running
any pipeline phase.

Instructions:
1. Set REPO_URL to a public Git repository with consistent bug-fix commit conventions
2. Set DATE_START/DATE_END to bound the commit window (timezone-aware UTC)
3. Adjust FIX_PATTERN_REGEX to match the target repo's issue/fix conventions
4. Run Phase 0 probe to verify ≥8% linkage density before proceeding to Phase 1
"""

from datetime import datetime, timezone

# Target repository
REPO_URL = "https://github.com/apache/commons-lang.git"
REPO_LOCAL_PATH = "data/target_repo"

# Date range for commit mining (inclusive, timezone-aware UTC)
# Initial configuration targets a 2-year window for ~2,000-10,000 commits
DATE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
DATE_END = datetime(2022, 12, 31, tzinfo=timezone.utc)

# Bug-fix commit identification pattern (case-insensitive)
# Matches: "fix", "fixes", "fixed", "bug", "defect", "resolve", "resolved", "resolves"
# Plus Jira-style issue IDs like "LANG-1234"
FIX_PATTERN_REGEX = r"\b(fix|fixes|fixed|bug|defect|resolve[sd]?)\b|LANG-\d+"

# Refactoring filter threshold
# Fix commits with (whitespace + comment-only changes) / total changes > threshold
# are excluded from SZZ to reduce false positives
REFACTOR_FILTER_THRESHOLD = 0.8

# Feature names (defines column order for train/serve consistency)
FEATURE_COLUMNS = [
    "la",                  # Lines added
    "ld",                  # Lines deleted
    "lt",                  # Total lines in touched files
    "nf",                  # Number of files touched
    "ns",                  # Number of subsystems (top-level dirs)
    "nd",                  # Number of directories touched
    "entropy",             # Shannon entropy of change distribution
    "ndev",                # Number of prior developers on touched files
    "age",                 # Mean days since touched files last changed
    "nuc",                 # Number of unique prior changes to touched files
    "exp",                 # Author's total prior commit count
    "rexp",                # Author's recency-weighted prior commit count
    "sexp",                # Author's subsystem-specific prior commit count
    "fix_ratio_author",    # Author's historical fix-commit ratio
    "hour_of_day",         # Commit hour (0-23)
    "is_weekend",          # True if Sat/Sun
    "msg_entropy",         # Shannon entropy of commit message
    "msg_length",          # Character length of commit message
]

# Model artifact paths
MODEL_ARTIFACTS_DIR = "model_artifacts"
MODEL_PATH = f"{MODEL_ARTIFACTS_DIR}/model.pkl"
SCALER_PATH = f"{MODEL_ARTIFACTS_DIR}/scaler.pkl"
METRICS_PATH = f"{MODEL_ARTIFACTS_DIR}/metrics.json"
EVAL_REPORT_PATH = f"{MODEL_ARTIFACTS_DIR}/eval_report.json"
FEATURE_IMPORTANCE_PATH = f"{MODEL_ARTIFACTS_DIR}/feature_importance.json"

# Train/test split configuration
TRAIN_FRACTION = 0.8  # Chronological split: first 80% train, last 20% test

# Model performance thresholds
MIN_AUC_THRESHOLD = 0.65  # Success Criterion 1 (SC1)
MIN_COMMITS = 2000         # Success Criterion 2 (SC2) - minimum dataset size
MIN_POSITIVE_RATE = 0.05   # Success Criterion 2 (SC2) - minimum bug-inducing rate
MAX_POSITIVE_RATE = 0.30   # Success Criterion 2 (SC2) - maximum bug-inducing rate
