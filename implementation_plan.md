# Implementation Plan — Just-in-Time (JIT) Defect Risk Predictor

**Mode**: PLAN  
**Status**: Ready for approval before ACT mode  
**Derived from**: `implementation.md`, `project_overview.md`, `architecture.md`

---

## Project Goal

Build a Just-In-Time Software Defect Prediction system that scores pull requests for defect risk by training a classifier on commit-level change metadata (using PyDriller mining + SZZ labeling), then automating risk-score comments via a GitHub Action and displaying model performance metrics on a static dashboard.

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Mining | PyDriller 2.x | Purpose-built Git API, avoids brittle git-log parsing |
| Labeling | SZZ (via subprocess git blame) + refactoring filter | Standard research methodology for bug-inducing commit detection |
| Machine Learning | scikit-learn LogisticRegression + XGBoost | Fast training on tabular data, interpretable coefficients (LR) |
| Data Storage | Apache Parquet (pyarrow) | Typed columnar format, avoids CSV dtype bugs |
| Action Runtime | Python 3.11 on GitHub Actions ubuntu-latest | Matches pipeline language, zero infra provisioning |
| PR Comments | GitHub REST API (requests + GITHUB_TOKEN) | Direct API control for idempotent create-or-update logic |
| Dashboard | Static HTML + Chart.js (CDN), GitHub Pages | Zero backend/hosting cost, fits batch metrics model |

---

## Directory Structure

```
jit-defect-predictor/
├── README.md                         # Final documentation (Phase 5)
├── requirements.txt                  # Pinned dependencies
├── .gitignore                        # Excludes data/*.parquet, __pycache__
├── .github/
│   └── workflows/
│       └── risk-score.yml            # PR-triggered scoring workflow
├── pipeline/
│   ├── __init__.py
│   ├── config.py                     # REPO_URL, dates, FIX_PATTERN_REGEX
│   ├── mine.py                       # Feature extraction (Phase 1)
│   ├── label_szz.py                  # SZZ labeling (Phase 2)
│   └── train.py                      # Train/evaluate/persist (Phase 3)
├── action/
│   ├── __init__.py
│   ├── extract_pr_features.py        # Per-PR feature computation (Phase 4)
│   └── score.py                      # Load model, score, comment (Phase 4)
├── dashboard/
│   ├── index.html                    # Static page shell (Phase 4)
│   ├── dashboard.js                  # Renders eval_report.json (Phase 4)
│   └── style.css                     # Dashboard styling (Phase 4)
├── data/                             # Gitignored
│   ├── .gitkeep
│   ├── commits.parquet               # Output of mine.py
│   └── commits_labeled.parquet       # Output of label_szz.py
├── model_artifacts/                  # Committed (small files)
│   ├── model.pkl                     # Selected trained model
│   ├── scaler.pkl                    # Only if LogisticRegression selected
│   ├── metrics.json                  # Full training metrics
│   ├── eval_report.json              # Dashboard-specific subset
│   └── feature_importance.json       # Ranked feature contributions
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   └── sample_commits.py         # In-memory test fixtures
    ├── test_mine.py
    ├── test_label_szz.py
    ├── test_train.py
    └── test_score.py
```

---

## Feature List

All 17 JIT features computed per commit:
- **Diff metrics**: `la` (lines added), `ld` (lines deleted), `lt` (total lines in touched files)
- **Scope metrics**: `nf` (files touched), `ns` (subsystems), `nd` (directories), `entropy` (change distribution)
- **File history**: `ndev` (prior developers on files), `age` (mean days since last change), `nuc` (prior unique changes)
- **Author experience**: `exp` (total prior commits), `rexp` (recent weighted commits), `sexp` (subsystem-specific commits), `fix_ratio_author` (prior fix rate)
- **Temporal**: `hour_of_day` (0-23), `is_weekend` (boolean)
- **Message quality**: `msg_entropy` (Shannon entropy), `msg_length` (character count)

**Target label**: `is_bug_inducing` (0 or 1, from SZZ)

---

## Build Sequence (Ordered Phases)

### Phase 0: Repository Selection & Setup
**What**: Install dependencies, probe candidate repo for fix-commit linkage density, full-clone selected repo, populate config.py  
**Output**: `pipeline/config.py` fully configured; `data/target_repo/.git` exists  
**Success Criteria**: Linkage density ≥8%, full history available for git blame

### Phase 1: Mining
**What**: Implement `pipeline/mine.py` — extract 17 JIT features per commit from target repo history  
**Output**: `data/commits.parquet` (≥2,000 rows)  
**Success Criteria**: Schema matches §6.1 (minus labeling columns), unit tests pass

### Phase 2: Labeling
**What**: Implement `pipeline/label_szz.py` — identify fix commits by regex, filter refactorings, run SZZ blame, join labels  
**Output**: `data/commits_labeled.parquet` (positive class rate 5-30%)  
**Success Criteria**: `is_bug_inducing` label added, foreign-key integrity validated

### Phase 3: Training
**What**: Implement `pipeline/train.py` — chronological split, train LogReg + XGBoost, evaluate both, select best, persist artifacts  
**Output**: `model_artifacts/` directory with model.pkl, metrics.json, eval_report.json, feature_importance.json  
**Success Criteria**: AUC ≥0.65 on test set (SC1) — **hard gate before Phase 4**

### Phase 4: Action & Dashboard
**What**: Build GitHub Action (extract_pr_features.py, score.py, workflow YAML) + static dashboard (HTML/JS/CSS)  
**Output**: PR comment posted within 60s (SC3), idempotent updates (SC4), dashboard live on GitHub Pages (SC5)  
**Success Criteria**: E2E test PR receives correct comment; dashboard matches eval_report.json

### Phase 5: Documentation
**What**: Write README.md with pitch, results table, architecture, features, setup, limitations, citations  
**Output**: README.md complete with exact AUC/precision/recall from metrics.json (SC6)  
**Success Criteria**: Definition of Done fully checked (§8.6)

---

## Key Constraints & Rules

1. **UI/UX First**: Dashboard (frontend) is built in Phase 4, before any additional backend logic — but this is not a multi-layer application; the "UI" here is the static dashboard and the PR comment format. The system is primarily a data pipeline + ML model.

2. **Vertical Slicing**: Each phase delivers one complete slice (data → model → serving) before moving to the next. Do not scaffold all files at once.

3. **One File at a Time**: After approval of this plan, generate files one at a time in the order specified in each phase's task breakdown (see below), waiting for approval between files.

4. **Self-Review Checklist** (per file):
   - **Testing**: Does it need a corresponding test file? Is it testable with fixtures?
   - **Accessibility**: N/A for backend Python; for dashboard HTML, use semantic tags and ARIA where appropriate
   - **Security**: No hardcoded secrets, subprocess calls use `shell=False`, commit hashes validated against regex
   - **Code Quality**: Clear naming, no dead code, typed function signatures, consistent snake_case

5. **No Auto-Features**: Do not add dependencies, tests, or files beyond what each step explicitly calls for without asking first.

6. **Stop Before**: Installing dependencies, modifying build config, touching >1 feature slice at once, or any destructive change (ask first).

---

## Detailed Task Breakdown (Per Phase)

### Phase 0 Tasks (in order)
1. Create `requirements.txt` with pinned versions (§9.2)
2. Create `.gitignore` (excludes data/*.parquet, __pycache__, .env, model_artifacts/*.pkl temporarily until Phase 3)
3. Create `data/.gitkeep` (ensures directory exists)
4. Create `pipeline/__init__.py` (empty)
5. Create `pipeline/config.py` (module constants: REPO_URL, REPO_LOCAL_PATH, DATE_START, DATE_END, FIX_PATTERN_REGEX, REFACTOR_FILTER_THRESHOLD)
6. Create `tests/__init__.py` and `tests/fixtures/__init__.py`

**Approval checkpoint**: Phase 0 complete, config ready

### Phase 1 Tasks (in order)
1. Create `pipeline/mine.py`:
   - `mine_commits(repo_path, since, to) -> pd.DataFrame`
   - `compute_commit_features(commit, author_state, file_state) -> dict`
2. Create `tests/test_mine.py` (unit tests per §11.1)

**Approval checkpoint**: Phase 1 complete, commits.parquet generated

### Phase 2 Tasks (in order)
1. Create `pipeline/label_szz.py`:
   - `identify_fix_commits(commits_df, pattern) -> pd.DataFrame`
   - `apply_refactoring_filter(fix_commits_df, repo_path, threshold) -> pd.DataFrame`
   - `run_szz(fix_commits_df, repo_path) -> pd.DataFrame`
   - `label_commits(commits_df, inducing_pairs_df) -> pd.DataFrame`
2. Create `tests/test_label_szz.py` (unit tests per §11.1)

**Approval checkpoint**: Phase 2 complete, commits_labeled.parquet generated

### Phase 3 Tasks (in order)
1. Create `pipeline/train.py`:
   - `chronological_split(df, date_col, train_frac) -> tuple`
   - `train_logistic_regression(X_train, y_train) -> tuple`
   - `train_xgboost(X_train, y_train) -> XGBClassifier`
   - `evaluate(model, X_test, y_test, scaler) -> dict`
   - `select_and_persist_best(...) -> str`
2. Create `tests/test_train.py` (unit tests per §11.1)
3. Update `.gitignore` to commit model_artifacts/ (no longer temporary)

**Approval checkpoint**: Phase 3 complete, AUC ≥0.65 verified (SC1)

### Phase 4 Tasks (in order)
1. Create `action/__init__.py`
2. Create `action/extract_pr_features.py`:
   - `extract_features_for_pr(repo_path, base_sha, head_sha, author_login) -> dict`
3. Create `action/score.py`:
   - `load_model(model_path, scaler_path) -> tuple`
   - `predict_risk(model, features, scaler, feature_order) -> float`
   - `format_comment(risk_score, top_factors, thresholds) -> str`
   - `post_or_update_comment(github_token, repo, pr_number, body) -> None`
4. Create `tests/test_score.py` (unit tests per §11.1)
5. Create `.github/workflows/risk-score.yml` (workflow definition per §7.3)
6. Create `dashboard/index.html` (static page shell with structure for cards, matrix, chart)
7. Create `dashboard/style.css` (clean, accessible styling)
8. Create `dashboard/dashboard.js` (fetch eval_report.json, render all sections)

**Approval checkpoint**: Phase 4 complete, E2E test PR passes SC3/SC4/SC5

### Phase 5 Tasks (in order)
1. Create `README.md`:
   - Pitch with citations (Kamei et al., SZZ)
   - Results table (exact AUC/precision/recall from metrics.json)
   - Architecture diagram (text or mermaid)
   - Feature explanation (link to implementation.md §6.1)
   - Setup instructions (clone, pip install, run phases)
   - Limitations section (SZZ noise, single-repo, no real-time retraining)
   - License and citations
2. Final Definition of Done checklist review (§8.6)

**Approval checkpoint**: Phase 5 complete, project shippable

---

## Success Criteria Summary

| ID | Criterion | Measured By |
|---|---|---|
| SC1 | AUC ≥ 0.65 on test set | `model_artifacts/metrics.json.selected_model.auc` |
| SC2 | ≥2,000 commits labeled, 5-30% positive rate | `data/commits_labeled.parquet` row count and mean(`is_bug_inducing`) |
| SC3 | PR comment within 60s | GitHub Actions run duration logs |
| SC4 | Idempotent comment (edits, not duplicates) | Manual E2E test verification |
| SC5 | Dashboard metrics match eval_report.json | Manual diff during acceptance testing |
| SC6 | README states exact AUC, repo, date range | Manual review against Definition of Done |

---

## What Happens Next

**If you approve this plan**, I will switch to **ACT mode** and begin executing Phase 0, Task 1 (create requirements.txt). I will generate one file per response and wait for your go-ahead before proceeding to the next file.

**If you want changes**, let me know which sections to revise before we begin.

---

**Ready to proceed?** Say "Act" or "approved" to begin implementation.
