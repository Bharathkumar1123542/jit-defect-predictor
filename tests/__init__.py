"""
Test suite for JIT Defect Prediction system.

Test organization:
- test_mine.py: Unit tests for feature extraction (pipeline.mine)
- test_label_szz.py: Unit tests for SZZ labeling (pipeline.label_szz)
- test_train.py: Unit tests for model training/evaluation (pipeline.train)
- test_score.py: Unit tests for PR scoring and commenting (action.score)
- fixtures/: In-memory test fixtures (no network/repo dependencies)

Testing philosophy (per §11.1):
- All tests use in-memory fixtures (no live Git operations)
- No network calls (GitHub API calls are mocked)
- Tests validate contracts, edge cases, and error handling
- Feature computation logic is tested with hand-computed expected values
"""
