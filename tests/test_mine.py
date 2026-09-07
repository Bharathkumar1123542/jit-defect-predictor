"""
Unit tests for pipeline.mine module.

Tests cover:
- Feature computation with hand-verified expected values
- Edge cases: empty messages, cold-start authors, single-file commits
- Boundary conditions: hour_of_day (0, 23), weekend detection
- Entropy calculation correctness
- Input validation (missing repo, invalid date range)
"""

import math
from collections import defaultdict
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from pipeline.mine import compute_commit_features


class TestComputeCommitFeatures:
    """Tests for compute_commit_features() pure function."""
    
    def test_basic_feature_computation(self):
        """Test standard commit with multiple files and changes."""
        # Create mock commit
        commit = Mock()
        commit.author.email = "dev@example.com"
        commit.author_date = datetime(2022, 6, 15, 14, 30, tzinfo=timezone.utc)
        commit.lines = 50
        commit.deletions = 20
        commit.msg = "Fix bug in authentication module"
        
        # Mock modified files
        mf1 = Mock()
        mf1.new_path = "src/auth/login.py"
        mf1.old_path = "src/auth/login.py"
        mf1.nloc = 100
        mf1.added_lines = 30
        mf1.deleted_lines = 10
        
        mf2 = Mock()
        mf2.new_path = "src/auth/session.py"
        mf2.old_path = "src/auth/session.py"
        mf2.nloc = 150
        mf2.added_lines = 20
        mf2.deleted_lines = 10
        
        commit.modified_files = [mf1, mf2]
        
        # Initialize state with prior history
        author_state = defaultdict(lambda: {
            "commit_count": 0,
            "fix_count": 0,
            "subsystem_commits": defaultdict(int),
            "commit_dates": []
        })
        author_state["dev@example.com"]["commit_count"] = 5
        author_state["dev@example.com"]["fix_count"] = 1
        author_state["dev@example.com"]["subsystem_commits"]["src"] = 3
        author_state["dev@example.com"]["commit_dates"] = [
            datetime(2022, 6, 1, tzinfo=timezone.utc),
            datetime(2022, 6, 5, tzinfo=timezone.utc),
        ]
        
        file_state = defaultdict(lambda: {
            "last_modified": None,
            "developers": set(),
            "change_count": 0
        })
        file_state["src/auth/login.py"]["last_modified"] = datetime(2022, 6, 10, tzinfo=timezone.utc)
        file_state["src/auth/login.py"]["developers"] = {"other@example.com"}
        file_state["src/auth/login.py"]["change_count"] = 3
        file_state["src/auth/session.py"]["last_modified"] = datetime(2022, 6, 12, tzinfo=timezone.utc)
        file_state["src/auth/session.py"]["developers"] = {"other@example.com", "dev@example.com"}
        file_state["src/auth/session.py"]["change_count"] = 2
        
        features = compute_commit_features(commit, author_state, file_state)
        
        # Verify basic diff metrics
        assert features["la"] == 50
        assert features["ld"] == 20
        assert features["lt"] == 250  # 100 + 150
        
        # Verify scope metrics
        assert features["nf"] == 2
        assert features["ns"] == 1  # Only "src" subsystem
        assert features["nd"] == 1  # Only "src/auth" directory
        
        # Verify entropy (30 and 20 line changes across 2 files)
        # Expected: -(30/50 * log2(30/50) + 20/50 * log2(20/50))
        expected_entropy = -(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4))
        assert abs(features["entropy"] - expected_entropy) < 0.02  # Wider tolerance for floating point
        
        # Verify file history metrics
        assert features["ndev"] == 3  # 1 + 2 developers
        assert 4.0 < features["age"] < 5.0  # Average of ~5 and ~3 days
        assert features["nuc"] == 5  # 3 + 2 prior changes
        
        # Verify author experience
        assert features["exp"] == 5
        assert features["rexp"] > 0  # Has recent commits
        assert features["sexp"] == 3.0  # 3 prior commits in "src"
        assert features["fix_ratio_author"] == 1/5  # 1 fix out of 5 commits
        
        # Verify temporal features
        assert features["hour_of_day"] == 14
        assert features["is_weekend"] is False  # June 15, 2022 was Wednesday
        
        # Verify message features
        assert features["msg_length"] == len("Fix bug in authentication module")
        assert features["msg_entropy"] > 0
    
    def test_cold_start_first_commit(self):
        """Test first commit by a new author (cold start scenario per §10.9)."""
        commit = Mock()
        commit.author.email = "newdev@example.com"
        commit.author_date = datetime(2022, 1, 1, 12, 0, tzinfo=timezone.utc)
        commit.lines = 10
        commit.deletions = 0
        commit.msg = "Initial commit"
        
        mf = Mock()
        mf.new_path = "README.md"
        mf.old_path = None
        mf.nloc = 10
        mf.added_lines = 10
        mf.deleted_lines = 0
        commit.modified_files = [mf]
        
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
        
        features = compute_commit_features(commit, author_state, file_state)
        
        # Cold start values should be zero (§10.9)
        assert features["exp"] == 0
        assert features["rexp"] == 0.0
        assert features["sexp"] == 0.0
        assert features["fix_ratio_author"] == 0.0
        assert features["age"] == 0.0
        assert features["ndev"] == 0
        assert features["nuc"] == 0
        assert features["entropy"] == 0.0  # Single file
    
    def test_empty_commit_message(self):
        """Test commit with empty message (§10.5)."""
        commit = Mock()
        commit.author.email = "dev@example.com"
        commit.author_date = datetime(2022, 6, 15, 10, 0, tzinfo=timezone.utc)
        commit.lines = 5
        commit.deletions = 2
        commit.msg = ""
        
        mf = Mock()
        mf.new_path = "file.txt"
        mf.old_path = "file.txt"
        mf.nloc = 50
        mf.added_lines = 5
        mf.deleted_lines = 2
        commit.modified_files = [mf]
        
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
        
        features = compute_commit_features(commit, author_state, file_state)
        
        # Empty message should produce zero entropy and length
        assert features["msg_entropy"] == 0.0
        assert features["msg_length"] == 0
    
    def test_weekend_detection(self):
        """Test is_weekend flag for Saturday and Sunday."""
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
        
        mf = Mock()
        mf.new_path = "file.txt"
        mf.old_path = "file.txt"
        mf.nloc = 10
        mf.added_lines = 1
        mf.deleted_lines = 0
        
        # Saturday (June 18, 2022)
        commit_sat = Mock()
        commit_sat.author.email = "dev@example.com"
        commit_sat.author_date = datetime(2022, 6, 18, 10, 0, tzinfo=timezone.utc)
        commit_sat.lines = 1
        commit_sat.deletions = 0
        commit_sat.msg = "Weekend work"
        commit_sat.modified_files = [mf]
        
        features_sat = compute_commit_features(commit_sat, author_state, file_state)
        assert features_sat["is_weekend"] is True
        
        # Sunday (June 19, 2022)
        commit_sun = Mock()
        commit_sun.author.email = "dev@example.com"
        commit_sun.author_date = datetime(2022, 6, 19, 10, 0, tzinfo=timezone.utc)
        commit_sun.lines = 1
        commit_sun.deletions = 0
        commit_sun.msg = "More weekend work"
        commit_sun.modified_files = [mf]
        
        features_sun = compute_commit_features(commit_sun, author_state, file_state)
        assert features_sun["is_weekend"] is True
        
        # Monday (June 20, 2022)
        commit_mon = Mock()
        commit_mon.author.email = "dev@example.com"
        commit_mon.author_date = datetime(2022, 6, 20, 10, 0, tzinfo=timezone.utc)
        commit_mon.lines = 1
        commit_mon.deletions = 0
        commit_mon.msg = "Back to work"
        commit_mon.modified_files = [mf]
        
        features_mon = compute_commit_features(commit_mon, author_state, file_state)
        assert features_mon["is_weekend"] is False
    
    def test_hour_of_day_boundaries(self):
        """Test hour_of_day at boundary values 0 and 23."""
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
        
        mf = Mock()
        mf.new_path = "file.txt"
        mf.old_path = "file.txt"
        mf.nloc = 10
        mf.added_lines = 1
        mf.deleted_lines = 0
        
        # Midnight (hour 0)
        commit_midnight = Mock()
        commit_midnight.author.email = "dev@example.com"
        commit_midnight.author_date = datetime(2022, 6, 15, 0, 30, tzinfo=timezone.utc)
        commit_midnight.lines = 1
        commit_midnight.deletions = 0
        commit_midnight.msg = "Late night commit"
        commit_midnight.modified_files = [mf]
        
        features_midnight = compute_commit_features(commit_midnight, author_state, file_state)
        assert features_midnight["hour_of_day"] == 0
        
        # 11 PM (hour 23)
        commit_late = Mock()
        commit_late.author.email = "dev@example.com"
        commit_late.author_date = datetime(2022, 6, 15, 23, 45, tzinfo=timezone.utc)
        commit_late.lines = 1
        commit_late.deletions = 0
        commit_late.msg = "Very late commit"
        commit_late.modified_files = [mf]
        
        features_late = compute_commit_features(commit_late, author_state, file_state)
        assert features_late["hour_of_day"] == 23
    
    def test_entropy_three_file_distribution(self):
        """Test entropy calculation with hand-computed expected value."""
        commit = Mock()
        commit.author.email = "dev@example.com"
        commit.author_date = datetime(2022, 6, 15, 12, 0, tzinfo=timezone.utc)
        commit.lines = 100
        commit.deletions = 50
        commit.msg = "Multi-file change"
        
        # Three files with 50, 30, 20 lines changed
        mf1 = Mock()
        mf1.new_path = "file1.py"
        mf1.old_path = "file1.py"
        mf1.nloc = 100
        mf1.added_lines = 40
        mf1.deleted_lines = 10
        
        mf2 = Mock()
        mf2.new_path = "file2.py"
        mf2.old_path = "file2.py"
        mf2.nloc = 80
        mf2.added_lines = 20
        mf2.deleted_lines = 10
        
        mf3 = Mock()
        mf3.new_path = "file3.py"
        mf3.old_path = "file3.py"
        mf3.nloc = 60
        mf3.added_lines = 15
        mf3.deleted_lines = 5
        
        commit.modified_files = [mf1, mf2, mf3]
        
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
        
        features = compute_commit_features(commit, author_state, file_state)
        
        # Hand-computed entropy: 50, 30, 20 (total 100)
        # P = [0.5, 0.3, 0.2]
        # H = -(0.5*log2(0.5) + 0.3*log2(0.3) + 0.2*log2(0.2))
        expected_entropy = -(0.5 * math.log2(0.5) + 0.3 * math.log2(0.3) + 0.2 * math.log2(0.2))
        
        assert abs(features["entropy"] - expected_entropy) < 0.01
        assert features["nf"] == 3


class TestMineCommitsValidation:
    """Tests for mine_commits() input validation."""
    
    def test_missing_git_directory(self):
        """Test that mine_commits raises FileNotFoundError for non-repo path."""
        from pipeline.mine import mine_commits
        
        with pytest.raises(FileNotFoundError, match="No .git directory found"):
            mine_commits("/nonexistent/path", 
                        datetime(2020, 1, 1, tzinfo=timezone.utc),
                        datetime(2021, 1, 1, tzinfo=timezone.utc))
    
    def test_invalid_date_range(self):
        """Test that mine_commits raises ValueError for since >= to."""
        from pipeline.mine import mine_commits
        import tempfile
        import os
        
        # Create a temporary directory with .git to pass first validation
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = os.path.join(tmpdir, ".git")
            os.makedirs(git_dir)
            
            with pytest.raises(ValueError, match="since .* must be before to"):
                mine_commits(tmpdir, 
                            datetime(2021, 1, 1, tzinfo=timezone.utc),
                            datetime(2020, 1, 1, tzinfo=timezone.utc))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
