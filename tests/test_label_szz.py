"""
Unit tests for pipeline.label_szz module.

Tests cover:
- Fix commit identification with regex patterns (true positives, true negatives, near-misses)
- Refactoring filter logic (cosmetic vs substantive changes)
- Label joining correctness (row count preservation, foreign key integrity)
- Edge cases: empty results, duplicate inducing commits
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from pipeline.label_szz import (
    identify_fix_commits,
    apply_refactoring_filter,
    label_commits,
    _looks_like_comment,
    _compute_cosmetic_ratio
)


class TestIdentifyFixCommits:
    """Tests for fix commit identification by message pattern."""
    
    def test_basic_fix_keywords(self):
        """Test pattern matches common fix keywords."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40, "d" * 40],
            "msg": [
                "Fix bug in login",
                "Add new feature",
                "Fixed memory leak",
                "Refactor code"
            ]
        })
        
        pattern = r"\b(fix|fixes|fixed|bug)\b"
        fixes = identify_fix_commits(commits_df, pattern)
        
        assert len(fixes) == 2
        assert set(fixes["msg"]) == {"Fix bug in login", "Fixed memory leak"}
    
    def test_case_insensitive_matching(self):
        """Test pattern matching is case-insensitive."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40],
            "msg": [
                "FIX: uppercase",
                "fix: lowercase",
                "Fix: mixed case"
            ]
        })
        
        pattern = r"\bfix\b"
        fixes = identify_fix_commits(commits_df, pattern)
        
        assert len(fixes) == 3
    
    def test_issue_tracker_patterns(self):
        """Test pattern matches issue tracker IDs (e.g., LANG-1234)."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40],
            "msg": [
                "LANG-123: fix authentication",
                "Update documentation",
                "Resolves LANG-456"
            ]
        })
        
        pattern = r"LANG-\d+"
        fixes = identify_fix_commits(commits_df, pattern)
        
        assert len(fixes) == 2
        assert "LANG-123" in fixes.iloc[0]["msg"]
        assert "LANG-456" in fixes.iloc[1]["msg"]
    
    def test_word_boundary_prevents_false_matches(self):
        """Test word boundaries prevent matching within words."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40],
            "msg": [
                "Fix the bug",           # Should match
                "Prefix and suffix",     # Should NOT match "fix" in "prefix"
                "bugfoot is a word"      # Should NOT match "bug" in "bugfoot"
            ]
        })
        
        pattern = r"\b(fix|bug)\b"
        fixes = identify_fix_commits(commits_df, pattern)
        
        # Only "Fix the bug" should match
        assert len(fixes) == 1
        assert fixes.iloc[0]["msg"] == "Fix the bug"
    
    def test_empty_result_when_no_matches(self):
        """Test returns empty DataFrame when no commits match."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40],
            "msg": [
                "Add feature",
                "Update docs"
            ]
        })
        
        pattern = r"\b(fix|bug)\b"
        fixes = identify_fix_commits(commits_df, pattern)
        
        assert len(fixes) == 0
        assert list(fixes.columns) == ["commit_hash", "msg"]


class TestRefactoringFilter:
    """Tests for cosmetic change filtering."""
    
    def test_looks_like_comment_detection(self):
        """Test comment pattern detection for multiple languages."""
        # Comments that should be detected
        assert _looks_like_comment("// This is a C++ comment")
        assert _looks_like_comment("# Python comment")
        assert _looks_like_comment("/* Block comment start")
        assert _looks_like_comment("*/ Block comment end")
        assert _looks_like_comment("* Javadoc line")
        assert _looks_like_comment("<!-- HTML comment")
        assert _looks_like_comment("--> HTML comment end")
        
        # Non-comments
        assert not _looks_like_comment("int x = 5;")
        assert not _looks_like_comment("def function():")
        assert not _looks_like_comment("console.log('test');")
    
    @patch("pipeline.label_szz.subprocess.run")
    def test_compute_cosmetic_ratio_high_whitespace(self, mock_run):
        """Test cosmetic ratio for mostly whitespace changes."""
        # Mock git diff output with 8 whitespace lines out of 10 total
        mock_run.return_value = Mock(
            stdout="\n".join([
                "+++ b/file.py",
                "@@ -1,5 +1,5 @@",
                "+",           # Whitespace
                "+    ",       # Whitespace
                "+ # Comment", # Comment
                "+",           # Whitespace
                "+",           # Whitespace
                "+ int x = 5", # Actual code
                "+",           # Whitespace
                "+",           # Whitespace
                "+ # Another", # Comment
                "+",           # Whitespace
            ])
        )
        
        ratio = _compute_cosmetic_ratio("a" * 40, "/repo/path")
        
        # 9 cosmetic out of 10 total (8 whitespace + 1 comment) = 0.9
        assert ratio == 0.9
    
    @patch("pipeline.label_szz.subprocess.run")
    def test_compute_cosmetic_ratio_substantive_changes(self, mock_run):
        """Test cosmetic ratio for mostly code changes."""
        mock_run.return_value = Mock(
            stdout="\n".join([
                "+++ b/file.py",
                "@@ -1,5 +1,5 @@",
                "+ int x = 5",
                "+ int y = 10",
                "+ return x + y",
                "+",           # One whitespace line
                "+ print(result)",
            ])
        )
        
        ratio = _compute_cosmetic_ratio("a" * 40, "/repo/path")
        
        # 1 cosmetic out of 5 total = 0.2
        assert ratio == 0.2
    
    @patch("pipeline.label_szz.subprocess.run")
    def test_compute_cosmetic_ratio_subprocess_failure(self, mock_run):
        """Test returns 0.0 when git command fails (don't filter)."""
        mock_run.side_effect = Exception("Git command failed")
        
        ratio = _compute_cosmetic_ratio("a" * 40, "/repo/path")
        
        # Failure returns 0.0 (assume substantive)
        assert ratio == 0.0
    
    @patch("pipeline.label_szz._compute_cosmetic_ratio")
    def test_apply_refactoring_filter_threshold(self, mock_ratio):
        """Test filter excludes commits above threshold."""
        fix_commits = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40],
        })
        
        # Mock ratios: 0.9 (cosmetic), 0.5 (substantive), 0.85 (cosmetic)
        mock_ratio.side_effect = [0.9, 0.5, 0.85]
        
        filtered = apply_refactoring_filter(fix_commits, "/repo", threshold=0.8)
        
        # Only commit "b" (ratio 0.5) should remain
        assert len(filtered) == 1
        assert filtered.iloc[0]["commit_hash"] == "b" * 40


class TestLabelCommits:
    """Tests for label joining logic."""
    
    def test_label_commits_basic(self):
        """Test basic labeling with some inducing commits."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40, "d" * 40],
            "msg": ["Commit A", "Commit B", "Commit C", "Commit D"]
        })
        
        inducing_pairs_df = pd.DataFrame({
            "inducing_commit_hash": ["a" * 40, "c" * 40],
            "fix_commit_hash": ["d" * 40, "d" * 40]
        })
        
        labeled = label_commits(commits_df, inducing_pairs_df)
        
        # Check row count unchanged
        assert len(labeled) == 4
        
        # Check columns added
        assert "is_bug_inducing" in labeled.columns
        assert "fix_commit_hash" in labeled.columns
        
        # Check labels
        assert labeled[labeled["commit_hash"] == "a" * 40]["is_bug_inducing"].iloc[0] == 1
        assert labeled[labeled["commit_hash"] == "b" * 40]["is_bug_inducing"].iloc[0] == 0
        assert labeled[labeled["commit_hash"] == "c" * 40]["is_bug_inducing"].iloc[0] == 1
        assert labeled[labeled["commit_hash"] == "d" * 40]["is_bug_inducing"].iloc[0] == 0
    
    def test_label_commits_preserves_row_count(self):
        """Test labeling never changes the number of rows."""
        commits_df = pd.DataFrame({
            "commit_hash": [f"{i:040x}" for i in range(100)],
            "msg": [f"Commit {i}" for i in range(100)]
        })
        
        inducing_pairs_df = pd.DataFrame({
            "inducing_commit_hash": [f"{i:040x}" for i in range(10)],
            "fix_commit_hash": [f"{(i+50):040x}" for i in range(10)]
        })
        
        labeled = label_commits(commits_df, inducing_pairs_df)
        
        assert len(labeled) == len(commits_df)
    
    def test_label_commits_foreign_key_integrity(self):
        """Test fix_commit_hash values exist in the commit set."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40],
            "msg": ["A", "B", "C"]
        })
        
        inducing_pairs_df = pd.DataFrame({
            "inducing_commit_hash": ["a" * 40],
            "fix_commit_hash": ["c" * 40]
        })
        
        labeled = label_commits(commits_df, inducing_pairs_df)
        
        # All non-null fix_commit_hash values should exist in commit_hash
        fix_hashes = labeled[labeled["fix_commit_hash"].notna()]["fix_commit_hash"]
        all_hashes = set(labeled["commit_hash"])
        
        assert all(fh in all_hashes for fh in fix_hashes)
    
    def test_label_commits_empty_inducing_pairs(self):
        """Test labeling with no inducing commits (all clean)."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40],
            "msg": ["A", "B"]
        })
        
        inducing_pairs_df = pd.DataFrame(columns=["inducing_commit_hash", "fix_commit_hash"])
        
        labeled = label_commits(commits_df, inducing_pairs_df)
        
        # All should be labeled 0 (clean)
        assert len(labeled) == 2
        assert labeled["is_bug_inducing"].sum() == 0
        assert labeled["fix_commit_hash"].isna().all()
    
    def test_label_commits_duplicate_inducing(self):
        """Test commit induced by multiple fixes keeps only one fix_commit_hash."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40, "c" * 40],
            "msg": ["A", "B", "C"]
        })
        
        # Commit "a" induced two different bugs (fixed by "b" and "c")
        inducing_pairs_df = pd.DataFrame({
            "inducing_commit_hash": ["a" * 40, "a" * 40],
            "fix_commit_hash": ["b" * 40, "c" * 40]
        })
        
        labeled = label_commits(commits_df, inducing_pairs_df)
        
        # Commit "a" should be labeled as inducing
        assert labeled[labeled["commit_hash"] == "a" * 40]["is_bug_inducing"].iloc[0] == 1
        
        # It should have exactly one fix_commit_hash (either "b" or "c", implementation picks first)
        fix_hash = labeled[labeled["commit_hash"] == "a" * 40]["fix_commit_hash"].iloc[0]
        assert fix_hash in ["b" * 40, "c" * 40]
    
    def test_label_commits_data_types(self):
        """Test output column data types match specification (§6.1)."""
        commits_df = pd.DataFrame({
            "commit_hash": ["a" * 40, "b" * 40],
            "msg": ["A", "B"]
        })
        
        inducing_pairs_df = pd.DataFrame({
            "inducing_commit_hash": ["a" * 40],
            "fix_commit_hash": ["b" * 40]
        })
        
        labeled = label_commits(commits_df, inducing_pairs_df)
        
        # is_bug_inducing should be int8
        assert labeled["is_bug_inducing"].dtype == "int8"
        
        # Values should be 0 or 1 only
        assert set(labeled["is_bug_inducing"].unique()).issubset({0, 1})
        
        # fix_commit_hash should be string (object dtype in pandas)
        assert labeled["fix_commit_hash"].dtype == "object"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
