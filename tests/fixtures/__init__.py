"""
Test fixtures for JIT Defect Prediction tests.

Contains in-memory fixture objects that mimic PyDriller's Commit and ModifiedFile
structures without requiring a real Git repository or network access.

Fixture design:
- Plain Python dicts or dataclasses (not actual PyDriller objects)
- Represent realistic commit scenarios (normal case, edge cases, boundary conditions)
- Hand-verified expected values for feature computation tests
- No file I/O or subprocess calls
"""
