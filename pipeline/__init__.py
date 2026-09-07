"""
Pipeline package for JIT Defect Prediction.

Contains offline components:
- config.py: Repository and feature configuration
- mine.py: Commit feature extraction via PyDriller
- label_szz.py: Bug-inducing commit labeling via SZZ algorithm
- train.py: Model training, evaluation, and persistence
"""
