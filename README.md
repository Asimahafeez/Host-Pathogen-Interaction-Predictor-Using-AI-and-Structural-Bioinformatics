# Host-Pathogen Interaction Predictor (Pro)

Advanced Streamlit app to predict host–pathogen protein interaction likelihood using sequence-based and structural-inspired features.

## Features
- Sequence-based feature engineering (AA composition, hydrophobicity, BLOSUM62 score, percent identity)
- Model training (Random Forest) with evaluation (ROC AUC, accuracy)
- Permutation-based feature importance
- Interactive network visualization of top predicted interactions
- Demo synthetic dataset included

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Input CSV columns (for training): host_id, pathogen_id, host_seq, pathogen_seq, label (0/1)
For prediction-only: omit 'label'.

Research prototype — not for clinical use.
