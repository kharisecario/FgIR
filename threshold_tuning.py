# -*- coding: utf-8 -*-
"""
Created on Tue June 23 2026
@author: khari

This script is for retraining model with threshold problem specifically
The list of FG choosen is based on previous MCC PR-AUC results.

Dependency:
- cuml from Rapids
- pandas
- cupy
- sklearn
- numpy
- joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import (matthews_corrcoef, average_precision_score,
                             precision_score, recall_score, f1_score)
from basefunction import notify


# ── Same data setup as your original script ──────────────────────────────────
X     = pd.read_csv('XYsnv.csv')
Y_ori = pd.read_csv('label_fg_present.csv')

less_appear_label = ['alkyl_halide','allylic_oxid','aniline','aryl_methyl','halogen',
                     'ketone_Topliss','methoxy','NH0','nitro_arom_nonortho','unbrch_alkane']
Y = Y_ori[less_appear_label]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Results placeholder ───────────────────────────────────────────────────────
threshold_results = {}

for fg_name in Y.columns.to_list():
    print(f'\n>>> Tuning threshold for {fg_name}...')
    dy = Y[fg_name]

    # Same split as original — same random_state is critical!
    X_train, X_test, dy_train, dy_test = train_test_split(
        X, dy, test_size=0.3, random_state=34, stratify=dy  # must match original
    )

    # ── Load saved model ──────────────────────────────────────────────────────
    model = joblib.load(f'pickled_model/cuml_{fg_name}.pkl')

    # ── OOF probabilities for threshold search ────────────────────────────────
    # cross_val_predict refits model on each fold — gives unbiased proba on X_train
    oof_proba = cross_val_predict(model, X_train, dy_train,
                                  cv=cv, method='predict_proba')[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.01)
    mcc_scores = [matthews_corrcoef(dy_train, (oof_proba >= t).astype(int))
                  for t in thresholds]

    best_thresh = thresholds[np.argmax(mcc_scores)]
    oof_mcc     = max(mcc_scores)
    print(f'    Default(0.5) MCC : {matthews_corrcoef(dy_train, (oof_proba >= 0.5).astype(int)):.4f}')
    print(f'    Best threshold   : {best_thresh:.2f}')
    print(f'    OOF MCC          : {oof_mcc:.4f}')

    # ── Evaluate on test set with tuned threshold ─────────────────────────────
    test_proba    = model.predict_proba(X_test).to_numpy()[:, 1]
    dy_pred_tuned = (test_proba >= best_thresh).astype(int)
    dy_pred_default = (test_proba >= 0.5).astype(int)

    threshold_results[fg_name] = {
        'best_threshold'  : best_thresh,
        'OOF_MCC'         : oof_mcc,
        # Compare tuned vs default side-by-side
        'MCC_default'     : matthews_corrcoef(dy_test, dy_pred_default),
        'MCC_tuned'       : matthews_corrcoef(dy_test, dy_pred_tuned),
        'PR_AUC'          : average_precision_score(dy_test, test_proba),
        'F1_tuned'        : f1_score(dy_test, dy_pred_tuned, average='weighted'),
        'Precision_tuned' : precision_score(dy_test, dy_pred_tuned, average='weighted'),
        'Recall_tuned'    : recall_score(dy_test, dy_pred_tuned, average='weighted'),
    }

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(threshold_results).T
results_df.to_csv('threshold_tuning_results.csv', index=True)
print('\n>>> Saved to threshold_tuning_results.csv')
print(results_df[['best_threshold', 'MCC_default', 'MCC_tuned', 'PR_AUC']])

notify(f"👌😎🎶 Program complete! \n Total time: ? hour(s)")
