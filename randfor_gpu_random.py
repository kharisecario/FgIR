# -*- coding: utf-8 -*-
"""
Created on Mon May 11 2026 
@author: khari

This script is for LOOPING single label random forest using cuml
do random search first, followed by gridsearchCV for top 10 params
made to shorten the training time

Dependency:
- cuml from Rapids
- pandas
- cupy
- sklearn
- numpy

"""
### Import all required dependencies
import os, time, joblib, shutil
from pathlib import Path            # For creating backup
from datetime import datetime       # For creating backup
from cuml import RandomForestClassifier
from cuml.model_selection import GridSearchCV
from cuml.metrics import confusion_matrix
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, average_precision_score,
                             roc_auc_score, f1_score)
import pandas as pd
from basefunction import notify


####################
#  Functions Here  # ─── Put all functions here  ───────────────
####################

# Supress user warning >>> UserWarning: To use pickling first train using float32 data to fit the estimator ret = func(*args, **kwargs)
import warnings
warnings.filterwarnings("ignore")

### Defining functions used in this file
# Taking results from Phase 1 and build parameter grid for Phase 2
def build_refined_grid(results_df, top_n=10, expand=1):
    """Extract dominant values from top_n results per parameter."""
    top = results_df.head(top_n)
    refined = {}
    numeric_params = ["max_depth", "n_estimators", "min_samples_leaf", "min_samples_split"]
    categorical_params = ["max_features"]

    for param in numeric_params:
        vals = top[param].dropna().unique()
        # If dont want to add neighbor, use this
        refined[param] = sorted(int(v) for v in vals)
        if top[param].isna().any():
            refined[param].append(None)
        ''' 
        # If want to add neighbor uncomment this part
        vals_sorted = sorted(int(v) for v in vals) # convert np.int64 → int
        # optionally add neighbors
        neighbors = set(vals_sorted)
        for v in vals_sorted:
            neighbors.add(max(1, v - expand)) 
            neighbors.add(v + expand) 
        refined[param] = sorted(neighbors)
        if top[param].isna().any():
            refined[param].append(None)  # retain None if it appeared in top
        '''

    for param in categorical_params:
        refined[param] = list(top[param].unique())

    return refined

####################
#   Backup Prep    # ─── Create Backup before Start  ───────────────
####################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = Path('backups')

# ─── Backup a directory if exist and create it if not  ───────────────
out_dir = ['pickled_model', 'confusion_matrix', 'model_performance', 'feature_importances']
for directory in out_dir:
    dir_path = Path(directory)
    if dir_path.exists():
        backup_path = backup_dir / f"{dir_path.name}_{timestamp}"
        shutil.copytree(dir_path, backup_path)
    dir_path.mkdir(parents=True, exist_ok=True)

# ─── Prepare a backup for consecutive output files  ───────────────
files_to_backup = ['randfor_performance.csv', 'feature_importances.csv']
backup_dir.mkdir(parents=True, exist_ok=True)
for file in files_to_backup:
    file_path = Path(file)
    if file_path.exists():
        backup_path = backup_dir / f"{file_path.stem}_{timestamp}.bak"
        shutil.copy2(file_path, backup_path)


####################
#   Preparation    # ─── Select FG and Param Grid  ───────────────
####################

# Prepare blank placeholder
model_performance = {}
model_feature_importance = {}
model_performance_index = ['Best Hyperparameter', 'Model Accuracy', 
                            'Model Precision', 'Model Recall', 
                            'Model MCC', 'Model PR-AUC', 
                            'Model ROC-AUC', 'Model F1_test', 
                            'Model F1_train', 'Model F1_drop', 
                            'Model OOB_Score', 
                            ]

# import dataset
X = pd.read_csv('XYsnv.csv') #XYnorm XYsnv XYsnvnorm
Y = pd.read_csv('label_fg_present.csv')

# Remove fg that has <6 appearance
less_appear_label = ['alkyl_carbamate','barbitur','benzodiazepine','diazo','dihydropyridine',
                     'guanido','lactam','quatN','tetrazole']
for i in less_appear_label:
    del Y[i]
# Remove fg that has <1% appeareance
less_appear_label = ['HOCCN', 'azide', 'N_O', 'oxazole', 'priamide', 
                     'amidine', 'azo', 'epoxide', 'furan', 'hdrzine',
                     'hdrzone', 'imidazole', 'imide', 'isocyan', 'lactone',
                     'morpholine', 'Ndealkylation1', 'Ndealkylation2', 'nitroso', 'oxime', 
                     'phos_acid', 'phos_ester', 'piperdine', 'piperzine', 'prisulfonamd',
                     'sulfonamd', 'sulfone', 'thiazole', 'thiocyan', 'urea']
for i in less_appear_label:
    del Y[i]

# Collect hyperparameters for tuning:
param_grid = {
    "max_depth":        [3, 5, 10, 15, 20, None],  # add shallower (3) and None
    "max_features":     ["log2", "sqrt", 0.3, 0.5, 0.8],  # add ratio-based
    "n_estimators":     [100, 300, 500],
    "min_samples_leaf": [1, 2, 4, 8],              # important for small datasets
    "min_samples_split":[2, 5, 10],
}

# Pick random params 
sampled_params = list(ParameterSampler(param_grid, n_iter=100, random_state=34))


####################
#   Main Program   # ─── Loop over each functional group  ───────────────
####################
start_runtime = time.time()

# Start evaluating each FG
for fg_name in Y.columns.to_list():
    print(f'>>> Evaluating {fg_name}...')
    dy = Y[fg_name]     #take only for current FG

    # Split dataset
    # Do split inside loop to avoid stratify error of many classes if outside
    X_train, X_test, dy_train, dy_test = train_test_split(X, dy, test_size=0.3, random_state=34, stratify=dy)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    ####################
    #   First Phase    # ─── Random Search and Broad Exploration  ───────────────
    ####################

    # Start first phase
    start_phase1 = time.time()
        
    results = []
    metrics = []
    model_performance_temp = None
    feature_importance_temp = None

    # Run random params
    for params in sampled_params:
        model = RandomForestClassifier(**params, n_bins=32, random_state=34)
        scores = cross_val_score(model, X_train, dy_train, cv=cv, scoring="f1_weighted")
        results.append({**params, "mean_f1": scores.mean(), "std_f1": scores.std()})

    results_df = pd.DataFrame(results).sort_values("mean_f1", ascending=False)
    phase1_time = (time.time()-start_phase1)/60
    #print(results_df.head(10))
    print(f'>>> Elapsed Time for Phase 1 = {phase1_time:.0f} min(s)')

    #notify(f"✅ Phase 1 completed successfully! \n Elapsed time: {phase1_time:.0f} min(s) \n Now executing Phase 2...")


    ####################
    #   Second Phase   # ─── Run Grid  ───────────────
    ####################

    refined_grid = build_refined_grid(results_df, top_n=10)
    print(refined_grid)

    # ─── Run Grid  ───────────────
    start_phase2 = time.time()
    grid_search = GridSearchCV(estimator=RandomForestClassifier(n_bins=32, random_state=34),
                            param_grid=refined_grid,
                            cv=cv,
                            scoring='f1_weighted')
    grid_search.fit(X_train,dy_train)

    best_params = grid_search.best_params_
    print("Best params: ", best_params)

    # ── Final model: refit with best params + oob_score ─────────────────────────
    final_model = RandomForestClassifier(
        **best_params,
        n_bins=32,
        random_state=42,
        oob_score=True            # enable oob
    )
    final_model.fit(X_train, dy_train)

    # Obtain feature importance
    model_feature_importance[fg_name] = final_model.feature_importances_
    # Capture feature importance for each FG separately
    feature_importance_temp = pd.DataFrame.from_dict({fg_name: final_model.feature_importances_})
    feature_importance_temp.index = X_train.columns
    feature_importance_temp.to_csv(f'feature_importances/FI_{fg_name}.csv', index=True)

    # ── Predictions ──────────────────────────────────────────────────────────────

    dy_pred       = final_model.predict(X_test)
    dy_pred_proba = final_model.predict_proba(X_test).to_numpy()[:, 1]
    dy_pred_train = final_model.predict(X_train)

    # ── Metrics ──────────────────────────────────────────────────────────────────

    f1_test  = f1_score(dy_test, dy_pred, average="weighted")
    f1_train = f1_score(dy_train, dy_pred_train, average="weighted")

    cm = confusion_matrix(dy_test.to_numpy(), dy_pred.to_numpy())
    cm_df = pd.DataFrame(cm.get(), index=[0,1], columns=[0,1])
    cm_df.to_csv(f'confusion_matrix/cm_{fg_name}.csv', index=True)

    # Safe roc_auc and pr_auc
    try:
        roc_auc = roc_auc_score(dy_test, dy_pred_proba)
        pr_auc  = average_precision_score(dy_test, dy_pred_proba)
    except ValueError:
        roc_auc = None
        pr_auc  = None

    metrics = [
        best_params,
        accuracy_score(dy_test, dy_pred),
        precision_score(dy_test, dy_pred, average="weighted"),
        recall_score(dy_test, dy_pred, average="weighted"),
        matthews_corrcoef(dy_test, dy_pred),
        average_precision_score(dy_test, dy_pred_proba),
        roc_auc_score(dy_test, dy_pred_proba),
        f1_test,
        f1_train,
        f1_test - f1_train,
        final_model.oob_score_,
    ]

    model_performance[fg_name] = metrics
    # Capture model performance for each FG separately
    model_performance_temp = pd.DataFrame.from_dict({fg_name: metrics})
    model_performance_temp.index = model_performance_index
    model_performance_temp.to_csv(f'model_performance/MP_{fg_name}.csv', index=True)

    joblib.dump(final_model, f'pickled_model/cuml_{fg_name}.pkl')

    #print(pd.Series(metrics))  # use for when metrics is displayed as dict

    phase2_time = (time.time()-start_phase2)/60
    print(f'>>> Elapsed Time for Phase 2 = {phase2_time:.0f} min(s)')

    #notify(f"✅ Phase 2 completed successfully! \n Elapsed time: {phase2_time:.0f} min(s)")
    notify(f"✅ Training {fg_name} completed! \n Elapsed time: \n Phase 1 > {phase1_time:.0f} min(s) \n Phase 2 > {phase2_time:.0f} min(s)")


model_performance_df = pd.DataFrame.from_dict(model_performance)
model_performance_df.index = model_performance_index
model_performance_df.transpose().to_csv('randfor_performance.csv', index=True, header=True)

model_feature_importance_df = pd.DataFrame.from_dict(model_feature_importance)
model_feature_importance_df.index = X_train.columns
model_feature_importance_df.to_csv('feature_importances.csv', index=True, header=True)

total_runtime = (time.time()-start_runtime)/60
notify(f"👌😎🎶 Program complete! \n Total time: {(total_runtime/60):.1f} hour(s)")
