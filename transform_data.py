# -*- coding: utf-8 -*-
"""

# =============================================================================
# Metabolomics Domain Alignment & Transferable Prediction Pipeline
# =============================================================================
# File: transform_data.py
# Version: 1.0.0
# Last updated: 2026-01-12
# Authors: Sergio Decherchi (sergio.decherchi@iit.it), Aigar Ottas (aigar.ottas@ut.ee)
#
# Overview
# --------
# This software aligns and harmonizes tabular metabolomics data (NMR or MS; in
# principle any feature matrix) across domains and then trains transferable
# binary classifiers (e.g., disease vs. non-disease). It implements multiple
# alignment strategies—Optimal Transport (OT), Bridge mapping (paired/nearest-
# neighbor linear regression and CORAL), and Empirical Bayes (ComBat-style)—
# and compares them against RAW and NORMALIZED baselines. The aligned data are
# fed into a model zoo (LR, RF, SVM, DT, NB, SGD, KNN) with optional feature
# selection, SMOTE, and automated hyperparameter tuning. SHAP-based
# explainability, result tables, and publication-quality plots/reports are
# produced automatically.
#
# Typical application
# -------------------
# • Align a destination dataset to a source/reference dataset when both derive
#   from the *same biosample population* but differ due to platform, batch,
#   instrument, protocol, or storage-deterioration effects.
# • Train classifiers using labels from the source cohort and evaluate on
#   destination data *before and after* alignment to assess transferability.
#
# Key capabilities
# ----------------
# • Data preprocessing: median imputation, MinMax scaling (source & destination).
# • Alignment / harmonization:
#   - Optimal Transport (POT MappingTransport; linear kernel).
#   - Bridge mapping: paired linear regression; NN-pseudo-pairing + linear
#     regression; CORAL (covariance alignment).
#   - Empirical Bayes (ComBat-style) batch adjustment (mean/variance correction).
# • Model training & evaluation:
#   - Seven classifiers with RandomizedSearchCV over compact grids.
#   - Optional SMOTE (auto-safe k) and train-only feature selection
#     (mutual_info_classif with keep_k_features).
#   - Stratified CV; balanced accuracy & MCC; confusion matrices for CV/train
#     and CV/val; held-out test metrics for DM/DNM splits.
# • Explainability & reporting:
#   - SHAP mean importance with top-20 plots (automatic Tree/Linear/Kernel
#     explainer selection).
#   - Method-wise bar plots (DNM/DM balanced accuracy), overall best summaries,
#     and a Markdown + CSV report with embedded plots (optional).
#
# Inputs: CSV tables & label files
# --------------------------------
# • --sm   : Source matched CSV (reference samples with labels).
# • --dm   : Destination matched CSV (paired or comparable reference samples).
# • --snm  : Source unmatched CSV (additional unlabeled/unpaired source samples).
# • --dnm  : Destination unmatched CSV (additional unlabeled/unpaired dest samples).
# • --sml  : Labels (single-column CSV) for source matched samples.
# • --dml  : Labels (single-column CSV) for destination matched samples.
# • --snml : Labels for source unmatched samples (optional).
# • --dnml : Labels for destination unmatched samples (optional).
# • --id_col : Optional column name containing sample IDs; retained in combined outputs.
#
# Matched vs. unmatched datasets
# ------------------------------
# • "Matched": Two datasets refer to the *same biosamples* measured under
#   different conditions (e.g., different machines/protocols or the same samples
#   after storage time). These are ideal for OT and paired/NN bridge mapping,
#   and they provide a reliable reference anchor.
# • "Unmatched": Two independent datasets. OT/bridge can still be attempted,
#   but reliability is reduced because the point clouds may only partially
#   overlap. The program will warn when unmatched-only alignment is requested.
#
# Assumptions & cautions
# ----------------------
# • Alignment methods assume that source and destination reflect the *same
#   underlying biological distribution* perturbed by technical shifts.
# • For predictive modeling, avoid data leakage: transforms and feature
#   selection are fitted strictly on the training fold, then applied to
#   validation/test splits. When using external test sets, ensure preprocessing
#   parameters come from the training data only.
# • ComBat-style EB requires adequate batch sizes and balanced covariates; note
#   that aggressive adjustment can remove true biological signal if batches are
#   confounded with outcome.
#
# Outputs
# -------
# • Baseline snapshots: X_training.csv, X_matched_source.csv, X_matched_destination.csv,
#   X_unmatched_source_reference.csv, X_unmatched_destination_reference.csv, etc.
# • Aligned tables by method:
#   - OT: X_*_after_ot.csv
#   - Bridge: X_*_after_bridge.csv
#   - EB: X_*_after_EB.csv and *_origscale.csv (inverse-scaled)
# • Combined tables per method with domain and group columns:
#   combined_OT.csv, combined_bridge.csv, combined_EB.csv,
#   combined_baseline.csv, combined_normalized.csv
# • ML results: ml_results.csv (schema-safe), plots/, report_summary.md/.csv
#
# Quick start
# -----------
# 1) Minimal all-pipelines run (OT, bridge, EB) with baselines:
#    $ python transform_data.py \
#        --sm path/to/source_matched.csv --sml path/to/source_labels.csv \
#        --dm path/to/dest_matched.csv --dml path/to/dest_labels.csv \
#        --outdir outputs/run_YYYYMMDD --report --report_plots
#
# 2) Unmatched-only alignment (will warn about reliability):
#    $ python transform_data.py \
#        --snm path/to/source_unmatched.csv --dnm path/to/dest_unmatched.csv \
#        --snml path/to/source_unmatched_labels.csv --dnml path/to/dest_unmatched_labels.csv \
#        --outdir outputs/run_YYYYMMDD --run_ot --run_bridge --run_eb
#
# Common flags
# ------------
# • --run_ot / --run_bridge / --run_eb : Select pipelines (default: all).
# • --exclude_models "SVM,NB"          : Omit specific classifiers.
# • --use_smote --smote_k 5            : Enable SMOTE (auto-safe fallback).
# • --keep_k_features 50               : Top-k MI features retained (train-only).
# • --bridge_mode auto|paired|nn|coral : Choose bridge strategy.
# • --nn_match_k 3                     : Top-k neighbors for NN pseudo-pairs.
# • --report --report_plots            : Generate summary and plots.
#
# Reproducibility & logging
# -------------------------
# • Saves the full CLI command to <outdir>/cli_command.txt.
# • Uses fixed random seed (RS=0) for shuffling and tuning splits.
# • All CSV outputs are schema-cleaned (no Unnamed columns).
#
# How to cite (suggested)
# -----------------------
# Please cite the method and underlying alignment strategies (Optimal Transport,
# CORAL, ComBat-style EB) and this pipeline if used in a publication:
#   "From Batch to Biology: Harmonized, Transferable ML in NMR Metabolomics"
#   and the specific alignment method(s) applied in your analysis.
#
# License (MIT)
# -------------
# The MIT License (MIT)
#
# Copyright (c) 2026 Sergio Decherchi and Aigar Ottas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# =============================================================================
# End of header
# =============================================================================

"""
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import argparse
# --- Plotting (headless) ---
import matplotlib
matplotlib.use('Agg')  # ensure non-interactive backend
import matplotlib.pyplot as plt
# Publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
})
from sklearn.utils import shuffle
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import (
    KFold, StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Helper: drop id column if present
def drop_id(df, id_col):
    return df.drop(columns=[id_col]) if (df is not None and id_col and id_col in df.columns) else df


warnings.filterwarnings("ignore")
log_err = '<<ERROR>> '
log_warn = '<<WARNING>> '
log_info = '<<INFO>> '
RS = 0
SUPPORTED_MODELS = ['LR','DT','RF','SVM','NB','SGD','KNN']
MAX_TUNE_SAMPLES_DEFAULT = 6000
N_ITER_DEFAULT = 15
N_ITER_BY_MODEL = {'SVM': 10, 'RF': 20, 'LR': 15, 'DT': 15}

class DomainAwareClassifier:
    """Example skeleton for domain adaptation experiments (not used by default)."""

    def __init__(self, base_estimator, domain_weight=0.3):
        self.base_estimator = base_estimator
        self.domain_weight = domain_weight

    def fit(self, X_source, y_source, X_target, y_target):
        sample_weights = np.hstack([
            np.ones(len(X_source)) * (1 - self.domain_weight),
            np.ones(len(X_target)) * self.domain_weight
        ])
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.hstack([y_source, y_target])
        self.estimator_ = clone(self.base_estimator)
        try:
            self.estimator_.fit(X_combined, y_combined, sample_weight=sample_weights)
        except TypeError:
            self.estimator_.fit(X_combined, y_combined)
        return self
    def predict(self, X): return self.estimator_.predict(X)
    def predict_proba(self, X):
        if hasattr(self.estimator_, 'predict_proba'): return self.estimator_.predict_proba(X)
        raise NotImplementedError("Base estimator does not support predict_proba")


class OTmetab:
    """Core pipeline: normalization, combined tables, ML CV/testing, OT/Bridge/EB."""
    ML_RESULTS_COLUMNS = [
        'phase','method','model','timestamp',
        'n_train','n_test_dm','n_test_dnm',
        'cv_bal_acc_mean','cv_bal_acc_std','cv_mcc_mean','cv_mcc_std',
        'cv_bal_acc_folds','cv_mcc_folds',
        'cv_train_cm_TN','cv_train_cm_FP','cv_train_cm_FN','cv_train_cm_TP',
        'cv_val_cm_TN','cv_val_cm_FP','cv_val_cm_FN','cv_val_cm_TP',
        'test_dm_bal_acc','test_dm_mcc','test_dm_cm_TN','test_dm_cm_FP','test_dm_cm_FN','test_dm_cm_TP',
        'test_dnm_bal_acc','test_dnm_mcc','test_dnm_cm_TN','test_dnm_cm_FP','test_dnm_cm_FN','test_dnm_cm_TP',
        'best_params',
        # NEW:
        'features',       # JSON of kept/discarded
        'shap_values',    # JSON of mean SHAP values
        'shap_plot',      # path to saved plot
        'best_models'     # filled on a summary row only
    ]


    def __init__(
        self,
        Xs_model=None, Xs_ot=None, Xd_ot=None, Xt=None,
        Xs_unmatched=None, Xd_unmatched=None,  # NEW
        ys_model=None, yd_ot=None, ysm=None, ysnm=None, ydnm=None,  # NEW ← added ysm
        unmatched_mode=False, output_dir='.', ml_results_path=None, exclude_models=None,
        adjust_dnm_test=False, dnm_adjustment_factor=1.0, dnm_adjustment_method='advanced_shift',
        use_domain_adaptation=False, dnm_validation_split=0.2, use_dnm_ensemble=False, use_smote=False,
        allow_dnm_tuning=False, smote_k=None, max_tune_samples=MAX_TUNE_SAMPLES_DEFAULT,
        bridge_mode='auto', nn_match_k=1,
        keep_k_features: int = 0,
        id_col=None  # <-- added
    ):

        self.max_iter_OT = 50
        self.unmatched_mode = unmatched_mode
        self.output_dir = output_dir; self._ensure_dir(self.output_dir)
        self.ml_results_path = ml_results_path
        self.exclude_models = set([m.upper() for m in (exclude_models or [])])
        
        
        self.ysm  = ysm          # <-- reliable source-matched labels
        self.ysnm = ysnm
        self.yd_ot = yd_ot
        self.ydnm = ydnm


        # Flags (kept for completeness)
        self.adjust_dnm_test = adjust_dnm_test
        self.dnm_adjustment_factor = dnm_adjustment_factor
        self.dnm_adjustment_method = dnm_adjustment_method
        self.use_domain_adaptation = use_domain_adaptation
        self.dnm_validation_split = dnm_validation_split
        self.use_dnm_ensemble = use_dnm_ensemble

        # Training controls
        self.use_smote = use_smote
        self.allow_dnm_tuning = allow_dnm_tuning
        self.smote_k_override = smote_k
        self.max_tune_samples = int(max_tune_samples) if max_tune_samples and max_tune_samples > 0 else MAX_TUNE_SAMPLES_DEFAULT

        # Bridge
        self.bridge_mode = (bridge_mode or 'auto').lower()
        self.nn_match_k = max(1, int(nn_match_k))

        # Feature selection
        self.keep_k_features = int(keep_k_features) if keep_k_features else 0

        # Handle ID column
        self.id_col = id_col
        self.ids_source = None
        self.ids_dest = None
        self.ids_xt = None
        self.ids_unmatched_source = None
        self.ids_unmatched_dest = None

        def split_ids(df):
            if df is not None and self.id_col and self.id_col in df.columns:
                ids = df[self.id_col].copy()
                df = df.drop(columns=[self.id_col])
                return df, ids
            return df, None


        
        Xs_unmatched, self.ids_unmatched_source = split_ids(Xs_unmatched)
        Xd_unmatched, self.ids_unmatched_dest = split_ids(Xd_unmatched)
        self.Xs_unmatched = Xs_unmatched.values if Xs_unmatched is not None else None
        self.Xd_unmatched = Xd_unmatched.values if Xd_unmatched is not None else None
        self.ysnm = ysnm
        


        Xs_model, self.ids_source = split_ids(Xs_model)
        Xs_ot, ids_sm = split_ids(Xs_ot)
        Xd_ot, ids_dm = split_ids(Xd_ot)
        Xt, ids_xt = split_ids(Xt)
        self.ids_dest = ids_dm
        self.ids_xt = ids_xt

        # Keep original DFs
        self.Xs_model_df, self.Xs_ot_df, self.Xd_ot_df, self.Xt_df = Xs_model, Xs_ot, Xd_ot, Xt

        # Convert to numpy
        to_np = lambda x: x.values if hasattr(x, 'values') else x
        Xs_model_np, Xs_ot_np, Xd_ot_np, Xt_np = to_np(Xs_model), to_np(Xs_ot), to_np(Xd_ot), to_np(Xt)

        # Column names
        if Xs_model is not None and hasattr(Xs_model, 'columns'):
            self.column_names = Xs_model.columns.tolist()
        elif Xs_ot is not None and hasattr(Xs_ot, 'columns'):
            self.column_names = Xs_ot.columns.tolist()
        elif Xd_ot is not None and hasattr(Xd_ot, 'columns'):
            self.column_names = Xd_ot.columns.tolist()
        else:
            self.column_names = []

        # Feature set
        if Xs_model_np is not None:
            featuresSet = np.arange(0, Xs_model_np.shape[1])
        elif Xs_ot_np is not None:
            featuresSet = np.arange(0, Xs_ot_np.shape[1])
        elif Xd_ot_np is not None:
            featuresSet = np.arange(0, Xd_ot_np.shape[1])
        else:
            featuresSet = np.array([])

        # Assign arrays
        self.Xs_model = Xs_model_np[:, featuresSet] if Xs_model_np is not None else None
        self.ys_model = ys_model
        self.Xs_ot = Xs_ot_np[:, featuresSet] if Xs_ot_np is not None else None
        self.Xd_ot = Xd_ot_np[:, featuresSet] if Xd_ot_np is not None else None
        self.yd_ot = yd_ot
        
        if Xt_np is not None:
            self.Xt = Xt_np[:, featuresSet]


        # Labels (binary)
        self.hasLabels = False
        if ys_model is not None:
            le = LabelEncoder()
            try:
                self.ys_model = le.fit_transform(np.asarray(ys_model).ravel())
            except Exception as e:
                print(log_err + f'Failed to encode labels: {e}'); exit(-1)
            classes_ = list(le.classes_)
            if len(classes_) != 2:
                print(log_err + f'Expect exactly 2 classes, got {len(classes_)}: {classes_}'); exit(-1)
            self.labels = [0, 1]
            try:
                pd.DataFrame({'encoded':[0,1], 'label':classes_}) \
                  .to_csv(os.path.join(self.output_dir, 'label_mapping.csv'), index=False)
                print(log_info + f"Saved label mapping to {os.path.join(self.output_dir, 'label_mapping.csv')}")
            except Exception:
                pass
            print(log_info + f"Labels encoded as 0/1 for classes: {classes_}")
            self.hasLabels = True
        else:
            print(log_info + "No labels provided - running in unsupervised mode")
            self.labels = [0, 1]

        self.scaler_s = self.scaler_d = None
        self._imputer_s = self._imputer_d = None


    # ------------------------ Utilities (I/O & Combined) ------------------------



    def _ensure_dir(self, d):
        if not os.path.exists(d): os.makedirs(d, exist_ok=True)
    def _save_as_csv(self, data, filename, index=False, ids=None):
        outpath = os.path.join(self.output_dir, filename)
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            try:
                if data is not None and hasattr(data, 'shape') and len(self.column_names) == data.shape[1]:
                    df = pd.DataFrame(data, columns=self.column_names)
                else:
                    df = pd.DataFrame(data)
            except Exception:
                df = pd.DataFrame(data)
        if df is None: return
       
        
        # Normalize column names to strings, then strip 'Unnamed' columns safely
        df.columns = pd.Index([str(c) for c in df.columns])
        df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]
        df.to_csv(outpath, index=index)

        print(log_info + f"Saved {outpath}")
    def _save_dataframe_csv(self, df, filename):
        if df is None: return
        outpath = os.path.join(self.output_dir, filename)
        
        df.columns = pd.Index([str(c) for c in df.columns])
        df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]
        df.to_csv(outpath, index=False)

        print(log_info + f"Saved {outpath}")
    def _read_csv_no_unnamed(self, path):
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        return df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    
    # --- Helper: resolve labels for a tag -> list of length df_len, optionally encoded ---
    def _resolve_group_labels(self, tag, df_len, labels_dict=None):
        # 1) Prefer labels_dict[tag] if provided
        labs = None
        if labels_dict is not None:
            labs = labels_dict.get(tag)

        # 2) Fallback to instance attributes
        if labs is None:
            fallback = {
                'sm': getattr(self, 'ysm', None),
                'snm': getattr(self, 'ysnm', None),
                'dm': getattr(self, 'yd_ot', None),
                'dnm': getattr(self, 'ydnm', None),
            }
            labs = fallback.get(tag)

        
        # 3) Normalize to list
        labs = list(labs) if labs is not None else []

        # Treat both None and NaN as missing
        def _is_missing(x):
            try:
                import math
                return x is None or (isinstance(x, float) and math.isnan(x))
            except Exception:
                return x is None

        # 4) Pad/trim to df_len
        if len(labs) > df_len:
            labs = labs[:df_len]
        elif len(labs) < df_len:
            labs = labs + [None] * (df_len - len(labs))

        # 5) Try to encode string labels to 0/1 if label_mapping.csv exists
        mapping_path = os.path.join(self.output_dir, 'label_mapping.csv')
        if os.path.exists(mapping_path):
            try:
                map_df = pd.read_csv(mapping_path)
                if {'encoded', 'label'}.issubset(set(map_df.columns)):
                    mapping = dict(zip(map_df['label'].astype(str), map_df['encoded']))
                    encoded = []
                    for v in labs:
                        if _is_missing(v):
                            encoded.append(None)
                        else:
                            key = str(v)
                            encoded.append(mapping.get(key, v))  # keep original if not in mapping
                    labs = encoded
            except Exception:
                pass  # silent fallback

        # 6) If everything is missing, provide a non-empty placeholder
        if all(_is_missing(v) for v in labs):
            labs = ['unknown'] * df_len



        return labs

    

    
    def _build_and_save_combined(self, method_tag, tables_dict, labels_dict=None):
        dfs, sources, groups, ids_all = [], [], [], []

        for tag in ['sm', 'snm', 'dm', 'dnm']:
            df = tables_dict.get(tag)
            if df is None:
                continue

            # Load if path given
            if isinstance(df, str):
                dfr = self._read_csv_no_unnamed(df)
                if dfr is None:
                    continue
                df = dfr

            # Ensure DataFrame
            if not isinstance(df, __import__('pandas').DataFrame):
                import pandas as pd
                try:
                    df = pd.DataFrame(df, columns=self.column_names)
                except Exception:
                    df = pd.DataFrame(df)

            # Clean columns
            import pandas as pd
            df.columns = pd.Index([str(c) for c in df.columns])
            df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]

            # Collect IDs — prefer IDs present in df when available; fallback to instance-held IDs
            if self.id_col and self.id_col in df.columns:
                ids_series = df[self.id_col].copy()
                df = df.drop(columns=[self.id_col])
            else:
                fallback_map = {
                    'sm': getattr(self, 'ids_source', None),
                    'snm': getattr(self, 'ids_unmatched_source', None),
                    'dm': getattr(self, 'ids_dest', None),
                    'dnm': getattr(self, 'ids_unmatched_dest', None),
                }
                fb = fallback_map.get(tag)
                if fb is not None:
                    ids_series = pd.Series(list(fb)[:df.shape[0]])
                else:
                    ids_series = pd.Series([None] * df.shape[0])

            dfs.append(df.reset_index(drop=True))
            sources.extend([tag] * df.shape[0])

            # Align IDs list
            if ids_series is not None:
                ids_all.extend(list(ids_series))
            else:
                ids_all.extend([None] * df.shape[0])

            # Labels (robust): prefer labels_dict[tag], else instance attrs; encode & pad/trim
            labs = self._resolve_group_labels(tag, df.shape[0], labels_dict)
            groups.extend(labs)

        if len(dfs) == 0:
            print(log_warn + f"No tables provided to combine for {method_tag}. Skipping combined CSV.")
            return

        import pandas as pd
        combined = pd.concat(dfs, axis=0).reset_index(drop=True)

        # Insert ID column as first column; align length
        if self.id_col:
            # Remove any existing column with same name (case-insensitive)
            combined = combined.loc[:, [c for c in combined.columns if c.lower() != self.id_col.lower()]]
            if len(ids_all) > len(combined):
                ids_all = ids_all[:len(combined)]
            elif len(ids_all) < len(combined):
                ids_all.extend([None] * (len(combined) - len(ids_all)))
            combined.insert(0, self.id_col, ids_all)

        # Domain-aware source and consistent 'group' column
        domain_map = {'sm': 'source', 'snm': 'source', 'dm': 'destination', 'dnm': 'destination'}

        # Keep exact tag values in 'source'
        combined['source'] = sources
        # Also expose coarse domain
        combined['domain'] = [domain_map.get(s, s) for s in sources]
        # Build 'group' from resolved labels (already padded/encoded)
        combined['group'] = groups

        outpath = os.path.join(self.output_dir, f"combined_{method_tag}.csv")
        combined.to_csv(outpath, index=False)
        print(log_info + f"Saved combined table: {outpath}")

        return combined

    
    def _save_baseline_snapshots(self):
        if self.Xs_model is not None:
            self._save_as_csv(self.Xs_model, 'X_training.csv')
        if self.hasLabels and self.ys_model is not None:
            self._save_dataframe_csv(pd.DataFrame(self.ys_model, columns=['label']), 'y_training.csv')
        if self.Xs_ot is not None:
            self._save_as_csv(self.Xs_ot, 'X_matched_source.csv')
        if hasattr(self, 'Xs_unmatched') and self.Xs_unmatched is not None:
            self._save_as_csv(self.Xs_unmatched, 'X_unmatched_source_reference.csv')
        if self.Xd_ot is not None:
            self._save_as_csv(self.Xd_ot, 'X_matched_destination.csv')
        if self.yd_ot is not None:
            self._save_dataframe_csv(pd.DataFrame(self.yd_ot, columns=['label']), 'y_matched.csv')
        if getattr(self, 'Xt', None) is not None:
            self._save_as_csv(self.Xt, 'X_test.csv')

    # ------------------------------- Normalization -------------------------------
    def normalize(self):
        if self.hasLabels and (self.Xs_model is not None):
            self.Xs_model, self.ys_model = shuffle(self.Xs_model, self.ys_model, random_state=RS)
        elif self.Xs_model is not None:
            self.Xs_model = shuffle(self.Xs_model, random_state=RS)
        if self.Xs_ot is not None:
            if hasattr(self, 'ysm') and self.ysm is not None:
                self.Xs_ot, self.ysm = shuffle(self.Xs_ot, self.ysm, random_state=RS)
            else:
                self.Xs_ot = shuffle(self.Xs_ot, random_state=RS)
        if self.Xd_ot is not None:
            if self.yd_ot is not None:
                self.Xd_ot, self.yd_ot = shuffle(self.Xd_ot, self.yd_ot, random_state=RS)
            else:
                self.Xd_ot = shuffle(self.Xd_ot, random_state=RS)
        self.scaler_s = MinMaxScaler(); self._imputer_s = SimpleImputer(strategy='median')
        self.scaler_d = MinMaxScaler(); self._imputer_d = SimpleImputer(strategy='median')
        if self.Xs_model is not None:
            self.Xs_model = self._imputer_s.fit_transform(self.Xs_model)
            self.scaler_s.fit(self.Xs_model)
            self.Xs_model = self.scaler_s.transform(self.Xs_model)
        if self.Xs_ot is not None:
            self.Xs_ot = self._imputer_s.transform(self.Xs_ot)
            try: self.Xs_ot = self.scaler_s.transform(self.Xs_ot)
            except Exception:
                self.scaler_s.fit(self.Xs_ot)
                self.Xs_ot = self.scaler_s.transform(self.Xs_ot)
        
        if hasattr(self, 'Xs_unmatched') and self.Xs_unmatched is not None:
            if hasattr(self, 'ysnm') and self.ysnm is not None:
                self.Xs_unmatched, self.ysnm = shuffle(self.Xs_unmatched, self.ysnm, random_state=RS)
            try:
                self.Xs_unmatched = self._imputer_s.transform(self.Xs_unmatched)
                self.Xs_unmatched = self.scaler_s.transform(self.Xs_unmatched)
            except Exception:
                # Fit on unmatched if source imputer/scaler were not fitted yet
                self.Xs_unmatched = self._imputer_s.fit_transform(self.Xs_unmatched)
                self.scaler_s.fit(self.Xs_unmatched)

        
        # ---- NEW: destination fitting/transform fallback ----
        Xd_fit_src = self.Xd_ot if self.Xd_ot is not None else self.Xd_unmatched
        if Xd_fit_src is not None:
            # Fit imputer/scaler on available destination data
            Xd_fit_src = self._imputer_d.fit_transform(Xd_fit_src)
            self.scaler_d.fit(Xd_fit_src)
            # Assign back to the correct attribute
            if self.Xd_ot is not None:
                self.Xd_ot = self.scaler_d.transform(Xd_fit_src)
            else:
                self.Xd_unmatched = self.scaler_d.transform(Xd_fit_src)
        # Transform the other destination set (if present and not the one used to fit)
        if self.Xd_ot is not None and self.Xd_unmatched is not None:
            self.Xd_unmatched = self._imputer_d.transform(self.Xd_unmatched)
            self.Xd_unmatched = self.scaler_d.transform(self.Xd_unmatched)

        self._save_baseline_snapshots()

    # -------------------- Hyperparameter Tuning & Scoring --------------------
    def _create_dnm_scorer(self, X_val_dnm, y_val_dnm):
        if self.allow_dnm_tuning and X_val_dnm is not None and y_val_dnm is not None and len(X_val_dnm) > 0:
            def dnm_balanced_accuracy_scorer(estimator, X, y):
                try:
                    preds = estimator.predict(X_val_dnm)
                    return float(balanced_accuracy_score(y_val_dnm, preds))
                except Exception as e:
                    print(log_warn + f"DNM scoring failed, fallback to in-fold BA: {e}")
                try:
                    preds_fallback = estimator.predict(X)
                    return float(balanced_accuracy_score(y, preds_fallback))
                except Exception:
                    return 0.0
            return dnm_balanced_accuracy_scorer
        return 'balanced_accuracy'

    def _subsample_for_tuning(self, X, y):
        if X is None or y is None: return X, y
        n = X.shape[0]; m = self.max_tune_samples if self.max_tune_samples and self.max_tune_samples > 0 else MAX_TUNE_SAMPLES_DEFAULT
        if n <= m: return X, y
        sss = StratifiedShuffleSplit(n_splits=1, train_size=m, random_state=RS)
        idx, _ = next(sss.split(X, y))
        return X[idx], y[idx]

    def _safe_smote_k(self, y):
        if y is None or len(y) == 0: return None
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2: return None
        minority = int(np.min(counts))
        if minority < 2:
            print(log_warn + "Minority class has <2 samples; SMOTE disabled for this run.")
            return None
        if self.smote_k_override is not None:
            if self.smote_k_override < 1:
                print(log_warn + f"Provided --smote_k={self.smote_k_override} < 1; using 1.")
                return 1
            if self.smote_k_override >= minority:
                adjusted = max(1, minority - 1)
                print(log_warn + f"--smote_k={self.smote_k_override} too large for minority={minority}; using {adjusted} instead.")
                return adjusted
            return int(self.smote_k_override)
        return max(1, min(5, minority - 1))

    def hyperparameter_tuning_advanced(self, X_train, y_train, model_type, X_val_dnm=None, y_val_dnm=None, use_smote=False):
        """
        RandomizedSearchCV over a compact grid per model with:
        • Optional SMOTE in an imblearn pipeline (k auto/overridden & fold-safe),
        • DNM-aware scorer when enabled,
        • Subsampled tuning set (max_tune_samples),
        • SMOTE + CV guarded on the tuning subset to avoid fold failures.
        """
        param_distributions = {
            'LR': {'C':[0.001,0.01,0.1,1,10,100], 'penalty':['l1','l2'], 'solver':['liblinear'], 'class_weight':['balanced',None]},
            'RF': {'n_estimators':[100,200,300,400], 'max_depth':[10,20,30,None], 'min_samples_split':[2,5,10],
                   'min_samples_leaf':[1,2,4], 'max_features':['sqrt','log2',None], 'bootstrap':[True,False], 'class_weight':['balanced',None]},
            'SVM': {'C':[0.1,1,10], 'kernel':['linear','rbf'], 'gamma':['scale'], 'class_weight':['balanced',None]},
            'DT': {'criterion':['gini','entropy'], 'max_depth':[5,10,20,None], 'min_samples_split':[2,5,10],
                   'min_samples_leaf':[1,2,4], 'class_weight':['balanced',None]}
        }
        # Fallbacks for simple models
        if model_type not in param_distributions:
            if model_type == 'NB': return GaussianNB()
            if model_type == 'SGD': return SGDClassifier(max_iter=1000, random_state=RS)
            if model_type == 'KNN': return KNeighborsClassifier(n_neighbors=7)
            return None
        # Base estimators
        if model_type == 'LR':
            base_estimator = LogisticRegression(random_state=RS, max_iter=2000)
        elif model_type == 'RF':
            base_estimator = RandomForestClassifier(random_state=RS, n_jobs=-1)
        elif model_type == 'SVM':
            base_estimator = SVC(random_state=RS, probability=False)
        elif model_type == 'DT':
            base_estimator = DecisionTreeClassifier(random_state=RS)
        else:
            return None
        # --- NEW: subsample for tuning first, then compute SMOTE/cv on that subset ---
        X_tune, y_tune = self._subsample_for_tuning(X_train, y_train)
        # Determine a safe SMOTE k for the tuning subset
        smote_k_tune = self._safe_smote_k(y_tune) if use_smote else None
        use_smote_tune = use_smote and (smote_k_tune is not None)
        # Determine tuning CV folds safely
        n_splits_tune = 3
        classes_tune, counts_tune = np.unique(y_tune, return_counts=True) if y_tune is not None else ([], [])
        min_count_tune = int(counts_tune.min()) if len(counts_tune) > 0 else 0
        n_splits_tune = min(n_splits_tune, max(2, min_count_tune))  # cannot exceed minority count
        if use_smote_tune and min_count_tune >= 2:
            # Ensure each training split has at least smote_k_tune + 1 minority samples
            while n_splits_tune > 2:
                train_min = int(np.floor(min_count_tune * (n_splits_tune - 1) / n_splits_tune))
                if train_min >= smote_k_tune + 1:
                    break
                n_splits_tune -= 1
            # If still not enough even with n_splits_tune=2, disable SMOTE for tuning
            train_min_final = int(np.floor(min_count_tune * (n_splits_tune - 1) / n_splits_tune))
            if train_min_final < smote_k_tune + 1:
                print(log_warn + "SMOTE disabled for tuning (insufficient minority per training fold). "
                                 "Will still use SMOTE/fallbacks for main CV/final fit.")
                use_smote_tune = False
        # Build pipeline and search grid for tuning
        if use_smote_tune:
            print(log_info + f"SMOTE (tuning) enabled with k_neighbors={smote_k_tune}, cv={n_splits_tune}")
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=RS, k_neighbors=smote_k_tune)),
                ('classifier', base_estimator)
            ])
            search_params = {f'classifier__{k}': v for k, v in param_distributions[model_type].items()}
        else:
            if use_smote and not use_smote_tune:
                print(log_warn + f"SMOTE requested but disabled for tuning (minority too small per fold). "
                                 f"Tuning cv={n_splits_tune}.")
            pipeline = base_estimator
            search_params = param_distributions[model_type]
        scoring = self._create_dnm_scorer(X_val_dnm, y_val_dnm)
        n_iter = N_ITER_BY_MODEL.get(model_type, N_ITER_DEFAULT)
        print(log_info + f"Tuning {model_type} on {X_tune.shape[0]} samples (cv={n_splits_tune}, n_iter={n_iter})")
        t0 = time.time()
        try:
            search = RandomizedSearchCV(
                pipeline,
                search_params,
                n_iter=n_iter,
                cv=n_splits_tune,
                scoring=scoring,
                n_jobs=-1,
                random_state=RS,
                verbose=0,
                error_score=np.nan  # if a trial fails, score=nan; if all fail, we'll catch below
            )
            search.fit(X_tune, y_tune)
            elapsed = time.time() - t0
            print(log_info + f"Best {model_type} parameters: {search.best_params_} (tuning {elapsed:.1f}s)")
            best_estimator = search.best_estimator_
            try:
                best_estimator.best_params_ = search.best_params_
            except Exception:
                pass
            return best_estimator
        except Exception as e:
            # If *all* fits fail, sklearn raises; we fall back gracefully
            print(log_warn + f"Advanced hyperparameter tuning failed for {model_type}: {e}")
            return base_estimator

    # ------------------------------- Results Writer -------------------------------
    def _append_ml_results(self, results_list):
        if self.ml_results_path is None:
            for r in results_list: print(r)
            return
        df_new = pd.DataFrame(results_list)
        
        int_cols = [
            'n_train','n_test_dm','n_test_dnm',
            'test_dm_cm_TN','test_dm_cm_FP','test_dm_cm_FN','test_dm_cm_TP',
            'test_dnm_cm_TN','test_dnm_cm_FP','test_dnm_cm_FN','test_dnm_cm_TP'
        ]
        for c in int_cols:
            if c in df_new.columns:
                df_new[c] = pd.to_numeric(df_new[c], errors='coerce').round().astype('Int64')

        if not df_new.empty: df_new = df_new.loc[:, ~df_new.columns.str.contains('^Unnamed')]
        # ensure all columns exist
        for col in self.ML_RESULTS_COLUMNS:
            if col not in df_new.columns:
                df_new[col] = np.nan
        df_new = df_new.reindex(columns=self.ML_RESULTS_COLUMNS)
        if os.path.exists(self.ml_results_path):
            try:
                df_old = pd.read_csv(self.ml_results_path)
                df_old = df_old.loc[:, ~df_old.columns.str.contains('^Unnamed')]
            except Exception:
                df_old = pd.DataFrame(columns=self.ML_RESULTS_COLUMNS)
            df_old = df_old.reindex(columns=self.ML_RESULTS_COLUMNS)
            df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)
        else:
            df_all = df_new.copy()
        os.makedirs(os.path.dirname(os.path.abspath(self.ml_results_path)), exist_ok=True)
        df_all.to_csv(self.ml_results_path, mode='w', header=True, index=False)
        print(log_info + f"Wrote {len(results_list)} ML results to {self.ml_results_path} (schema-safe rewrite)")

    # ------------------------------- ML Runner -------------------------------
    def _select_features(self, X, y):
        """Return (X_selected, kept_idx, kept_names, discarded_names)."""
        if X is None or self.keep_k_features is None or self.keep_k_features <= 0:
            cols = self.column_names if self.column_names else [f'f{i}' for i in range(X.shape[1])]
            return X, np.arange(X.shape[1]), cols, []
        k = min(self.keep_k_features, X.shape[1])
        # mutual information can handle non-linear; robust to scaling
        try:
            mi = mutual_info_classif(X, y, random_state=RS, discrete_features=False)
        except Exception as e:
            print(log_warn + f"mutual_info_classif failed: {e}. Falling back to keeping first k features.")
            mi = np.arange(X.shape[1], dtype=float)
        idx_sorted = np.argsort(mi)[::-1][:k]
        kept_idx = np.sort(idx_sorted)
        cols = self.column_names if self.column_names else [f'f{i}' for i in range(X.shape[1])]
        kept_names = [cols[i] for i in kept_idx]
        discarded_names = [cols[i] for i in range(X.shape[1]) if i not in set(kept_idx)]
        return X[:, kept_idx], kept_idx, kept_names, discarded_names

    def _compute_and_plot_shap(self, estimator, X_train_sel, feature_names, method_tag, model_name, phase='pre'):
        shap_info = {}
        plot_path = ''
        try:
            import shap  # lazy import
            # sample to speed up SHAP
            n = X_train_sel.shape[0]
            idx = np.random.choice(n, size=min(200, n), replace=False)
            X_sample = X_train_sel[idx]

            base_est = estimator
            if hasattr(estimator, 'named_steps') and 'classifier' in estimator.named_steps:
                base_est = estimator.named_steps['classifier']

            # ---- choose explainer
            if isinstance(base_est, (RandomForestClassifier, DecisionTreeClassifier)):
                explainer = shap.TreeExplainer(base_est)
                sv = explainer.shap_values(X_sample)
            else:
                # Try LinearExplainer first (fast), fallback to KernelExplainer
                try:
                    explainer = shap.LinearExplainer(base_est, X_sample)
                    sv = explainer.shap_values(X_sample)
                except Exception:
                    def fX(X):
                        if hasattr(base_est, 'predict_proba'):
                            return base_est.predict_proba(X)[:, 1]
                        return base_est.decision_function(X)
                    background = shap.sample(X_sample, min(100, X_sample.shape[0]))
                    explainer = shap.KernelExplainer(fX, background)
                    sv = explainer.shap_values(
                        X_sample,
                        nsamples=min(100, X_sample.shape[0] * 10)
                    )

            # ---- normalize SHAP outputs to a 2D numeric array (n_samples, n_features)
            # Accept list/tuple → choose class 1 if present (binary), else class 0
            if isinstance(sv, (list, tuple)):
                chosen = sv[1] if len(sv) > 1 else sv[0]
            else:
                chosen = sv

            # Some SHAP versions return Explanation objects
            # If so, use .values which holds the (n_samples, n_features) array
            if hasattr(chosen, "values"):
                chosen = chosen.values

            # chosen should now be an array-like of shape (n_samples, n_features)
            chosen = np.asarray(chosen)
            if chosen.ndim == 1:
                # Rare edge case: make it 2D
                chosen = chosen.reshape(1, -1)
            elif chosen.ndim > 2:
                # If a 3D array sneaks in (e.g., multioutput), collapse the first "output" axis
                # by taking the last axis as features and averaging across extra outputs
                chosen = np.mean(chosen, axis=1) if chosen.shape[1] <= chosen.shape[-1] else np.mean(chosen, axis=0)

            # ---- compute mean |SHAP|
            p = chosen.shape[1]
            if len(feature_names) != p:
                # trim or pad feature_names to match p (pad with generic names if needed)
                fn = list(feature_names)
                if len(fn) > p:
                    feature_names = fn[:p]
                else:
                    feature_names = fn + [f'f_extra_{i}' for i in range(p - len(fn))]

            mean_abs = np.mean(np.abs(chosen), axis=0).ravel()
            shap_info = {fn: float(val) for fn, val in zip(feature_names, mean_abs)}

            # ---- Plot top-20
            sorted_items = sorted(shap_info.items(), key=lambda x: x[1], reverse=True)[:20]
            names = [k for k, _ in sorted_items]
            vals = [v for _, v in sorted_items]
            if len(names) > 0:
                
                plot_dir = os.path.join(self.output_dir, 'plots')
                self._ensure_dir(plot_dir)
                plt.figure(figsize=(8, 5))
                plot_fname = f"shap_{phase}_{method_tag}_{model_name}.png".replace(' ', '_')
                plot_path = os.path.join(plot_dir, plot_fname)
                plt.savefig(plot_path, dpi=300)

                plt.close()
                print(log_info + f"Saved SHAP plot: {plot_path}")

        except ImportError:
            print(log_warn + "Package 'shap' not installed; skipping SHAP importance and plots.")
        except Exception as e:
            print(log_warn + f"SHAP computation failed: {e}")

        return shap_info, plot_path

    def run_ml_classifiers_enhanced(
        self, X_train, y_train, method_tag, phase='post',
        X_test_dm=None, y_test_dm=None, X_test_dnm=None, y_test_dnm=None,
        n_splits=5, baseline_cv=None, skip_cv=False
    ):
        results = []
        if X_train is None or y_train is None:
            print(log_warn + f"No training data/labels provided for ML in {method_tag} ({phase}). Skipping ML.")
            return results
        candidates = {'LR':'LogisticRegression','DT':'DecisionTree','RF':'RandomForest','SVM':'SVM','NB':'NaiveBayes','SGD':'SGD','KNN':'KNN'}
        models_to_run = [m for m in SUPPORTED_MODELS if m not in self.exclude_models]
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)
        if X_test_dm is not None: X_test_dm = np.asarray(X_test_dm)
        if y_test_dm is not None: y_test_dm = np.asarray(y_test_dm)
        if X_test_dnm is not None: X_test_dnm = np.asarray(X_test_dnm)
        if y_test_dnm is not None: y_test_dnm = np.asarray(y_test_dnm)
        # Feature selection (fit on training only)
        X_train_sel, kept_idx, kept_names, discarded_names = self._select_features(X_train, y_train)
        X_dm_sel = X_test_dm[:, kept_idx] if X_test_dm is not None else None
        X_dnm_sel = X_test_dnm[:, kept_idx] if X_test_dnm is not None else None
        features_json = json.dumps({'kept': kept_names, 'discarded': discarded_names})
        # DNM validation slice (optional)
        X_val_dnm = y_val_dnm = None
        if self.allow_dnm_tuning and X_dnm_sel is not None and y_test_dnm is not None and X_dnm_sel.shape[0] > 10:
            
            frac = max(0.0, min(1.0, float(self.dnm_validation_split)))
            val_size = min(int(X_dnm_sel.shape[0] * frac), 20)

            if val_size > 2: X_val_dnm, y_val_dnm = X_dnm_sel[:val_size], y_test_dnm[:val_size]
        uniq, counts = np.unique(y_train, return_counts=True)
        min_count = int(counts.min()) if len(counts)>0 else 0
        n_splits_eff = min(n_splits, max(2, min_count))
        if self.use_smote and min_count >= 2:
            safe_k = self._safe_smote_k(y_train) or 1
            while n_splits_eff > 2:
                train_min = int(np.floor(min_count * (n_splits_eff - 1) / n_splits_eff))
                if train_min >= safe_k + 1: break
                n_splits_eff -= 1
        try:
            skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=RS)
        except ValueError:
            skf = KFold(n_splits=2, shuffle=True, random_state=RS)

        for mkey in models_to_run:
            print(log_info + f"==> Model {mkey} ({candidates[mkey]}) starting ...")
            row = {'phase': phase, 'method': method_tag, 'model': candidates[mkey], 'timestamp': time.strftime("%Y%m%d_%H%M%S")}
            try:
                best = self.hyperparameter_tuning_advanced(X_train_sel, y_train, mkey, X_val_dnm, y_val_dnm, use_smote=self.use_smote)
                cv_ba_scores, cv_mcc_scores = [], []
                labels_for_cm = np.array([0, 1]) if len(np.unique(y_train)) == 2 else np.unique(y_train)
                cv_train_cm_sum = np.zeros((2, 2), dtype=float)
                cv_val_cm_sum = np.zeros((2, 2), dtype=float)
                def _fit_with_fallback(estimator, X, y):
                    try:
                        estimator.fit(X, y)
                        return estimator
                    except ValueError as e:
                        if 'n_neighbors' in str(e) and 'n_samples_fit' in str(e):
                            print(log_warn + "SMOTE k_neighbors too large for this fold; refitting without SMOTE.")
                            if hasattr(estimator, 'named_steps') and 'classifier' in estimator.named_steps:
                                clf = clone(estimator.named_steps['classifier'])
                            else:
                                clf = clone(estimator)
                            clf.fit(X, y)
                            return clf
                        raise
                if not skip_cv:
                    for train_idx, val_idx in skf.split(X_train_sel, y_train):
                        est = clone(best)
                        est = _fit_with_fallback(est, X_train_sel[train_idx], y_train[train_idx])
                        preds_val = est.predict(X_train_sel[val_idx])
                        cv_ba_scores.append(balanced_accuracy_score(y_train[val_idx], preds_val))
                        cv_mcc_scores.append(matthews_corrcoef(y_train[val_idx], preds_val))
                        cm_val = confusion_matrix(y_train[val_idx], preds_val, labels=labels_for_cm)
                        cv_val_cm_sum += cm_val
                        preds_tr = est.predict(X_train_sel[train_idx])
                        cm_tr = confusion_matrix(y_train[train_idx], preds_tr, labels=labels_for_cm)
                        cv_train_cm_sum += cm_tr
                    row.update({
                        'cv_bal_acc_mean': float(np.mean(cv_ba_scores)), 'cv_bal_acc_std': float(np.std(cv_ba_scores)),
                        'cv_mcc_mean': float(np.mean(cv_mcc_scores)), 'cv_mcc_std': float(np.std(cv_mcc_scores)),
                        'cv_bal_acc_folds': ';'.join(f'{x:.6f}' for x in cv_ba_scores),
                        'cv_mcc_folds': ';'.join(f'{x:.6f}' for x in cv_mcc_scores),
                    })
                    cv_train_cm_avg = (cv_train_cm_sum / skf.get_n_splits()).astype(float)
                    cv_val_cm_avg = (cv_val_cm_sum / skf.get_n_splits()).astype(float)
                    row.update({
                        'cv_train_cm_TN': float(cv_train_cm_avg[0,0]), 'cv_train_cm_FP': float(cv_train_cm_avg[0,1]),
                        'cv_train_cm_FN': float(cv_train_cm_avg[1,0]), 'cv_train_cm_TP': float(cv_train_cm_avg[1,1]),
                        'cv_val_cm_TN': float(cv_val_cm_avg[0,0]), 'cv_val_cm_FP': float(cv_val_cm_avg[0,1]),
                        'cv_val_cm_FN': float(cv_val_cm_avg[1,0]), 'cv_val_cm_TP': float(cv_val_cm_avg[1,1]),
                    })
                else:
                    if baseline_cv is not None and candidates[mkey] in baseline_cv:
                        base = baseline_cv[candidates[mkey]]
                        for k in ['cv_bal_acc_mean','cv_bal_acc_std','cv_mcc_mean','cv_mcc_std',
                                  'cv_train_cm_TN','cv_train_cm_FP','cv_train_cm_FN','cv_train_cm_TP',
                                  'cv_val_cm_TN','cv_val_cm_FP','cv_val_cm_FN','cv_val_cm_TP',
                                  'cv_bal_acc_folds','cv_mcc_folds']:
                            if k in base: row[k] = base[k]
                    else:
                        row.update({'cv_bal_acc_mean': np.nan, 'cv_bal_acc_std': np.nan,
                                    'cv_mcc_mean': np.nan, 'cv_mcc_std': np.nan,
                                    'cv_train_cm_TN': np.nan,'cv_train_cm_FP': np.nan,'cv_train_cm_FN': np.nan,'cv_train_cm_TP': np.nan,
                                    'cv_val_cm_TN': np.nan, 'cv_val_cm_FP': np.nan, 'cv_val_cm_FN': np.nan, 'cv_val_cm_TP': np.nan,
                                    'cv_bal_acc_folds': '', 'cv_mcc_folds': ''})
                row.update({'n_train': int(X_train_sel.shape[0]),
                            'n_test_dm': int(X_dm_sel.shape[0]) if X_dm_sel is not None else 0,
                            'n_test_dnm': int(X_dnm_sel.shape[0]) if X_dnm_sel is not None else 0})
                est_final = clone(best)
                est_final = _fit_with_fallback(est_final, X_train_sel, y_train)
                
                try:
                    bp = getattr(best, 'best_params_', None)
                    row['best_params'] = json.dumps(bp) if bp is not None else ''
                except Exception:
                    row['best_params'] = ''

                if X_dm_sel is not None and y_test_dm is not None:
                    preds_dm = est_final.predict(X_dm_sel)
                    row.update({'test_dm_bal_acc': float(balanced_accuracy_score(y_test_dm, preds_dm)),
                                'test_dm_mcc': float(matthews_corrcoef(y_test_dm, preds_dm))})
                    cm_dm = confusion_matrix(y_test_dm, preds_dm, labels=labels_for_cm)
                    row.update({'test_dm_cm_TN': float(cm_dm[0,0]), 'test_dm_cm_FP': float(cm_dm[0,1]),
                                'test_dm_cm_FN': float(cm_dm[1,0]), 'test_dm_cm_TP': float(cm_dm[1,1])})
                else:
                    row.update({'test_dm_bal_acc': np.nan, 'test_dm_mcc': np.nan,
                                'test_dm_cm_TN': np.nan, 'test_dm_cm_FP': np.nan,
                                'test_dm_cm_FN': np.nan, 'test_dm_cm_TP': np.nan})
                if X_dnm_sel is not None and y_test_dnm is not None:
                    preds_dnm = est_final.predict(X_dnm_sel)
                    row.update({'test_dnm_bal_acc': float(balanced_accuracy_score(y_test_dnm, preds_dnm)),
                                'test_dnm_mcc': float(matthews_corrcoef(y_test_dnm, preds_dnm))})
                    cm_dnm = confusion_matrix(y_test_dnm, preds_dnm, labels=labels_for_cm)
                    row.update({'test_dnm_cm_TN': float(cm_dnm[0,0]), 'test_dnm_cm_FP': float(cm_dnm[0,1]),
                                'test_dnm_cm_FN': float(cm_dnm[1,0]), 'test_dnm_cm_TP': float(cm_dnm[1,1])})
                else:
                    row.update({'test_dnm_bal_acc': np.nan, 'test_dnm_mcc': np.nan,
                                'test_dnm_cm_TN': np.nan, 'test_dnm_cm_FP': np.nan,
                                'test_dnm_cm_FN': np.nan, 'test_dnm_cm_TP': np.nan})
                # Attach feature-selection info
                row['features'] = features_json
                # SHAP
                shap_vals, shap_plot = self._compute_and_plot_shap(est_final, X_train_sel, kept_names, method_tag, candidates[mkey], phase)
                row['shap_values'] = json.dumps(shap_vals) if shap_vals else ''
                row['shap_plot'] = shap_plot
                results.append(row)
            except Exception as e:
                print(log_warn + f"Model {mkey} failed for {method_tag} ({phase}): {e}")
        if len(results) > 0: self._append_ml_results(results)
        return results

    # ------------------------------- Optimal Transport -------------------------------
    def buildTransportMap(self):
        try:
            import ot as _ot
        except Exception as e:
            print(log_warn + f"POT (ot) not available: {e}. Skipping OT mapping.")
            self.ot_mapping_linear = None; return
        self.ot_mapping_linear = _ot.da.MappingTransport(
            kernel="linear", mu=1e0, eta=1e-3, bias=True, max_iter=self.max_iter_OT, verbose=False
        )
        self.ot_mapping_linear.fit(Xs=self.Xd_ot, Xt=self.Xs_ot)
    def applyTransportMap(self):
        if getattr(self, 'ot_mapping_linear', None) is None: return
        if self.Xs_model is not None:
            self._save_as_csv(self.Xs_model, 'X_training_before_ot.csv')
        dest_ref_fname = 'X_unmatched_destination_after_ot.csv' if self.unmatched_mode else 'X_matched_destination_after_ot.csv'
        if self.Xd_ot is not None:
            self.Xd_ot_after = self.ot_mapping_linear.transform(Xs=self.Xd_ot)
            self._save_as_csv(self.Xd_ot_after, dest_ref_fname)
        if hasattr(self, 'Xd_unmatched') and self.Xd_unmatched is not None:
            self.Xd_unmatched_after = self.ot_mapping_linear.transform(Xs=self.Xd_unmatched)
            self._save_as_csv(self.Xd_unmatched_after, 'X_unmatched_destination_after_ot.csv')
        sm_df = pd.DataFrame(self.Xs_ot, columns=self.column_names) if self.Xs_ot is not None else None
        snm_df = pd.DataFrame(self.Xs_unmatched, columns=self.column_names) if hasattr(self,'Xs_unmatched') and self.Xs_unmatched is not None else None
        dm_df = pd.DataFrame(self.Xd_ot_after, columns=self.column_names) if hasattr(self,'Xd_ot_after') else None
        dnm_df = pd.DataFrame(self.Xd_unmatched_after, columns=self.column_names) if hasattr(self,'Xd_unmatched_after') else None
        
        
        
        labels_dict = {
            'sm': getattr(self, 'ysm', None),    # --sml
            'snm': getattr(self, 'ysnm', None),  # --snml
            'dm': getattr(self, 'yd_ot', None),  # --dml
            'dnm': getattr(self, 'ydnm', None)   # --dnml
        }
        
        tables = {'snm': snm_df, 'dnm': dnm_df} if self.unmatched_mode else {'sm': sm_df, 'snm': snm_df, 'dm': dm_df, 'dnm': dnm_df}
        self._build_and_save_combined('OT', tables, labels_dict)



        if self.hasLabels and (self.ys_model is not None):
            X_train, y_train = self.Xs_model, self.ys_model
            X_test_dm = y_test_dm = X_test_dnm = y_test_dnm = None
            if self.unmatched_mode:
                if hasattr(self, 'Xd_ot_after'): X_test_dnm, y_test_dnm = self.Xd_ot_after, self.yd_ot
            else:
                if hasattr(self, 'Xd_ot_after'): X_test_dm, y_test_dm = self.Xd_ot_after, self.yd_ot
            if hasattr(self, 'Xd_unmatched_after') and hasattr(self,'ydnm'):
                X_test_dnm, y_test_dnm = self.Xd_unmatched_after, self.ydnm
            self.run_ml_classifiers_enhanced(
                X_train, y_train, 'OT', 'post', X_test_dm, y_test_dm, X_test_dnm, y_test_dnm,
                baseline_cv=getattr(self, 'baseline_cv_cache', None), skip_cv=False
            )
    def run(self):
        self.normalize()
        self._save_baseline_snapshots()
        if (self.Xd_ot is None) or (self.Xs_ot is None):
            print(log_warn + "Skipping OT mapping: missing reference data.")
        else:
            self.buildTransportMap()
            self.applyTransportMap()
        print(log_info + f"Finished OT pipeline in {self.output_dir}")

    # -------------------------------- BRIDGE Mapping --------------------------------
    def _bridge_map_paired_linear(self):
        lr = LinearRegression(fit_intercept=True).fit(self.Xd_ot, self.Xs_ot)
        return lr.predict(self.Xd_ot)
    def _bridge_map_nn_linear(self):
        k = self.nn_match_k
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.Xs_ot)
        indices = nn.kneighbors(self.Xd_ot, return_distance=False)
        Xs_targets = np.array([self.Xs_ot[idx_list].mean(axis=0) for idx_list in indices])
        lr = LinearRegression(fit_intercept=True).fit(self.Xd_ot, Xs_targets)
        return lr.predict(self.Xd_ot)
    def _bridge_map_coral(self):
        Xs, Xd = self.Xs_ot, self.Xd_ot
        mu_s, mu_d = Xs.mean(axis=0, keepdims=True), Xd.mean(axis=0, keepdims=True)
        Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
        Cd = np.cov(Xd, rowvar=False) + np.eye(Xd.shape[1]) * 1e-6
        def sym_sqrt(m):
            vals, vecs = np.linalg.eigh(m); vals = np.clip(vals, 1e-8, None)
            return vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        def sym_invsqrt(m):
            vals, vecs = np.linalg.eigh(m); vals = np.clip(vals, 1e-8, None)
            return vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T
        A = sym_invsqrt(Cd) @ sym_sqrt(Cs)
        return (Xd - mu_d) @ A + mu_s
    def _choose_bridge_mode(self):
        mode = self.bridge_mode
        if mode == 'auto':
            if (self.Xs_ot is not None and self.Xd_ot is not None and
                self.Xs_ot.shape[0] == self.Xd_ot.shape[0] and not self.unmatched_mode):
                return 'paired'
            return 'coral'
        return mode
    def run_bridge_mapping(self, outdir: str | None = None):
        if outdir:
            self.output_dir = outdir; self._ensure_dir(self.output_dir)
        print(log_info + f"Running bridge-sample mapping pipeline; outputs to {self.output_dir}")
        self.normalize(); self._save_baseline_snapshots()
        if self.Xs_ot is None or self.Xd_ot is None:
            print(log_warn + "Bridge mapping requires reference data (Xs_ot and Xd_ot). Skipping.")
            return
        mode = self._choose_bridge_mode()
        print(log_info + f"Bridge mode: {mode.upper()}")
        Xd_mapped = None
        try:
            if mode == 'paired':
                if self.Xd_ot.shape[0] != self.Xs_ot.shape[0]:
                    raise ValueError("Paired bridge requires equal number of rows in Xd_ot and Xs_ot.")
                Xd_mapped = self._bridge_map_paired_linear()
            elif mode == 'nn':
                Xd_mapped = self._bridge_map_nn_linear()
            elif mode == 'coral':
                Xd_mapped = self._bridge_map_coral()
            else:
                print(log_warn + f"Unknown bridge_mode='{mode}', using CORAL.")
                Xd_mapped = self._bridge_map_coral()
        except Exception as e:
            print(log_err + f"Bridge mapping failed in mode '{mode}': {e}. Falling back to CORAL.")
            try: Xd_mapped = self._bridge_map_coral()
            except Exception as e2:
                print(log_err + f"Bridge CORAL fallback failed: {e2}")
                return
        dest_ref_fname = 'X_unmatched_destination_after_bridge.csv' if self.unmatched_mode else 'X_matched_destination_after_bridge.csv'
        self._save_as_csv(Xd_mapped, dest_ref_fname)
        Xd_unmatched_mapped = None
        if hasattr(self, 'Xd_unmatched') and self.Xd_unmatched is not None:
            if mode in ['paired','nn']:
                if mode == 'paired':
                    lr = LinearRegression(fit_intercept=True).fit(self.Xd_ot, self.Xs_ot)
                else:
                    k = self.nn_match_k
                    nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.Xs_ot)
                    idx = nn.kneighbors(self.Xd_ot, return_distance=False)
                    Xs_targets = np.array([self.Xs_ot[row].mean(axis=0) for row in idx])
                    lr = LinearRegression(fit_intercept=True).fit(self.Xd_ot, Xs_targets)
                Xd_unmatched_mapped = lr.predict(self.Xd_unmatched)
            else:
                Xs, Xd = self.Xs_ot, self.Xd_ot
                mu_s, mu_d = Xs.mean(axis=0, keepdims=True), Xd.mean(axis=0, keepdims=True)
                Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
                Cd = np.cov(Xd, rowvar=False) + np.eye(Xd.shape[1]) * 1e-6
                def sym_sqrt(m):
                    vals, vecs = np.linalg.eigh(m); vals = np.clip(vals,1e-8,None)
                    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T
                def sym_invsqrt(m):
                    vals, vecs = np.linalg.eigh(m); vals = np.clip(vals,1e-8,None)
                    return vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T
                A = sym_invsqrt(Cd) @ sym_sqrt(Cs)
                Xdu_centered = self.Xd_unmatched - mu_d
                Xd_unmatched_mapped = Xdu_centered @ A + mu_s
        self._save_as_csv(Xd_unmatched_mapped, 'X_unmatched_destination_after_bridge.csv') if Xd_unmatched_mapped is not None else None
        sm_df = pd.DataFrame(self.Xs_ot, columns=self.column_names) if self.Xs_ot is not None else None
        snm_df = pd.DataFrame(self.Xs_unmatched, columns=self.column_names) if hasattr(self,'Xs_unmatched') and self.Xs_unmatched is not None else None
        dm_df = pd.DataFrame(Xd_mapped, columns=self.column_names) if Xd_mapped is not None else None
        dnm_df = pd.DataFrame(Xd_unmatched_mapped, columns=self.column_names) if Xd_unmatched_mapped is not None else None
        
        
        # Build labels_dict from any available tag-level labels (no hasLabels gate)
        
        # labels_dict MUST be built ONLY from CLI-provided label files
        # The four arrays were already passed into this OTmetab instance in __init__
        labels_dict = {
            'sm': getattr(self, 'ysm', None),    # <-- from --sml (CLI), may be None
            'snm': getattr(self, 'ysnm', None),  # <-- from --snml (CLI)
            'dm': getattr(self, 'yd_ot', None),  # <-- from --dml (CLI)
            'dnm': getattr(self, 'ydnm', None)   # <-- from --dnml (CLI)
        }

        
        tables = {'snm': snm_df, 'dnm': dnm_df} if self.unmatched_mode else {'sm': sm_df, 'snm': snm_df, 'dm': dm_df, 'dnm': dnm_df}
        self._build_and_save_combined('bridge', tables, labels_dict)



        if self.hasLabels and (self.ys_model is not None):
            X_train, y_train = self.Xs_model, self.ys_model
            X_test_dm = y_test_dm = X_test_dnm = y_test_dnm = None
            if self.unmatched_mode:
                X_test_dnm, y_test_dnm = Xd_mapped, self.yd_ot
            else:
                X_test_dm, y_test_dm = Xd_mapped, self.yd_ot
            if Xd_unmatched_mapped is not None and hasattr(self,'ydnm'):
                X_test_dnm, y_test_dnm = Xd_unmatched_mapped, self.ydnm
            self.run_ml_classifiers_enhanced(
                X_train, y_train, 'bridge', 'post', X_test_dm, y_test_dm, X_test_dnm, y_test_dnm,
                baseline_cv=getattr(self, 'baseline_cv_cache', None), skip_cv=False
            )
        print(log_info + "Bridge-sample mapping pipeline finished.")

    # ---------------------------- Empirical Bayes (ComBat) ----------------------------
    @staticmethod
    def _combat_adjust(data_df, batch_array, covariates_df=None):
        Y = data_df.values.copy()
        batch = np.array(batch_array); batches = np.unique(batch)
        n_samples, n_features = Y.shape
        X = np.column_stack([np.ones(n_samples), covariates_df.values]) if covariates_df is not None else np.ones((n_samples, 1))
        XtX_inv = np.linalg.pinv(X.T @ X)
        Beta_hat = XtX_inv @ X.T @ Y
        Res = Y - (X @ Beta_hat)
        batch_info = {}
        for b in batches:
            idx = np.where(batch == b)[0]
            if len(idx) <= 1: raise ValueError(f"Batch {b} has <=1 samples.")
            res_b = Res[idx, :]
            var_b = np.var(res_b, axis=0, ddof=1); var_b[var_b == 0] = 1e-6
            batch_info[b] = {"idx": idx, "n": len(idx), "mean": np.mean(res_b, axis=0), "var": var_b}
        pooled_var = sum([(batch_info[b]["n"] - 1) * batch_info[b]["var"] for b in batches]) / (n_samples - len(batches))
        pooled_var[pooled_var == 0] = 1e-6
        gamma_hat = np.vstack([batch_info[b]["mean"] for b in batches])
        delta_hat = np.vstack([batch_info[b]["var"] for b in batches])
        gamma_bar = np.mean(gamma_hat, axis=0)
        t2 = np.var(gamma_hat, axis=0, ddof=1)
        s2 = np.mean(delta_hat, axis=0)
        var_delta_hat = np.var(delta_hat, axis=0, ddof=1); var_delta_hat[var_delta_hat == 0] = 1e-8
        a_prior = (2 * (s2 ** 2)) / var_delta_hat + 2
        b_prior = s2 * (a_prior - 1)
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)
        for j in range(n_features):
            d_hat_j = delta_hat[:, j]; d_hat_j[d_hat_j == 0] = 1e-6
            for b_idx, b in enumerate(batches):
                denom = t2[j] * batch_info[b]["n"] + d_hat_j[b_idx]
                gamma_star[b_idx, j] = (t2[j] * batch_info[b]["n"] * gamma_hat[b_idx, j] + d_hat_j[b_idx] * gamma_bar[j]) / (denom if denom != 0 else 1.0)
                a_post = a_prior[j] + batch_info[b]["n"] / 2.0
                b_post = b_prior[j] + 0.5 * np.sum((Res[batch_info[b]["idx"], j] - gamma_hat[b_idx, j]) ** 2)
                delta_star[b_idx, j] = b_post / (a_post - 1.0) if a_post > 1 else d_hat_j[b_idx]
        adjusted = np.zeros_like(Y)
        for b_idx, b in enumerate(batches):
            idx = batch_info[b]["idx"]
            denom = np.sqrt(delta_star[b_idx, :]); denom[denom == 0] = 1e-6
            scaled = (Res[idx, :] - gamma_star[b_idx, :]) / denom
            adjusted[idx, :] = scaled * np.sqrt(pooled_var) + (X @ Beta_hat)[idx, :] + gamma_bar
        return pd.DataFrame(adjusted, index=data_df.index, columns=data_df.columns)

    def run_empirical_bayes(self, outdir: str | None = None, covariates_A_df=None, covariates_B_df=None):
        if outdir:
            self.output_dir = outdir; self._ensure_dir(self.output_dir)
        print(log_info + f"Running ComBat-style Empirical Bayes pipeline; outputs to {self.output_dir}")
        self.normalize(); self._save_baseline_snapshots()
        if self.Xs_ot is None or self.Xd_ot is None:
            print(log_warn + "Empirical Bayes requires reference data. Skipping."); return
        df_sm = pd.DataFrame(self.Xs_ot, columns=self.column_names)
        df_dm = pd.DataFrame(self.Xd_ot, columns=self.column_names)
        combined_df = pd.concat([df_sm, df_dm], axis=0).reset_index(drop=True)
        batch = np.array([0]*df_sm.shape[0] + [1]*df_dm.shape[0])
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(combined_df.values), index=combined_df.index, columns=combined_df.columns)
        try:
            harmonized_scaled = OTmetab._combat_adjust(scaled_df, batch)
        except Exception as e:
            print(log_warn + f"ComBat adjustment failed: {e}."); return
        harmonized_orig = pd.DataFrame(scaler.inverse_transform(harmonized_scaled.values), columns=combined_df.columns)
        n_sm = df_sm.shape[0]
        harmonized_sm = harmonized_scaled.iloc[:n_sm, :].reset_index(drop=True)
        harmonized_dm = harmonized_scaled.iloc[n_sm:, :].reset_index(drop=True)
        harmonized_sm_orig = harmonized_orig.iloc[:n_sm, :].reset_index(drop=True)
        harmonized_dm_orig = harmonized_orig.iloc[n_sm:, :].reset_index(drop=True)
        
        # Get IDs if available
        
        sm_ids = self.ids_source if self.ids_source is not None else None
        dm_ids = self.ids_dest   if self.ids_dest   is not None else None
        # If you need slicing, align lengths explicitly:
        sm_ids = sm_ids[:n_sm] if sm_ids is not None else None

        # Create DataFrames with IDs
        harmonized_sm_df = pd.DataFrame(harmonized_sm, columns=self.column_names)
        if sm_ids is not None:
            harmonized_sm_df.insert(0, self.id_col, sm_ids)

        harmonized_dm_df = pd.DataFrame(harmonized_dm, columns=self.column_names)
        if dm_ids is not None:
            harmonized_dm_df.insert(0, self.id_col, dm_ids)

        dest_ref_fname = 'X_unmatched_destination_after_EB.csv' if self.unmatched_mode else 'X_matched_destination_after_EB.csv'
        source_ref_fname = 'X_unmatched_source_after_EB.csv' if self.unmatched_mode else 'X_matched_source_after_EB.csv'
        self._save_dataframe_csv(harmonized_sm_df, source_ref_fname)
        self._save_dataframe_csv(harmonized_dm_df, dest_ref_fname)
        self._save_dataframe_csv(harmonized_sm_orig, source_ref_fname.replace('.csv', '_origscale.csv'))
        self._save_dataframe_csv(harmonized_dm_orig, dest_ref_fname.replace('.csv', '_origscale.csv'))
        
        # Handle unmatched source
        snm_df_harm_with_ids = None
        if hasattr(self, 'Xs_unmatched') and self.Xs_unmatched is not None:
            df_snm_scaled = pd.DataFrame(self.Xs_unmatched, columns=self.column_names)
            mean_source_harm = harmonized_sm.mean(axis=0).values
            mean_source_scaled = scaled_df.iloc[:n_sm, :].mean(axis=0).values
            snm_df_harm = (df_snm_scaled - mean_source_scaled) + mean_source_harm
            snm_df_harm_df = pd.DataFrame(snm_df_harm, columns=self.column_names)
            if self.ids_unmatched_source is not None:
                snm_df_harm_df.insert(0, self.id_col, self.ids_unmatched_source)
            snm_df_harm_with_ids = snm_df_harm_df

        # Handle unmatched destination
        df_dnm_adj_with_ids = None
        if hasattr(self, 'Xd_unmatched') and self.Xd_unmatched is not None:
            df_dnm = pd.DataFrame(self.Xd_unmatched, columns=self.column_names)
            mean_dest_harm = harmonized_dm.mean(axis=0).values
            mean_dest_scaled = scaled_df.iloc[n_sm:, :].mean(axis=0).values
            df_dnm_adj = (df_dnm - mean_dest_scaled) + mean_dest_harm
            df_dnm_adj_df = pd.DataFrame(df_dnm_adj, columns=self.column_names)
            if self.ids_unmatched_dest is not None:
                df_dnm_adj_df.insert(0, self.id_col, self.ids_unmatched_dest)
            df_dnm_adj_with_ids = df_dnm_adj_df
            self._save_dataframe_csv(df_dnm_adj_df, 'X_unmatched_destination_after_EB.csv')
        
        

        
        # Build labels_dict EXCLUSIVELY from the four CLI label arrays
        labels_dict = {
            'sm': getattr(self, 'ysm', None),     # --sml
            'snm': getattr(self, 'ysnm', None),   # --snml
            'dm': getattr(self, 'yd_ot', None),   # --dml
            'dnm': getattr(self, 'ydnm', None)    # --dnml
        }

        
        tables = {'snm': snm_df_harm_with_ids, 'dnm': df_dnm_adj_with_ids} if self.unmatched_mode else \
                {'sm': harmonized_sm_df, 'snm': snm_df_harm_with_ids, 'dm': harmonized_dm_df, 'dnm': df_dnm_adj_with_ids}
        self._build_and_save_combined('EB', tables, labels_dict)




        
        if self.hasLabels and self.ys_model is not None:
            X_train, y_train = self.Xs_model, self.ys_model
            X_test_dm = y_test_dm = X_test_dnm = y_test_dnm = None
            if self.unmatched_mode:
                X_test_dnm, y_test_dnm = (harmonized_dm.values, self.yd_ot) if harmonized_dm is not None else (None, None)
            else:
                X_test_dm, y_test_dm = (harmonized_dm.values, self.yd_ot) if harmonized_dm is not None else (None, None)
            if df_dnm_adj is not None and hasattr(self, 'ydnm'):
                X_test_dnm, y_test_dnm = df_dnm_adj.values, self.ydnm
            self.run_ml_classifiers_enhanced(
                X_train, y_train, 'EB', 'post', X_test_dm, y_test_dm, X_test_dnm, y_test_dnm,
                baseline_cv=None, skip_cv=False
            )
        print(log_info + "Empirical Bayes (ComBat-style) harmonization pipeline finished.")

# ----------------------------------- Reporting & Plots -----------------------------------
def _annotate_bars(ax):
    for p in ax.patches:
        try:
            height = p.get_height()
            if np.isnan(height):
                continue
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width()/2., height),
                        ha='center', va='bottom', fontsize=10, rotation=0, xytext=(0,3), textcoords='offset points')
        except Exception:
            pass

def generate_report(base_outdir: str, ml_results_path: str, embed_plots: bool = True):
    """
    Create a compact Markdown + CSV report summarizing:
    • Top model per method by primary metric (DNM BA → DM BA → CV BA),
    • Associated MCC, CV means, and sample sizes,
    • Overall best by DNM BA (and best DM BA if available).
    If embed_plots=True, links to plot images are appended when files exist.
    """
    if not os.path.exists(ml_results_path):
        print(log_warn + f"Cannot write report: {ml_results_path} not found.")
        return
    try:
        df = pd.read_csv(ml_results_path)
    except Exception as e:
        print(log_warn + f"Cannot read ml_results.csv: {e}")
        return
    df.columns = pd.Index([str(c) for c in df.columns])
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]
    if df.empty:
        print(log_warn + "ml_results.csv is empty; skipping report.")
        return
    # Ensure columns exist
    for col in ['test_dnm_bal_acc','test_dm_bal_acc','cv_bal_acc_mean',
                'test_dnm_mcc','test_dm_mcc','cv_mcc_mean',
                'n_train','n_test_dm','n_test_dnm','method','model']:
        if col not in df.columns:
            df[col] = np.nan
    methods_order_pref = ['baseline_raw','baseline_normalized','OT','bridge','EB']
    methods_present = [m for m in methods_order_pref if m in df['method'].unique()] + \
                      [m for m in df['method'].unique() if m not in methods_order_pref]
    summary_rows = []

    def _pick_best_row(dfm, priority_cols=('test_dnm_bal_acc','test_dm_bal_acc','cv_bal_acc_mean')):
        for col in priority_cols:
            if col in dfm.columns:
                s = pd.to_numeric(dfm[col], errors='coerce')
                if s.notna().any():
                    # robust idxmax even with NaNs
                    idx = s.fillna(-np.inf).idxmax()
                    return dfm.loc[idx].to_dict(), col
        # last-resort: if dfm has rows, take the first row
        if not dfm.empty:
            return dfm.iloc[0].to_dict(), None
        return None, None

    for method in methods_present:
        dfm = df[df['method'] == method].copy()
        if dfm.empty:
            continue

        br, chosen_col = _pick_best_row(dfm)
        if br is None:
            # nothing to summarize for this method
            continue

        primary_metric = (
            'DNM_BA' if chosen_col == 'test_dnm_bal_acc' else
            'DM_BA'  if chosen_col == 'test_dm_bal_acc' else
            'CV_BA'  if chosen_col == 'cv_bal_acc_mean' else
            'NA'
        )

        summary_rows.append({
            'method': method,
            'model': br.get('model'),
            'primary_metric': primary_metric,
            'primary_value': (
                br.get('test_dnm_bal_acc') if primary_metric=='DNM_BA' else
                br.get('test_dm_bal_acc')  if primary_metric=='DM_BA' else
                br.get('cv_bal_acc_mean')
            ),
            'test_dnm_bal_acc': br.get('test_dnm_bal_acc'),
            'test_dnm_mcc':     br.get('test_dnm_mcc'),
            'test_dm_bal_acc':  br.get('test_dm_bal_acc'),
            'test_dm_mcc':      br.get('test_dm_mcc'),
            'cv_bal_acc_mean':  br.get('cv_bal_acc_mean'),
            'cv_mcc_mean':      br.get('cv_mcc_mean'),
            'n_train':          br.get('n_train'),
            'n_test_dm':        br.get('n_test_dm'),
            'n_test_dnm':       br.get('n_test_dnm'),
            'best_params':      br.get('best_params')
        })
    if not summary_rows:
        print(log_warn + "No method rows found to summarize; skipping report.")
        return
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(base_outdir, 'report_summary.csv')
    summary_df.to_csv(csv_path, index=False)

    def fmt(x):
        try:
            if pd.isna(x): return "NA"
            val = float(x)
            return f"{val:.3f}" if 0.0 <= val <= 1.0 else f"{val:.0f}"
        except Exception:
            return str(x)

    md_lines = []
    md_lines.append("# ML Summary Report\n")
    md_lines.append(f"- Source folder: `{base_outdir}`\n")
    md_lines.append(f"- Results table: `{os.path.basename(ml_results_path)}`\n")
    if summary_df['test_dnm_bal_acc'].notna().any():
        best_dnm_row = summary_df.loc[summary_df['test_dnm_bal_acc'].astype(float).idxmax()]
        md_lines.append(f"**Best DNM BA overall** → `{best_dnm_row['method']}` / `{best_dnm_row['model']}` "
                        f"(BA={fmt(best_dnm_row['test_dnm_bal_acc'])}, MCC={fmt(best_dnm_row['test_dnm_mcc'])}) ")
    if summary_df['test_dm_bal_acc'].notna().any():
        best_dm_row = summary_df.loc[summary_df['test_dm_bal_acc'].astype(float).idxmax()]
        md_lines.append(f"**Best DM BA overall** → `{best_dm_row['method']}` / `{best_dm_row['model']}` "
                        f"(BA={fmt(best_dm_row['test_dm_bal_acc'])}, MCC={fmt(best_dm_row['test_dm_mcc'])}) ")
    # Embed plots if present
    plots_dir = os.path.join(base_outdir, 'plots')
    overall_plot = os.path.join(plots_dir, 'plots_overall_best.png')
    if embed_plots and os.path.exists(overall_plot):
        md_lines.append(f"\n![Overall Best](plots/{os.path.basename(overall_plot)})\n")
    md_lines.append("\n---\n")
    md_lines.append("## Top model per method\n")
    for _, r in summary_df.iterrows():
        md_lines.append(f"### {r['method']}")
        md_lines.append(f"- Model: `{r['model']}`")
        md_lines.append(f"- Primary: **{r['primary_metric']} = {fmt(r['primary_value'])}**")
        md_lines.append(f"- Test DNM: BA={fmt(r['test_dnm_bal_acc'])}, MCC={fmt(r['test_dnm_mcc'])}")
        md_lines.append(f"- Test DM : BA={fmt(r['test_dm_bal_acc'])}, MCC={fmt(r['test_dm_mcc'])}")
        md_lines.append(f"- CV mean : BA={fmt(r['cv_bal_acc_mean'])}, MCC={fmt(r['cv_mcc_mean'])}")
        md_lines.append(f"- Sizes : n_train={fmt(r['n_train'])}, n_test_dm={fmt(r['n_test_dm'])}, n_test_dnm={fmt(r['n_test_dnm'])}\n")
        per_method_plot = os.path.join(plots_dir, f"plots_per_method_{r['method']}.png")
        if embed_plots and os.path.exists(per_method_plot):
            md_lines.append(f"![{r['method']} Models](plots/{os.path.basename(per_method_plot)})\n")
        shap_plot_glob = [p for p in os.listdir(plots_dir) if p.startswith('shap_') and r['method'] in p]
        if embed_plots and shap_plot_glob:
            # embed the first available SHAP for that method
            md_lines.append(f"![SHAP {r['method']}](plots/{shap_plot_glob[0]})\n")
    md_lines.append("\n---\n*Report generated automatically by `--report`.*\n")
    md_text = "\n".join(md_lines)
    md_path = os.path.join(base_outdir, 'report_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(log_info + f"Report written:\n - {md_path}\n - {csv_path}")


def generate_report_plots(base_outdir: str, ml_results_path: str):
    """
    Generate PNG plots for:
    • Per-method comparison of DNM BA and DM BA across models (side-by-side bars),
    • Overall best DNM BA per method (and best DM BA if available),
    • Best model per method across methods (DM and DNM in separate plots).
    Saves under: outputs/<timestamp>/plots/
    """
    if not os.path.exists(ml_results_path):
        print(log_warn + f"Cannot plot: {ml_results_path} not found.")
        return
    try:
        df = pd.read_csv(ml_results_path)
    except Exception as e:
        print(log_warn + f"Cannot read ml_results.csv for plots: {e}")
        return
    df.columns = pd.Index([str(c) for c in df.columns])
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]
    if df.empty:
        print(log_warn + "ml_results.csv is empty; skipping plots.")
        return
    plots_dir = os.path.join(base_outdir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    # Ensure columns exist
    for col in ['test_dnm_bal_acc','test_dm_bal_acc','cv_bal_acc_mean','method','model']:
        if col not in df.columns:
            df[col] = np.nan
    # Per-method bar charts
    methods = sorted([m for m in df['method'].dropna().unique().tolist() if m not in ['best_models']])
    for method in methods:
        dfm = df[df['method'] == method].copy()
        if dfm.empty:
            continue
        # Aggregate by model
        agg = dfm.groupby('model', as_index=False).agg({
            'test_dnm_bal_acc':'mean',
            'test_dm_bal_acc':'mean'
        })
        if agg.empty:
            continue
        models = agg['model'].tolist()
        dnm_vals = agg['test_dnm_bal_acc'].astype(float).values
        dm_vals = agg['test_dm_bal_acc'].astype(float).values
        # If both are NaN for all models, skip plotting
        if np.all(np.isnan(dnm_vals)) and np.all(np.isnan(dm_vals)):
            continue
        x = np.arange(len(models))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(models)*0.9), 5))
        bars1 = ax.bar(x - width/2, np.nan_to_num(dnm_vals, nan=0.0), width, label='DNM BA')
        bars2 = ax.bar(x + width/2, np.nan_to_num(dm_vals, nan=0.0), width, label='DM BA')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'{method}: Model comparison (DNM/DM BA)')
        ax.set_ylim(0, 1.0)
        ax.legend()
        _annotate_bars(ax)
        fig.tight_layout()
        outpath = os.path.join(plots_dir, f'plots_per_method_{method}.png')
        fig.savefig(outpath, dpi=300)
        plt.close(fig)
        print(log_info + f"Saved plot: {outpath}")
    # Overall best per method
    rows = []
    for method in methods:
        dfm = df[df['method'] == method]
        if dfm.empty:
            continue
        best_dnm = None
        best_dm = None
        if dfm['test_dnm_bal_acc'].notna().any():
            ridx = dfm['test_dnm_bal_acc'].astype(float).idxmax()
            r = dfm.loc[ridx]
            best_dnm = (method, r['model'], float(r['test_dnm_bal_acc']))
        if dfm['test_dm_bal_acc'].notna().any():
            ridx = dfm['test_dm_bal_acc'].astype(float).idxmax()
            r = dfm.loc[ridx]
            best_dm = (method, r['model'], float(r['test_dm_bal_acc']))
        rows.append((method, best_dnm, best_dm))
    if rows:
        methods_plot = [m for (m, _, _) in rows]
        dnm_vals = [bd[2] if bd else np.nan for (_, bd, _) in rows]
        dm_vals  = [bm[2] if bm else np.nan for (_, _, bm) in rows]
        x = np.arange(len(methods_plot))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(methods_plot)*0.9), 5))
        bars1 = ax.bar(x - width/2, np.nan_to_num(dnm_vals, nan=0.0), width, label='Best DNM BA')
        bars2 = ax.bar(x + width/2, np.nan_to_num(dm_vals,  nan=0.0), width, label='Best DM BA')
        ax.set_xticks(x); ax.set_xticklabels(methods_plot)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Overall: best BA by method')
        ax.set_ylim(0, 1.0)
        ax.legend()
        _annotate_bars(ax)
        fig.tight_layout()
        overall_path = os.path.join(plots_dir, 'plots_overall_best.png')
        fig.savefig(overall_path, dpi=300)
        plt.close(fig)
        print(log_info + f"Saved plot: {overall_path}")

    # Best model across methods — two separate plots (DNM and DM)
    best_map = {}
    for method in methods:
        dfm = df[df['method'] == method]
        if dfm.empty:
            continue
        entry = {}
        if dfm['test_dnm_bal_acc'].notna().any():
            ridx = dfm['test_dnm_bal_acc'].astype(float).idxmax(); r = dfm.loc[ridx]
            entry['dnm'] = {'model': r['model'], 'score': float(r['test_dnm_bal_acc'])}
        if dfm['test_dm_bal_acc'].notna().any():
            ridx = dfm['test_dm_bal_acc'].astype(float).idxmax(); r = dfm.loc[ridx]
            entry['dm'] = {'model': r['model'], 'score': float(r['test_dm_bal_acc'])}
        if entry:
            best_map[method] = entry
    # Save best_models mapping back into ml_results.csv as a summary row
    if best_map:
        try:
            df2 = pd.read_csv(ml_results_path)
            summary_row = {col: np.nan for col in OTmetab.ML_RESULTS_COLUMNS}
            summary_row.update({'phase': 'summary', 'method': 'best_models', 'model': '-', 'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                                'best_models': json.dumps(best_map)})
            df2 = pd.concat([df2, pd.DataFrame([summary_row])], ignore_index=True)
            df2.to_csv(ml_results_path, index=False)
            print(log_info + "Appended 'best_models' summary row to ml_results.csv")
        except Exception as e:
            print(log_warn + f"Failed to append best_models to ml_results.csv: {e}")
    # Plot DM-only and DNM-only best maps
    if best_map:
        # DNM
        methods_have_dnm = [m for m in methods if m in best_map and 'dnm' in best_map[m]]
        if methods_have_dnm:
            scores = [best_map[m]['dnm']['score'] for m in methods_have_dnm]
            models = [best_map[m]['dnm']['model'] for m in methods_have_dnm]
            x = np.arange(len(methods_have_dnm)); width = 0.6
            fig, ax = plt.subplots(figsize=(max(8, len(methods_have_dnm)*0.9), 5))
            bars = ax.bar(x, scores, width, color='#4C72B0')
            ax.set_xticks(x); ax.set_xticklabels([f"{m}\n({mod})" for m,mod in zip(methods_have_dnm, models)])
            ax.set_ylim(0,1.0)
            ax.set_ylabel('Balanced Accuracy (DNM)')
            ax.set_title('Best model per method — DNM')
            _annotate_bars(ax)
            fig.tight_layout()
            p = os.path.join(plots_dir, 'best_models_dnm.png')
            fig.savefig(p, dpi=300); plt.close(fig)
            print(log_info + f"Saved plot: {p}")
        # DM
        methods_have_dm = [m for m in methods if m in best_map and 'dm' in best_map[m]]
        if methods_have_dm:
            scores = [best_map[m]['dm']['score'] for m in methods_have_dm]
            models = [best_map[m]['dm']['model'] for m in methods_have_dm]
            x = np.arange(len(methods_have_dm)); width = 0.6
            fig, ax = plt.subplots(figsize=(max(8, len(methods_have_dm)*0.9), 5))
            bars = ax.bar(x, scores, width, color='#55A868')
            ax.set_xticks(x); ax.set_xticklabels([f"{m}\n({mod})" for m,mod in zip(methods_have_dm, models)])
            ax.set_ylim(0,1.0)
            ax.set_ylabel('Balanced Accuracy (DM)')
            ax.set_title('Best model per method — DM')
            _annotate_bars(ax)
            fig.tight_layout()
            p = os.path.join(plots_dir, 'best_models_dm.png')
            fig.savefig(p, dpi=300); plt.close(fig)
            print(log_info + f"Saved plot: {p}")

# -------------------------------------------- Main --------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metabolomics Data Harmonization and Prediction Pipeline (October update)')
    # Data
    
    parser.add_argument('--id_col', default=None, help='Optional column name for sample IDs')
    parser.add_argument('--sm', help='Source matched CSV file')
    parser.add_argument('--snm', help='Source unmatched CSV file')
    parser.add_argument('--dm', help='Destination matched CSV file')
    parser.add_argument('--dnm', help='Destination unmatched CSV file')
    parser.add_argument('--sml', help='Source matched labels CSV (single column)')
    parser.add_argument('--dml', help='Destination matched labels CSV (single column)')
    parser.add_argument('--snml', help='Source unmatched labels CSV (single column)')
    parser.add_argument('--dnml', help='Destination unmatched labels CSV (single column)')
    # Pipelines
    parser.add_argument('--run_ot', action='store_true', help='Run Optimal Transport pipeline')
    parser.add_argument('--run_bridge', action='store_true', help='Run Bridge-sample mapping pipeline')
    parser.add_argument('--run_eb', action='store_true', help='Run ComBat-style Empirical Bayes')
    # General
    parser.add_argument('--outdir', default=None, help='Base output directory (default: ./outputs/<timestamp>/)')

    parser.add_argument('--exclude_models', default='', help='Comma-separated short names to exclude (e.g., "SVM,NB")')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE during model training')
    parser.add_argument('--allow_dnm_tuning', action='store_true', help='Allow DNM-guided HP tuning (may bias DNM test)')
    parser.add_argument('--use_domain_adaptation', action='store_true', help='(Reserved) Use domain adaptation flags')
    parser.add_argument('--use_dnm_ensemble', action='store_true', help='(Reserved) Use DNM ensemble flags')
    parser.add_argument('--adjust_dnm_test', action='store_true', help='(Reserved) Adjust DNM test data')
    parser.add_argument('--dnm_adjustment_factor', type=float, default=1.0, help='(Reserved) DNM adjustment factor')
    parser.add_argument('--dnm_adjustment_method', default='advanced_shift', help='(Reserved) DNM adjustment method')
    parser.add_argument('--dnm_validation_split', type=float, default=0.2, help='DNM validation fraction for HP tuning')
    # Tuning & SMOTE
    parser.add_argument('--smote_k', type=int, default=None, help='Override SMOTE k_neighbors (default: auto/clamped)')
    parser.add_argument('--max_tune_samples', type=int, default=MAX_TUNE_SAMPLES_DEFAULT,
                        help='Max samples for hyperparameter tuning only (default: 6000)')
    # Bridge
    parser.add_argument('--bridge_mode', default='auto', help="Bridge mode: 'auto' (default), 'paired', 'nn', 'coral'")
    parser.add_argument('--nn_match_k', type=int, default=1, help='Top-k neighbors for pseudo-pairs (bridge_mode=nn)')
    # Reporting
    parser.add_argument('--report', action='store_true', help='Write report_summary.md and report_summary.csv at the run root')
    parser.add_argument('--report_plots', action='store_true', help='Generate PNG plots under run root /plots and embed in report')
    # NEW: Feature selection
    parser.add_argument('--keep_k_features', type=int, default=0, help='Number of top features to keep (0 = keep all)')

    args = parser.parse_args()

    # Stop when any provided path does not exist (requirement 1)
    provided_paths = {k:getattr(args,k) for k in ['sm','snm','dm','dnm','sml','dml','snml','dnml']}
    missing = [p for p in provided_paths.values() if p and not os.path.exists(p)]
    if missing:
        print(log_err + "The following file(s) were specified but not found:\n  - " + "\n  - ".join(missing))
        sys.exit(2)

    if not (args.run_ot or args.run_bridge or args.run_eb):
        print(log_info + "No --run_* flags provided: running all pipelines (OT, bridge, EB).")
        args.run_ot, args.run_bridge, args.run_eb = True, True, True

    BASE_OUTPUT_DIR = args.outdir if args.outdir else os.path.join("outputs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(log_info + f"Base outputs will be written to: {BASE_OUTPUT_DIR}")

    # Save full CLI command (requirement 7)
    try:
        with open(os.path.join(BASE_OUTPUT_DIR, 'cli_command.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(sys.argv))
    except Exception as e:
        print(log_warn + f"Failed to save cli_command.txt: {e}")

    ML_RESULTS_PATH = os.path.join(BASE_OUTPUT_DIR, 'ml_results.csv')
    exclude_models_set = {m.strip().upper() for m in args.exclude_models.split(',') if m.strip()}

    def load_csv_no_index(path):
        if not path or not os.path.exists(path): return None
        df = pd.read_csv(path)
        return df.loc[:, ~df.columns.str.contains('^Unnamed')]

    Xsm, Xdm, Xsnm, Xdnm = load_csv_no_index(args.sm), load_csv_no_index(args.dm), load_csv_no_index(args.snm), load_csv_no_index(args.dnm)
    ysm = load_csv_no_index(args.sml).iloc[:, 0].values if args.sml and os.path.exists(args.sml) else None
    ydm = load_csv_no_index(args.dml).iloc[:, 0].values if args.dml and os.path.exists(args.dml) else None
    ysnm = load_csv_no_index(args.snml).iloc[:, 0].values if args.snml and os.path.exists(args.snml) else None
    ydnm = load_csv_no_index(args.dnml).iloc[:, 0].values if args.dnml and os.path.exists(args.dnml) else None

    
    def run_all_pipelines(Xs_model_df, Xs_ot_df, Xd_ot_df, Xt_df, Xsnm_df, Xdnm_df,
                        ys_model, yd_ot, ysnm, ydnm,
                        run_ot, run_bridge, run_eb, unmatched_mode,
                        base_outdir, exclude_models, use_smote):
        
        # Preserve original DataFrames for combined tables
        sm_orig = Xs_ot_df.copy() if Xs_ot_df is not None else None
        snm_orig = Xsnm_df.copy() if Xsnm_df is not None else None
        dm_orig = Xd_ot_df.copy() if Xd_ot_df is not None else None
        dnm_orig = Xdnm_df.copy() if Xdnm_df is not None else None
        
        
        ids_dict_raw = {
    'sm': sm_orig[args.id_col] if sm_orig is not None and args.id_col in sm_orig.columns else None,
    'snm': snm_orig[args.id_col] if snm_orig is not None and args.id_col in snm_orig.columns else None,
    'dm': dm_orig[args.id_col] if dm_orig is not None and args.id_col in dm_orig.columns else None,
    'dnm': dnm_orig[args.id_col] if dnm_orig is not None and args.id_col in dnm_orig.columns else None
}


  
        
        common_ctor = dict(
            Xs_model=Xs_model_df, Xs_ot=Xs_ot_df, Xd_ot=Xd_ot_df, Xt=Xt_df,
            Xs_unmatched=Xsnm_df, Xd_unmatched=Xdnm_df,  # NEW
            ys_model=ys_model, yd_ot=yd_ot, ysm=ysm, ysnm=ysnm, ydnm=ydnm,  # NEW labels for unmatched
            unmatched_mode=unmatched_mode,
            ml_results_path=ML_RESULTS_PATH, exclude_models=exclude_models,
            adjust_dnm_test=args.adjust_dnm_test, dnm_adjustment_factor=args.dnm_adjustment_factor,
            dnm_adjustment_method=args.dnm_adjustment_method, use_domain_adaptation=args.use_domain_adaptation,
            dnm_validation_split=args.dnm_validation_split, use_dnm_ensemble=args.use_dnm_ensemble,
            use_smote=use_smote, allow_dnm_tuning=args.allow_dnm_tuning,
            smote_k=args.smote_k, max_tune_samples=args.max_tune_samples,
            bridge_mode=args.bridge_mode, nn_match_k=args.nn_match_k,
            keep_k_features=args.keep_k_features, id_col=args.id_col
        )


        # --- Baseline (RAW) ---
        baseline_dir = os.path.join(base_outdir, 'baseline')
        baseline_inst = OTmetab(output_dir=baseline_dir, **common_ctor)
        



        if Xs_ot_df is not None and ys_model is not None:
            n_sm = Xs_ot_df.shape[0]
            baseline_inst.ysm = ys_model[:n_sm]
            if Xsnm_df is not None and ys_model is not None and len(ys_model) >= n_sm + Xsnm_df.shape[0]:
                baseline_inst.ysnm = ys_model[n_sm:n_sm + Xsnm_df.shape[0]]

        # Drop ID column before assigning unmatched arrays
        baseline_inst.Xs_unmatched = drop_id(Xsnm_df, args.id_col).values if Xsnm_df is not None else None
        baseline_inst.Xd_unmatched = drop_id(Xdnm_df, args.id_col).values if Xdnm_df is not None else None
        baseline_inst.ydnm = ydnm

        baseline_inst._save_baseline_snapshots()
        
        
        labels_dict_raw = {'sm': ysm, 'snm': ysnm, 'dm': ydm, 'dnm': ydnm}

        tables_baseline = {}
        if unmatched_mode:
            # Only unmatched tables in unmatched-only runs
            if snm_orig is not None: tables_baseline['snm'] = snm_orig
            if dnm_orig is not None: tables_baseline['dnm'] = dnm_orig
        else:
            # Include matched (and unmatched if present)
            if sm_orig is not None:  tables_baseline['sm']  = sm_orig
            if dm_orig is not None:  tables_baseline['dm']  = dm_orig
            if snm_orig is not None: tables_baseline['snm'] = snm_orig
            if dnm_orig is not None: tables_baseline['dnm'] = dnm_orig

        baseline_inst._build_and_save_combined('baseline', tables_baseline, labels_dict_raw)



        if baseline_inst.hasLabels and baseline_inst.ys_model is not None:
            X_train, y_train = baseline_inst.Xs_model, baseline_inst.ys_model
            X_test_dm = y_test_dm = X_test_dnm = y_test_dnm = None
            if unmatched_mode:
                X_test_dnm, y_test_dnm = baseline_inst.Xd_ot, baseline_inst.yd_ot
            else:
                X_test_dm, y_test_dm = baseline_inst.Xd_ot, baseline_inst.yd_ot
            if Xdnm_df is not None:
                X_test_dnm, y_test_dnm = drop_id(Xdnm_df, args.id_col).values, ydnm
            print(log_info + f"Baseline (RAW) ML: Training on {X_train.shape[0]}, "
                            f"DM={0 if X_test_dm is None else X_test_dm.shape[0]}, "
                            f"DNM={0 if X_test_dnm is None else X_test_dnm.shape[0]}")
            baseline_inst.run_ml_classifiers_enhanced(X_train, y_train, 'baseline_raw', 'pre',
                                                    X_test_dm, y_test_dm, X_test_dnm, y_test_dnm)

        # --- Baseline (NORMALIZED) ---
        normalized_dir = os.path.join(base_outdir, 'normalized')
        normalized_inst = OTmetab(output_dir=normalized_dir, **common_ctor)

        if Xs_ot_df is not None and ys_model is not None:
            n_sm = Xs_ot_df.shape[0]
            normalized_inst.ysm = ys_model[:n_sm]
            if Xsnm_df is not None and ys_model is not None and len(ys_model) >= n_sm + Xsnm_df.shape[0]:
                normalized_inst.ysnm = ys_model[n_sm:n_sm + Xsnm_df.shape[0]]

        normalized_inst.Xs_unmatched = drop_id(Xsnm_df, args.id_col).values if Xsnm_df is not None else None
        normalized_inst.Xd_unmatched = drop_id(Xdnm_df, args.id_col).values if Xdnm_df is not None else None
        normalized_inst.ydnm = ydnm

        normalized_inst.normalize()
        normalized_inst._save_baseline_snapshots()

        sm_df_norm = pd.DataFrame(normalized_inst.Xs_ot, columns=normalized_inst.column_names) if normalized_inst.Xs_ot is not None else None
        snm_df_norm = pd.DataFrame(normalized_inst.Xs_unmatched, columns=normalized_inst.column_names) if normalized_inst.Xs_unmatched is not None else None
        dm_df_norm = pd.DataFrame(normalized_inst.Xd_ot, columns=normalized_inst.column_names) if normalized_inst.Xd_ot is not None else None
        dnm_df_norm = pd.DataFrame(normalized_inst.Xd_unmatched, columns=normalized_inst.column_names) if normalized_inst.Xd_unmatched is not None else None

        
        labels_dict_norm = {
            'sm': getattr(normalized_inst, 'ysm', None),
            'snm': getattr(normalized_inst, 'ysnm', None),
            'dm': getattr(normalized_inst, 'yd_ot', None),
            'dnm': getattr(normalized_inst, 'ydnm', None),
        }

        
        # ---- NEW: gate normalized combined tables by unmatched_mode ----
        tables_normalized = {}
        if normalized_inst.unmatched_mode:
            # Only include unmatched tables when running unmatched-only
            if snm_df_norm is not None: tables_normalized['snm'] = snm_df_norm
            if dnm_df_norm is not None: tables_normalized['dnm'] = dnm_df_norm
        else:
            # Include matched (and also unmatched if provided)
            if sm_df_norm is not None:  tables_normalized['sm']  = sm_df_norm
            if dm_df_norm is not None:  tables_normalized['dm']  = dm_df_norm
            if snm_df_norm is not None: tables_normalized['snm'] = snm_df_norm
            if dnm_df_norm is not None: tables_normalized['dnm'] = dnm_df_norm

        normalized_inst._build_and_save_combined('normalized', tables_normalized, labels_dict_norm)
        # ---- END NEW ----


        baseline_rows_norm = []
        if normalized_inst.hasLabels and normalized_inst.ys_model is not None:
            X_train, y_train = normalized_inst.Xs_model, normalized_inst.ys_model
            X_test_dm = y_test_dm = X_test_dnm = y_test_dnm = None
            if unmatched_mode:
                X_test_dnm, y_test_dnm = normalized_inst.Xd_ot, normalized_inst.yd_ot
            else:
                X_test_dm, y_test_dm = normalized_inst.Xd_ot, normalized_inst.yd_ot
            if normalized_inst.Xd_unmatched is not None and hasattr(normalized_inst, 'ydnm'):
                X_test_dnm, y_test_dnm = normalized_inst.Xd_unmatched, normalized_inst.ydnm
            print(log_info + f"Normalized ML: Training on {X_train.shape[0]}, "
                            f"DM={0 if X_test_dm is None else X_test_dm.shape[0]}, "
                            f"DNM={0 if X_test_dnm is None else X_test_dnm.shape[0]}")
            baseline_rows_norm = normalized_inst.run_ml_classifiers_enhanced(X_train, y_train,
                                                                            'baseline_normalized', 'pre',
                                                                            X_test_dm, y_test_dm,
                                                                            X_test_dnm, y_test_dnm)

        baseline_cv_cache = {r['model']: r for r in (baseline_rows_norm or [])}

        # --- Run OT, Bridge, EB pipelines ---
        
        pipelines = {'OT': run_ot, 'bridge': run_bridge, 'EB': run_eb}
        for name, should_run in pipelines.items():
            if not should_run:
                continue

            # Default instance (matched references if available)
            inst_out = os.path.join(base_outdir, f"results_{name}")
            inst = OTmetab(output_dir=inst_out, **common_ctor)
            inst.ysm = ys_model[:Xs_ot_df.shape[0]] if (ys_model is not None and Xs_ot_df is not None) else None
            inst.ysnm = ysnm if ysnm is not None else (
                ys_model[Xs_ot_df.shape[0]: Xs_ot_df.shape[0] + (Xsnm_df.shape[0] if Xsnm_df is not None else 0)]
                if ys_model is not None and Xs_ot_df is not None else None
            )
            inst.Xs_unmatched = drop_id(Xsnm_df, args.id_col).values if Xsnm_df is not None else None
            inst.Xd_unmatched = drop_id(Xdnm_df, args.id_col).values if Xdnm_df is not None else None
            inst.ydnm = ydnm
            inst.baseline_cv_cache = baseline_cv_cache

            # If matched references are missing but unmatched are present, build a
            # temporary "reference-from-unmatched" instance just for the pipeline.
            needs_ref = (name in ['OT', 'bridge', 'EB'])
            missing_ref = (Xs_ot_df is None or Xd_ot_df is None)
            have_unmatched = (Xsnm_df is not None and Xdnm_df is not None)

            if needs_ref and missing_ref and have_unmatched:
                # Construct a new instance that uses snm/dnm as reference pairs
                
                inst_ref = OTmetab(
                    output_dir=inst_out,
                    Xs_model=Xsnm_df,
                    Xs_ot=Xsnm_df,          # use snm as source reference
                    Xd_ot=Xdnm_df,          # use dnm as destination reference
                    Xt=None,
                    Xs_unmatched=Xsnm_df,   # avoid duplicating in combined
                    Xd_unmatched=Xdnm_df,
                    # IMPORTANT: keep the CLI unmatched label arrays attached to the instance
                    ys_model=ysnm,          # training labels (still fine to use)
                    yd_ot=ydnm,             # destination labels for the “reference”
                    ysnm=ysnm,              # <- pass through snm labels so combined tables can use them
                    ydnm=ydnm,              # <- pass through dnm labels so combined tables can use them
                    unmatched_mode=True,
                    ml_results_path=ML_RESULTS_PATH,
                    exclude_models=exclude_models,
                    adjust_dnm_test=args.adjust_dnm_test,
                    dnm_adjustment_factor=args.dnm_adjustment_factor,
                    dnm_adjustment_method=args.dnm_adjustment_method,
                    use_domain_adaptation=args.use_domain_adaptation,
                    dnm_validation_split=args.dnm_validation_split,
                    use_dnm_ensemble=args.use_dnm_ensemble,
                    use_smote=use_smote,
                    allow_dnm_tuning=args.allow_dnm_tuning,
                    smote_k=args.smote_k,
                    max_tune_samples=args.max_tune_samples,
                    bridge_mode=args.bridge_mode,
                    nn_match_k=args.nn_match_k,
                    keep_k_features=args.keep_k_features,
                    id_col=args.id_col
                )


                print(log_info + f"Running {name} pipeline (unmatched-as-reference)")
                if name == 'OT':
                    inst_ref.run()
                elif name == 'bridge':
                    inst_ref.run_bridge_mapping()
                elif name == 'EB':
                    inst_ref.run_empirical_bayes()
            else:
                # Normal path (will no-op if references truly missing and you want strict skipping)
                print(log_info + f"Running {name} pipeline (results_{name}/)")
                if name == 'OT':
                    inst.run()
                elif name == 'bridge':
                    inst.run_bridge_mapping()
                elif name == 'EB':
                    inst.run_empirical_bayes()



    # Scenario detection
    if Xsm is not None and Xdm is not None and Xsnm is None and Xdnm is None:
        print(log_info + "Mode: Matched data only")
        run_all_pipelines(Xsm, Xsm, Xdm, None, None, None, ysm, ydm, None, None,
                          args.run_ot, args.run_bridge, args.run_eb, False, BASE_OUTPUT_DIR, exclude_models_set, args.use_smote)
    
    elif Xsm is None and Xdm is None and Xsnm is not None and Xdnm is not None:
        print(log_info + "Mode: Unmatched data only")
        run_all_pipelines(
            Xsnm,     # Xs_model_df
            None,     # Xs_ot_df  -> None (no matched source)
            None,     # Xd_ot_df  -> None (no matched destination)
            None,     # Xt_df
            Xsnm,     # Xsnm_df   -> keep
            Xdnm,     # Xdnm_df   -> keep
            ysnm,     # ys_model
            None,     # yd_ot     -> None (no matched dest labels)
            ysnm,     # ysnm
            ydnm,     # ydnm
            args.run_ot, args.run_bridge, args.run_eb,
            True, BASE_OUTPUT_DIR, exclude_models_set, args.use_smote
        )


    elif Xsm is not None and Xdm is not None and Xsnm is not None and Xdnm is None:
        print(log_info + "Mode: Matched + Unmatched Source")
        Xs_model = pd.concat([Xsm, Xsnm], axis=0, ignore_index=True)
        ys_model = np.concatenate((ysm, ysnm)) if ysm is not None and ysnm is not None else None
        run_all_pipelines(Xs_model, Xsm, Xdm, None, Xsnm, None, ys_model, ydm, ysnm, None,
                          args.run_ot, args.run_bridge, args.run_eb, False, BASE_OUTPUT_DIR, exclude_models_set, args.use_smote)
    elif Xsm is not None and Xdm is not None and Xsnm is not None and Xdnm is not None:
        print(log_info + "Mode: All data available")
        Xs_model = pd.concat([Xsm, Xsnm], axis=0, ignore_index=True)
        ys_model = np.concatenate((ysm, ysnm)) if ysm is not None and ysnm is not None else None
        run_all_pipelines(Xs_model, Xsm, Xdm, Xdnm, Xsnm, Xdnm, ys_model, ydm, ysnm, ydnm,
                          args.run_ot, args.run_bridge, args.run_eb, False, BASE_OUTPUT_DIR, exclude_models_set, args.use_smote)
    elif Xsm is not None and Xdm is not None and Xsnm is None and Xdnm is not None:
        print(log_info + "Mode: Matched + Unmatched Destination")
        run_all_pipelines(Xsm, Xsm, Xdm, Xdnm, None, Xdnm, ysm, ydm, None, ydnm,
                          args.run_ot, args.run_bridge, args.run_eb, False, BASE_OUTPUT_DIR, exclude_models_set, args.use_smote)
    else:
        print(log_err + "Cannot proceed with the provided combination of data files.")

    print(log_info + f"All requested pipelines finished. ML results (if any) are saved at: {ML_RESULTS_PATH}")

    # NEW: plots + report
    if args.report_plots:
        generate_report_plots(BASE_OUTPUT_DIR, ML_RESULTS_PATH)
    if args.report:
        # Embed plots if they were requested and thus exist
        generate_report(BASE_OUTPUT_DIR, ML_RESULTS_PATH, embed_plots=args.report_plots)

# FIXED EB FUNCTION

import pandas as pd

def run_empirical_bayes_fixed(self, outdir: str | None = None, covariates_A_df=None, covariates_B_df=None):
    if outdir:
        self.output_dir = outdir; self._ensure_dir(self.output_dir)
    print(log_info + f"Running ComBat-style Empirical Bayes pipeline; outputs to {self.output_dir}")
    self.normalize(); self._save_baseline_snapshots()
    if self.Xs_ot is None or self.Xd_ot is None:
        print(log_warn + "Empirical Bayes requires reference data. Skipping."); return

    df_sm = pd.DataFrame(self.Xs_ot, columns=self.column_names)
    df_dm = pd.DataFrame(self.Xd_ot, columns=self.column_names)
    combined_df = pd.concat([df_sm, df_dm], axis=0).reset_index(drop=True)
    batch = np.array([0]*df_sm.shape[0] + [1]*df_dm.shape[0])
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(combined_df.values), index=combined_df.index, columns=combined_df.columns)
    try:
        harmonized_scaled = OTmetab._combat_adjust(scaled_df, batch)
    except Exception as e:
        print(log_warn + f"ComBat adjustment failed: {e}."); return

    harmonized_orig = pd.DataFrame(scaler.inverse_transform(harmonized_scaled.values), columns=combined_df.columns)
    n_sm = df_sm.shape[0]
    harmonized_sm = harmonized_scaled.iloc[:n_sm, :].reset_index(drop=True)
    harmonized_dm = harmonized_scaled.iloc[n_sm:, :].reset_index(drop=True)
    harmonized_sm_orig = harmonized_orig.iloc[:n_sm, :].reset_index(drop=True)
    harmonized_dm_orig = harmonized_orig.iloc[n_sm:, :].reset_index(drop=True)

    # IDs
    
    sm_ids = self.ids_source if self.ids_source is not None else None
    dm_ids = self.ids_dest   if self.ids_dest   is not None else None
    # If you need slicing, align lengths explicitly:
    sm_ids = sm_ids[:n_sm] if sm_ids is not None else None

    harmonized_sm_df = pd.DataFrame(harmonized_sm, columns=self.column_names)
    if sm_ids is not None:
        harmonized_sm_df.insert(0, self.id_col, sm_ids)
    harmonized_dm_df = pd.DataFrame(harmonized_dm, columns=self.column_names)
    if dm_ids is not None:
        harmonized_dm_df.insert(0, self.id_col, dm_ids)

    self._save_dataframe_csv(harmonized_sm_df, 'X_matched_source_after_EB.csv')
    self._save_dataframe_csv(harmonized_dm_df, 'X_matched_destination_after_EB.csv')
    self._save_dataframe_csv(harmonized_sm_orig, 'X_matched_source_after_EB_origscale.csv')
    self._save_dataframe_csv(harmonized_dm_orig, 'X_matched_destination_after_EB_origscale.csv')

    # Prepare unmatched tables (empty if None)
    snm_df_harm_with_ids = pd.DataFrame(columns=self.column_names)
    if hasattr(self, 'Xs_unmatched') and self.Xs_unmatched is not None:
        df_snm_scaled = pd.DataFrame(self.Xs_unmatched, columns=self.column_names)
        mean_source_harm = harmonized_sm.mean(axis=0).values
        mean_source_scaled = scaled_df.iloc[:n_sm, :].mean(axis=0).values
        snm_df_harm = (df_snm_scaled - mean_source_scaled) + mean_source_harm
        snm_df_harm_with_ids = pd.DataFrame(snm_df_harm, columns=self.column_names)
        if self.ids_unmatched_source is not None:
            snm_df_harm_with_ids.insert(0, self.id_col, self.ids_unmatched_source)

    df_dnm_adj_with_ids = pd.DataFrame(columns=self.column_names)
    if hasattr(self, 'Xd_unmatched') and self.Xd_unmatched is not None:
        df_dnm = pd.DataFrame(self.Xd_unmatched, columns=self.column_names)
        mean_dest_harm = harmonized_dm.mean(axis=0).values
        mean_dest_scaled = scaled_df.iloc[n_sm:, :].mean(axis=0).values
        df_dnm_adj = (df_dnm - mean_dest_scaled) + mean_dest_harm
        df_dnm_adj_with_ids = pd.DataFrame(df_dnm_adj, columns=self.column_names)
        if self.ids_unmatched_dest is not None:
            df_dnm_adj_with_ids.insert(0, self.id_col, self.ids_unmatched_dest)
        self._save_dataframe_csv(df_dnm_adj_with_ids, 'X_unmatched_destination_after_EB.csv')

    labels_dict = None
    if self.hasLabels:
        labels_dict = {'sm': getattr(self,'ysm',None), 'snm': getattr(self,'ysnm',None), 'dm': self.yd_ot, 'dnm': getattr(self,'ydnm',None)}

    # Always build combined table with sm and dm, even if snm/dnm are empty
    self._build_and_save_combined('EB', {'sm': harmonized_sm_df, 'snm': snm_df_harm_with_ids, 'dm': harmonized_dm_df, 'dnm': df_dnm_adj_with_ids}, labels_dict)

    print(log_info + "Empirical Bayes (ComBat-style) harmonization pipeline finished.")
