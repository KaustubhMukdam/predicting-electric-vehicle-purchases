import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LGBM_OOF_PATH = Path("oof_lgbm_v2.npy")
XGB_OOF_PATH = Path("oof_xgb_v3.npy")
TRAIN_PATH = Path("train.csv")
LGBM_TEST_PATH = Path("test_pred_lgbm_v2.npy")
XGB_TEST_PATH = Path("test_pred_xgb_v3.npy")
XGB_SUBMISSION_PATH = Path("submission_xgb_v3.csv")
OUTPUT_PATH = Path("submission_e4_logistic_stack.csv")

RANDOM_STATE = 42
META_VALIDATION_SIZE = 0.20

def load_xgb_test():
    if XGB_TEST_PATH.exists():
        print(f"Loaded XGB test predictions from: {XGB_TEST_PATH}")
        return np.load(XGB_TEST_PATH)
    sub = pd.read_csv(XGB_SUBMISSION_PATH)
    if "Will_Buy_EV" not in sub.columns:
        raise ValueError("Will_Buy_EV column not found in XGB submission.")
    print(f"Loaded XGB test predictions from: {XGB_SUBMISSION_PATH}")
    return sub["Will_Buy_EV"].to_numpy()

def make_stacker():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                           random_state=RANDOM_STATE)
    )

# ---- Load OOF predictions and target ----
lgbm_oof = np.load(LGBM_OOF_PATH)
xgb_oof = np.load(XGB_OOF_PATH)
y = pd.read_csv(TRAIN_PATH, usecols=["Will_Buy_EV"])["Will_Buy_EV"].astype(int).to_numpy()

if not (len(lgbm_oof) == len(xgb_oof) == len(y)):
    raise ValueError(f"Length mismatch: LGBM={len(lgbm_oof)}, XGB={len(xgb_oof)}, y={len(y)}")

X_oof = np.column_stack([lgbm_oof, xgb_oof])

print("\nBase OOF performance")
print("---------------------")
print(f"Rows     : {len(y):,}")
print(f"LGBM AUC : {roc_auc_score(y, lgbm_oof):.8f}")
print(f"XGB AUC  : {roc_auc_score(y, xgb_oof):.8f}")

# ---- Honest meta-validation ----
X_tr, X_va, y_tr, y_va = train_test_split(
    X_oof, y, test_size=META_VALIDATION_SIZE,
    stratify=y, random_state=RANDOM_STATE
)

stacker = make_stacker()
stacker.fit(X_tr, y_tr)
p_va = stacker.predict_proba(X_va)[:, 1]

auc_lgbm = roc_auc_score(y_va, X_va[:, 0])
auc_xgb = roc_auc_score(y_va, X_va[:, 1])
auc_stack = roc_auc_score(y_va, p_va)

print("\nE4-A held-out meta-validation")
print("------------------------------")
print(f"Validation rows : {len(y_va):,}")
print(f"LGBM AUC        : {auc_lgbm:.8f}")
print(f"XGB AUC         : {auc_xgb:.8f}")
print(f"Stacker AUC     : {auc_stack:.8f}")

lr = stacker.named_steps["logisticregression"]
print("\nLearned coefficients")
print("---------------------")
print(f"Intercept : {lr.intercept_[0]:.8f}")
print(f"LGBM coef : {lr.coef_[0][0]:.8f}")
print(f"XGB coef  : {lr.coef_[0][1]:.8f}")

# ---- Retrain meta-model on all OOF predictions ----
final_stacker = make_stacker()
final_stacker.fit(X_oof, y)

lgbm_test = np.load(LGBM_TEST_PATH)
xgb_test = load_xgb_test()

if len(lgbm_test) != len(xgb_test):
    raise ValueError(f"Test length mismatch: LGBM={len(lgbm_test)}, XGB={len(xgb_test)}")

X_test = np.column_stack([lgbm_test, xgb_test])
stack_test = final_stacker.predict_proba(X_test)[:, 1]

template = pd.read_csv(XGB_SUBMISSION_PATH)
if len(template) != len(stack_test):
    raise ValueError(f"Submission length mismatch: template={len(template)}, pred={len(stack_test)}")

submission = template.copy()
submission["Will_Buy_EV"] = stack_test
submission.to_csv(OUTPUT_PATH, index=False)

print("\nE4 submission")
print("-------------")
print(f"Rows            : {len(submission):,}")
print(f"Prediction mean : {stack_test.mean():.8f}")
print(f"Prediction min  : {stack_test.min():.8f}")
print(f"Prediction max  : {stack_test.max():.8f}")
print(f"Saved           : {OUTPUT_PATH}")

if auc_stack > auc_lgbm:
    print("\nRESULT: Stacker beats LGBM on held-out meta-validation.")
else:
    print("\nRESULT: Stacker does not beat LGBM on held-out meta-validation.")
