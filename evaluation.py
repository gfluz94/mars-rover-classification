import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, precision_recall_curve, roc_curve)

def evaluate(y_true, y_pred, y_proba):
        METRICS = ["Accuracy", "Recall", "Precision", "F1-Score", "AUC-ROC"]
        metrics = {}
        if len(np.unique(y_true))==2:
            for metric in METRICS:
                if metric=="Accuracy":
                    metrics[metric] = accuracy_score(y_true, y_pred)
                elif metric=="Precision":
                    metrics[metric] = precision_score(y_true, y_pred)
                elif metric=="Recall":
                    metrics[metric] = recall_score(y_true, y_pred)
                elif metric=="F1-Score":
                    metrics[metric] = f1_score(y_true, y_pred)
                elif metric=="AUC-ROC" and y_proba is not None:
                    metrics[metric] = roc_auc_score(y_true, y_proba)
        else:
            for metric in METRICS:
                if metric=="Accuracy":
                    metrics[metric] = accuracy_score(y_true, y_pred)
                elif metric=="Precision":
                    metrics[metric] = precision_score(y_true, y_pred, average="weighted")
                elif metric=="Recall":
                    metrics[metric] = recall_score(y_true, y_pred, average="weighted")
                elif metric=="F1-Score":
                    metrics[metric] = f1_score(y_true, y_pred, average="weighted")

        return pd.DataFrame(metrics, index=["InceptionNet"])

def binary_evaluation_plot(y_true, y_proba):
    if len(np.unique(y_true))!=2:
        raise ValueError("Multiclass Problem!")

    fig, ax = plt.subplots(2,2,figsize=(12,10))
    __plot_roc(y_true, y_proba, ax[0][0])
    __plot_pr(y_true, y_proba, ax[0][1])
    __plot_cap(y_true, y_proba, ax[1][0])
    __plot_ks(y_true, y_proba, ax[1][1])
    plt.tight_layout()
    fig.savefig("./utils/classification_analysis.png", dpi=500)
    plt.show()
    
def __plot_cap(y_test, y_proba, ax):
    cap_df = pd.DataFrame(data=y_test)
    cap_df["Probability"] = y_proba

    total = cap_df.iloc[:, 0].sum()
    perfect_model = (cap_df.iloc[:, 0].sort_values(ascending=False).cumsum()/total).values
    current_model = (cap_df.sort_values(by="Probability", ascending=False).iloc[:, 0].cumsum()/total).values

    max_area = 0
    covered_area = 0
    h = 1/len(perfect_model)
    random = np.linspace(0, 1, len(perfect_model))
    for i, (am, ap) in enumerate(zip(current_model, perfect_model)):
        try:
            max_area += (ap-random[i]+perfect_model[i+1]-random[i+1])*h/2
            covered_area += (am-random[i]+current_model[i+1]-random[i+1])*h/2
        except:
            continue
    accuracy_ratio = covered_area/max_area

    ax.plot(np.linspace(0, 1, len(current_model)), current_model, 
                        color="green", label=f"InceptionNet: AR = {accuracy_ratio:.3f}")
    ax.plot(np.linspace(0, 1, len(perfect_model)), perfect_model, color="red", label="Perfect Model")
    ax.plot([0,1], [0,1], color="navy")
    ax.set_xlabel("Individuals", fontsize=12)
    ax.set_ylabel("Target Individuals", fontsize=12)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.01))
    ax.legend(loc=4, fontsize=10)
    ax.set_title("CAP Analysis", fontsize=13)
    
def __plot_roc(y_test, y_proba, ax):
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    ax.plot(fpr, tpr, color="red", label=f"InceptionNet (AUC = {roc_auc_score(y_test, y_proba):.3f})")
    ax.plot([0,1], [0,1], color="navy")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.001))
    ax.legend(loc=4)
    ax.set_title("ROC Analysis", fontsize=13)
    
def __plot_pr(y_test, y_proba, ax):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    ax.plot(recall, precision, color="red", label=f"InceptionNet")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.001))
    ax.legend(loc=4)
    ax.set_title("Precision-Recall Analysis", fontsize=13)
    
def __plot_ks(y_test, y_proba, ax):
    prediction_labels = pd.DataFrame(y_test, columns=["True Label"])
    prediction_labels["Probabilities"] = y_proba
    prediction_labels["Thresholds"] = prediction_labels["Probabilities"].apply(lambda x: np.round(x, 2))
    df = prediction_labels.groupby("Thresholds").agg(["count", "sum"])[["True Label"]]
    ks_df = pd.DataFrame(df["True Label"]["sum"]).rename(columns={"sum":"Negative"})
    ks_df["Positive"] = df["True Label"]["count"]-df["True Label"]["sum"]
    ks_df["Negative"] = ks_df["Negative"].cumsum()/ks_df["Negative"].sum()
    ks_df["Positive"] = ks_df["Positive"].cumsum()/ks_df["Positive"].sum()
    ks_df["KS"] = ks_df["Positive"]-ks_df["Negative"]
    ks_df.loc[0.0, :] = [0.0, 0.0, 0.0]
    ks_df = ks_df.sort_index()
    max_ks_thresh = ks_df.KS.idxmax()

    ks_df.drop("KS", axis=1).Negative.plot(color="red", ax=ax)
    ks_df.drop("KS", axis=1).Positive.plot(color="navy", ax=ax)
    ax.set_title("KS Analysis", fontsize=13)
    ax.plot([max_ks_thresh, max_ks_thresh], 
            [ks_df.loc[max_ks_thresh,"Negative"], ks_df.loc[max_ks_thresh,"Positive"]], 
             color="green", label="Max KS")
    ax.text(max_ks_thresh-0.15, 0.5, f"KS={ks_df.loc[max_ks_thresh,'KS']:.3f}", fontsize=12, color="green")
    ax.legend()