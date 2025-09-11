# main.py
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans

from collections import Counter

# Optional (present in your original)
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# ===== Optional fairlearn (safe import) =====
try:
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, DemographicParity
    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False

from dataclasses import dataclass

# --- Page config
st.set_page_config(
    page_title="AI Fairness Playbooks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# Helper metrics & utilities (group fairness)
# ======================================================================

@dataclass
class GroupMetrics:
    tpr: float
    fpr: float
    fnr: float
    fdr: float

def _rates(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true==1) & (y_pred==1)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    tpr = tp / (tp+fn) if (tp+fn)>0 else 0.0
    fpr = fp / (fp+tn) if (fp+tn)>0 else 0.0
    fnr = fn / (tp+fn) if (tp+fn)>0 else 0.0
    fdr = fp / (tp+fp) if (tp+fp)>0 else 0.0
    return GroupMetrics(tpr, fpr, fnr, fdr)

def group_fairness_summary(y_true, y_pred, sensitive):
    """Return per-group metrics and gaps (max-min) for TPR/FPR/FNR/FDR."""
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "s": sensitive})
    groups = sorted(df["s"].unique())
    per_group = {}
    for g in groups:
        sub = df[df["s"]==g]
        per_group[g] = _rates(sub["y"].values, sub["yhat"].values)
    def gap(field):
        vals = [getattr(per_group[g], field) for g in groups]
        return max(vals) - min(vals)
    gaps = {k: gap(k) for k in ["tpr","fpr","fnr","fdr"]}
    return per_group, gaps

def _show_group_metrics(per_group, gaps, label=""):
    cols = st.columns(max(2, len(per_group)+1))
    with cols[0]:
        st.markdown(f"**{label}**")
        st.write({k: f"{v:.3f}" for k,v in gaps.items()})
    i=1
    for g, m in per_group.items():
        with cols[i]:
            st.metric(f"Group {g} TPR", f"{m.tpr:.3f}")
            st.metric(f"Group {g} FPR", f"{m.fpr:.3f}")
        i+=1

def _build_case_report_md(title, notes):
    out = f"# {title}\n\n## Key Notes\n"
    for n in notes:
        out += f"- {n}\n"
    return out

# ======================================================================
# SIMULATIONS (from your app + kept intact)
# ======================================================================

def run_threshold_simulation():
    """Simulation for post-processing threshold optimization."""
    st.markdown("#### Threshold Optimization Simulation")
    st.write("Adjust decision thresholds for two groups and observe how error rates change to achieve **Equal Opportunity** (equal true positive rates).")

    np.random.seed(42)
    scores_a_pos = np.random.normal(0.7, 0.15, 80)
    scores_a_neg = np.random.normal(0.4, 0.15, 120)
    scores_b_pos = np.random.normal(0.6, 0.15, 50)
    scores_b_neg = np.random.normal(0.3, 0.15, 150)

    df_a = pd.DataFrame({'Score': np.concatenate([scores_a_pos, scores_a_neg]), 'Actual': [1]*80 + [0]*120})
    df_b = pd.DataFrame({'Score': np.concatenate([scores_b_pos, scores_b_neg]), 'Actual': [1]*50 + [0]*150})

    col1, col2 = st.columns(2)
    with col1:
        threshold_a = st.slider("Threshold for Group A", 0.0, 1.0, 0.5, key="sim_thresh_a")
    with col2:
        threshold_b = st.slider("Threshold for Group B", 0.0, 1.0, 0.5, key="sim_thresh_b")

    tpr_a = np.mean(df_a[df_a['Actual'] == 1]['Score'] >= threshold_a)
    fpr_a = np.mean(df_a[df_a['Actual'] == 0]['Score'] >= threshold_a)
    tpr_b = np.mean(df_b[df_b['Actual'] == 1]['Score'] >= threshold_b)
    fpr_b = np.mean(df_b[df_b['Actual'] == 0]['Score'] >= threshold_b)

    st.markdown("##### Results")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="True Positive Rate (Group A)", value=f"{tpr_a:.2%}")
        st.metric(label="False Positive Rate (Group A)", value=f"{fpr_a:.2%}")
    with res_col2:
        st.metric(label="True Positive Rate (Group B)", value=f"{tpr_b:.2%}")
        st.metric(label="False Positive Rate (Group B)", value=f"{fpr_b:.2%}")

    if abs(tpr_a - tpr_b) < 0.02:
        st.success(f"You've almost achieved Equal Opportunity! The difference in TPR is only {abs(tpr_a - tpr_b):.2%}.")
    else:
        st.warning(f"Adjust thresholds to equalize True Positive Rates. Current difference: {abs(tpr_a - tpr_b):.2%}")

def run_calibration_simulation():
    st.markdown("#### Calibration Simulation")
    st.write("See how raw model scores can be miscalibrated and how **Platt Scaling** or **Isotonic Regression** adjust them toward the perfect diagonal.")

    np.random.seed(0)
    raw_scores = np.sort(np.random.rand(100))
    true_probs = 1 / (1 + np.exp(-(raw_scores * 4 - 2)))  # simulate "reality"

    # Platt
    lr = LogisticRegression()
    lr.fit(raw_scores.reshape(-1, 1), (true_probs > 0.5).astype(int))
    calibrated_platt = lr.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    # Isotonic
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(raw_scores, true_probs)
    calibrated_isotonic = isotonic.predict(raw_scores)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(raw_scores, true_probs, label='Original Scores (Miscalibrated)')
    ax.plot(raw_scores, calibrated_platt, label='Platt Scaling Calibrated')
    ax.plot(raw_scores, calibrated_isotonic, label='Isotonic Regression Calibrated')
    ax.set_title("Calibration Techniques Comparison")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Positive Fraction")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.info("Goal: curves as close as possible to the dashed line (perfect calibration).")

def run_rejection_simulation():
    st.markdown("#### Classification with Rejection Simulation")
    st.write("Set a confidence threshold; uncertain cases are routed for human review.")

    np.random.seed(1)
    scores = np.random.beta(2, 2, 200)

    low_thresh = st.slider("Lower Confidence Threshold", 0.0, 0.5, 0.25)
    high_thresh = st.slider("Upper Confidence Threshold", 0.5, 1.0, 0.75)

    automated_low = scores[scores <= low_thresh]
    automated_high = scores[scores >= high_thresh]
    rejected = scores[(scores > low_thresh) & (scores < high_thresh)]

    fig, ax = plt.subplots()
    ax.hist(automated_low, bins=10, range=(0,1), label=f'Automatic (Low, n={len(automated_low)})', alpha=0.7)
    ax.hist(rejected, bins=10, range=(0,1), label=f'Rejected (n={len(rejected)})', alpha=0.7)
    ax.hist(automated_high, bins=10, range=(0,1), label=f'Automatic (High, n={len(automated_high)})', alpha=0.7)
    ax.set_title("Decision Distribution")
    ax.set_xlabel("Model Probability Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    coverage = (len(automated_low) + len(automated_high)) / len(scores)
    st.metric("Coverage Rate (Automation)", f"{coverage:.1%}")
    st.info("Wider rejection range → more fairness in hard cases, less automation.")

def run_matching_simulation():
    st.markdown("#### Matching Simulation")
    st.write("Compare two groups by matching similar individuals.")

    np.random.seed(0)
    x_treat = np.random.normal(5, 1.5, 50)
    y_treat = 2 * x_treat + 5 + np.random.normal(0, 2, 50)
    x_control = np.random.normal(3.5, 1.5, 50)
    y_control = 2 * x_control + np.random.normal(0, 2, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.scatter(x_treat, y_treat, c='red', label='Treatment', alpha=0.7)
    ax1.scatter(x_control, y_control, c='blue', label='Control', alpha=0.7)
    ax1.set_title("Before Matching")
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Outcome")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    matched_indices = [np.argmin(np.abs(x_c - x_treat)) for x_c in x_control]
    x_treat_matched = x_treat[matched_indices]
    y_treat_matched = y_treat[matched_indices]

    ax2.scatter(x_treat_matched, y_treat_matched, c='red', label='Treatment (Matched)', alpha=0.7)
    ax2.scatter(x_control, y_control, c='blue', label='Control', alpha=0.7)
    ax2.set_title("After Matching")
    ax2.set_xlabel("Feature")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)
    st.info("Right: matched subset makes groups more comparable.")

def run_rd_simulation():
    st.markdown("#### Regression Discontinuity (RD) Simulation")
    st.write("Treatment assigned based on a cutoff; compare just above/below cutoff.")

    np.random.seed(42)
    cutoff = st.slider("Cutoff Value", 40, 60, 50, key="rd_cutoff")

    x = np.linspace(0, 100, 200)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 200)
    treatment_effect = 15
    y[x >= cutoff] += treatment_effect

    fig, ax = plt.subplots()
    ax.scatter(x[x < cutoff], y[x < cutoff], c='blue', label='Control')
    ax.scatter(x[x >= cutoff], y[x >= cutoff], c='red', label='Treatment')
    ax.axvline(x=cutoff, color='gray', linestyle='--', label=f'Cutoff at {cutoff}')
    ax.set_title("Treatment Effect at the Cutoff")
    ax.set_xlabel("Assignment Variable")
    ax.set_ylabel("Outcome")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.info(f"The outcome jump at cutoff ≈ **{treatment_effect}** units.")

def run_did_simulation():
    st.markdown("#### Difference-in-Differences (DiD) Simulation")
    st.write("Compare changes over time: treatment vs control.")

    time = ['Before', 'After']
    control_outcomes = [20, 25]
    treat_outcomes = [15, 28]

    fig, ax = plt.subplots()
    ax.plot(time, control_outcomes, 'o-', label='Control')
    ax.plot(time, treat_outcomes, 'o-', label='Treatment')

    counterfactual = [treat_outcomes[0], treat_outcomes[0] + (control_outcomes[1] - control_outcomes[0])]
    ax.plot(time, counterfactual, '--', label='Treatment (Counterfactual)')

    ax.set_title("Treatment Effect Estimation with DiD")
    ax.set_ylabel("Outcome")
    ax.set_ylim(10, 35)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    effect = treat_outcomes[1] - counterfactual[1]
    st.info(f"Estimated treatment effect: **{effect}** units.")

# ======================================================================
# Bias Mitigation Techniques Toolkit (pre/in/post helpers)
# ======================================================================

def bias_mitigation_techniques_toolkit():
    """Interactive toolkit + code templates for bias mitigation techniques."""
    st.header("🔧 Bias Mitigation Techniques Toolkit")

    with st.expander("🔍 Friendly Definition"):
        st.write("""
**Bias Mitigation Techniques** balance your dataset before training: oversampling, undersampling, reweighting, SMOTE, augmentation, and fair clustering.
""")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resampling Techniques", "Reweighting", "Data Augmentation",
        "Fair Clustering", "SMOTE", "Interactive Comparison"
    ])

    # TAB 1: Resampling
    with tab1:
        st.subheader("Resampling Techniques")
        with st.expander("💡 Interactive Oversampling Simulation"):
            np.random.seed(42)
            majority_size = st.slider("Majority group size", 100, 1000, 800, key="maj_size")
            minority_size = st.slider("Minority group size", 50, 500, 200, key="min_size")

            original_ratio = minority_size / (majority_size + minority_size)
            target_ratio = st.radio("Target balance", ["50-50", "60-40", "70-30"], key="target_balance")
            if target_ratio == "50-50":
                new_minority_size = majority_size
            elif target_ratio == "60-40":
                new_minority_size = int(majority_size * 0.67)
            else:
                new_minority_size = int(majority_size * 0.43)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.bar(['Majority', 'Minority'], [majority_size, minority_size])
            ax1.set_title(f"Original Ratio: {original_ratio:.1%} minority")
            ax1.set_ylabel("Count")
            ax2.bar(['Majority', 'Minority'], [majority_size, new_minority_size])
            ax2.set_title(f"After Oversampling: {new_minority_size/(majority_size+new_minority_size):.1%} minority")
            ax2.set_ylabel("Count")
            st.pyplot(fig)

            replication_factor = new_minority_size / minority_size
            st.info(f"Replication factor ≈ {replication_factor:.1f}x")

        st.code("""
from sklearn.utils import resample

def apply_oversampling(data, target_column, minority_class):
    majority = data[data[target_column] != minority_class]
    minority = data[data[target_column] == minority_class]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    return pd.concat([majority, minority_upsampled])
        """, language="python")

        st.text_area("Apply to your case:", key="resample_plan",
                     placeholder="Example: Hiring dataset 80/20 → oversample to ~60/40.")

    # TAB 2: Reweighting
    with tab2:
        st.subheader("Sample Reweighting")
        with st.expander("💡 Interactive Weighting Simulation"):
            legitimate_pct = st.slider("Percentage legitimate", 80, 99, 95, key="legit_pct")
            fraud_pct = 100 - legitimate_pct
            weight_legit = 1 / (legitimate_pct / 100)
            weight_fraud = 1 / (fraud_pct / 100)

            fig, ax = plt.subplots()
            categories = ['Legitimate', 'Fraudulent']
            percentages = [legitimate_pct, fraud_pct]
            weights = [weight_legit, weight_fraud]
            x = np.arange(len(categories)); width = 0.35
            ax.bar(x - width/2, percentages, width, label='Data %')
            ax.bar(x + width/2, weights, width, label='Weight')
            ax.set_xticks(x); ax.set_xticklabels(categories)
            ax.legend()
            st.pyplot(fig)
            st.info(f"Fraud gets {weight_fraud:.1f}x weight")

        st.code("""
from sklearn.utils.class_weight import compute_class_weight
def apply_reweighting(X, y):
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    sample_weights = np.array([weights[label] for label in y])
    return sample_weights
        """, language="python")

        st.text_area("Apply to your case:", key="reweight_plan",
                     placeholder="Example: Disease prevalence 3% → inverse frequency weights.")

    # TAB 3: Augmentation
    with tab3:
        st.subheader("Data Augmentation for Fairness")
        with st.expander("💡 Interactive Augmentation Visualization"):
            augmentation_factor = st.slider("Augmentation factor (minority)", 1, 10, 5, key="aug_factor")
            np.random.seed(1)
            original_samples = 20
            augmented_samples = original_samples * augmentation_factor
            original_data = np.random.multivariate_normal([2, 3], [[1, 0.5], [0.5, 1]], original_samples)

            augmented_data = []
            for _ in range(augmented_samples - original_samples):
                base_sample = original_data[np.random.randint(0, original_samples)]
                augmented_sample = base_sample + np.random.normal(0, 0.3, 2)
                augmented_data.append(augmented_sample)
            augmented_data = np.array(augmented_data) if len(augmented_data)>0 else np.empty((0,2))
            combined_data = np.vstack([original_data, augmented_data]) if len(augmented_data)>0 else original_data

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(original_data[:, 0], original_data[:, 1], alpha=0.7, s=50)
            ax1.set_title(f"Original Minority (n={original_samples})")
            ax2.scatter(original_data[:, 0], original_data[:, 1], alpha=0.7, s=50, label='Original')
            if len(augmented_data)>0:
                ax2.scatter(augmented_data[:, 0], augmented_data[:, 1], alpha=0.5, s=30, marker='x', label='Augmented')
            ax2.set_title(f"After Augmentation (n={len(combined_data)})"); ax2.legend()
            st.pyplot(fig)
            st.info(f"Minority: {original_samples} → {len(combined_data)}")

        st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_image_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=15,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    return datagen
        """, language="python")

        st.text_area("Apply to your case:", key="augment_plan",
                     placeholder="Example: Facial recognition—rotation/lighting/background augmentation for underrepresented groups.")

    # TAB 4: Fair Clustering (simplified demo)
    with tab4:
        st.subheader("Fair Clustering Techniques")
        with st.expander("💡 Interactive Fair Clustering Demo"):
            np.random.seed(10)
            group_a = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 80)
            group_b = np.random.multivariate_normal([6, 6], [[1, 0.3], [0.3, 1]], 20)
            all_data = np.vstack([group_a, group_b])
            group_labels = np.array(['A'] * 80 + ['B'] * 20)

            n_clusters = st.slider("Number of clusters", 2, 5, 3, key="fair_clusters")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            traditional_clusters = kmeans.fit_predict(all_data)

            fair_clusters = traditional_clusters.copy()
            for cluster_id in range(n_clusters):
                cluster_mask = traditional_clusters == cluster_id
                cluster_groups = group_labels[cluster_mask]
                group_a_pct = np.mean(cluster_groups == 'A')
                if group_a_pct > 0.9:
                    b_points = np.where((group_labels == 'B') & (traditional_clusters != cluster_id))[0]
                    if len(b_points) > 0:
                        fair_clusters[b_points[:2]] = cluster_id
                elif group_a_pct < 0.1:
                    a_points = np.where((group_labels == 'A') & (traditional_clusters != cluster_id))[0]
                    if len(a_points) > 0:
                        fair_clusters[a_points[:2]] = cluster_id

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            for i in range(n_clusters):
                pts = all_data[traditional_clusters == i]
                if len(pts) > 0:
                    ax1.scatter(pts[:,0], pts[:,1], c=colors[i], alpha=0.7, label=f'Cluster {i}')
            ax1.set_title("Traditional K-Means"); ax1.legend(); ax1.grid(True, alpha=0.3)

            for i in range(n_clusters):
                pts = all_data[fair_clusters == i]
                if len(pts) > 0:
                    ax2.scatter(pts[:,0], pts[:,1], c=colors[i], alpha=0.7, label=f'Cluster {i}')
            ax2.set_title("Fair Clustering (Balanced)"); ax2.legend(); ax2.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.code("""
def fair_clustering(X, sensitive_features, n_clusters, balance_threshold=0.3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    initial_clusters = kmeans.fit_predict(X)
    # ... check cluster composition and rebalance if needed ...
    return initial_clusters
        """, language="python")

    # TAB 5: SMOTE
    with tab5:
        st.subheader("SMOTE (Synthetic Minority Oversampling Technique)")
        with st.expander("💡 Interactive SMOTE Visualization"):
            np.random.seed(5)
            minority_samples = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 1]], 15)
            majority_samples = np.random.multivariate_normal([1, 1], [[1, 0.2], [0.2, 1]], 85)
            k_neighbors = st.slider("Neighbors (simulated)", 1, 5, 3, key="smote_k")
            synthetic_count = st.slider("Synthetic samples", 10, 50, 25, key="smote_count")
            synthetic = []
            for _ in range(synthetic_count):
                base_idx = np.random.randint(0, len(minority_samples))
                base = minority_samples[base_idx]
                distances = np.linalg.norm(minority_samples - base, axis=1)
                neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
                neighbor = minority_samples[np.random.choice(neighbor_indices)]
                alpha = np.random.random()
                synthetic.append(base + alpha*(neighbor - base))
            synthetic = np.array(synthetic)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(majority_samples[:,0], majority_samples[:,1], alpha=0.6, s=50, label=f'Majority (n={len(majority_samples)})')
            ax.scatter(minority_samples[:,0], minority_samples[:,1], alpha=0.8, s=80, label=f'Minority (n={len(minority_samples)})')
            ax.scatter(synthetic[:,0], synthetic[:,1], alpha=0.7, s=60, marker='x', label=f'Synthetic (n={len(synthetic)})')
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.code("""
from imblearn.over_sampling import SMOTE

def apply_smote(X, y, sampling_strategy='auto'):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    print("Original:", Counter(y)); print("After:", Counter(y_res))
    return X_res, y_res
        """, language="python")

    # TAB 6: Comparison
    with tab6:
        st.subheader("Technique Comparison Tool")
        col1, col2 = st.columns(2)
        with col1:
            majority_size = st.number_input("Majority size", 100, 2000, 800, key="comp_maj")
            minority_size = st.number_input("Minority size", 20, 500, 150, key="comp_min")
        with col2:
            target_balance = st.selectbox("Target balance", ["50-50","60-40","70-30"], key="comp_balance")
            techniques = st.multiselect("Select techniques", ["Oversampling", "Undersampling", "Reweighting", "SMOTE"],
                                        default=["Oversampling","SMOTE"], key="comp_techniques")

        results = {}
        if "Oversampling" in techniques:
            new_minority = {"50-50": majority_size, "60-40": int(0.67*majority_size), "70-30": int(0.43*majority_size)}[target_balance]
            results["Oversampling"] = {
                "Final Majority": majority_size,
                "Final Minority": new_minority,
                "Total": majority_size + new_minority,
                "Information Loss": "None",
                "Overfitting Risk": "Medium" if new_minority > minority_size*3 else "Low"
            }
        if "Undersampling" in techniques:
            new_majority = minority_size
            results["Undersampling"] = {
                "Final Majority": new_majority,
                "Final Minority": minority_size,
                "Total": new_majority + minority_size,
                "Information Loss": f"{((majority_size - new_majority)/majority_size*100):.1f}%",
                "Overfitting Risk": "Low"
            }
        if "Reweighting" in techniques:
            weight_ratio = majority_size/minority_size
            results["Reweighting"] = {
                "Final Majority": majority_size,
                "Final Minority": minority_size,
                "Total": majority_size + minority_size,
                "Minority Weight": f"{weight_ratio:.1f}x",
                "Information Loss": "None"
            }
        if "SMOTE" in techniques:
            synthetic_needed = max(0, majority_size - minority_size)
            results["SMOTE"] = {
                "Final Majority": majority_size,
                "Final Minority": minority_size + synthetic_needed,
                "Total": majority_size + minority_size + synthetic_needed,
                "Synthetic Samples": synthetic_needed,
                "Information Loss": "None"
            }

        if results:
            st.dataframe(pd.DataFrame(results).T)
            if majority_size > 5*minority_size:
                st.warning("High imbalance: consider SMOTE + slight undersampling.")
            if "Undersampling" in results and float(results["Undersampling"]["Information Loss"].strip('%')) > 50:
                st.warning("Undersampling would lose >50% of data—prefer oversampling/SMOTE.")

        st.text_area("Document your selection & rationale:", key="comparison_conclusion",
                     placeholder="Given 85-15 imbalance and small minority (n=150), we'll use SMOTE plus slight undersampling to reach ~70-30.")

    # Pipeline + Reports hook (from your original flow)
    st.markdown("---")
    st.subheader("🔗 Complete Bias Mitigation Pipeline")
    st.code("""
def complete_bias_mitigation_pipeline(X, y, sensitive_attr, strategy='balanced'):
    from sklearn.model_selection import train_test_split
    print("=== BIAS ANALYSIS ==="); analyze_bias(X, y, sensitive_attr)
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    weights = compute_class_weight('balanced', classes=np.unique(y_bal), y=y_bal)
    sample_weights = np.array([weights[val] for val in y_bal])
    model = LogisticRegression(random_state=42)
    model.fit(X_bal, y_bal, sample_weight=sample_weights)
    X_test_bal, _, y_test_bal, _ = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
    validate_fairness(model, X_test_bal, y_test_bal, sensitive_attr)
    return model, X_bal, y_bal
    """, language="python")

    st.text_area("SMOTE adaptation to your data:", key="smote_plan",
                 placeholder="Tabular medical data: SMOTE with k=5 for rare diseases; constrain feature ranges for realism.")

    # Report generator
    st.markdown("---")
    st.header("Generate Bias Mitigation Report")
    if st.button("Generate Bias Mitigation Report", key="gen_bias_mit_report"):
        report_data = {
            "Resampling Strategy": {"Selected Approach": st.session_state.get('resample_plan', 'Not completed')},
            "Reweighting Strategy": {"Implementation Plan": st.session_state.get('reweight_plan', 'Not completed')},
            "Data Augmentation": {"Augmentation Strategy": st.session_state.get('augment_plan', 'Not completed')},
            "SMOTE Application": {"SMOTE Adaptation": st.session_state.get('smote_plan', 'Not completed')},
            "Technique Comparison": {"Selection Rationale": st.session_state.get('comparison_conclusion', 'Not completed')}
        }
        report_md = "# Bias Mitigation Techniques Report\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        st.session_state.bias_mit_report_md = report_md
        st.success("✅ Bias Mitigation Report generated!")

    if 'bias_mit_report_md' in st.session_state and st.session_state.bias_mit_report_md:
        st.subheader("Report Preview")
        st.markdown(st.session_state.bias_mit_report_md)
        st.download_button("Download Bias Mitigation Report",
                           st.session_state.bias_mit_report_md,
                           "bias_mitigation_report.md",
                           "text/markdown")

def analyze_bias(X, y, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    for g in unique_groups:
        mask = sensitive_attr == g
        rate = np.mean(y[mask])
        print(f"Group {g}: {np.sum(mask)} samples, {rate:.2%} positive rate")

def validate_fairness(model, X_test, y_test, sensitive_attr):
    preds = model.predict(X_test)
    per_group, gaps = group_fairness_summary(y_test, preds, sensitive_attr)
    print("=== FAIRNESS METRICS ===")
    for g, m in per_group.items():
        print(f"Group {g}: TPR={m.tpr:.3f}, FPR={m.fpr:.3f}")
    print("Gaps:", {k: f"{v:.3f}" for k,v in gaps.items()})

# ======================================================================
# Causal Fairness Toolkit (kept and expanded with your content)
# ======================================================================

def causal_fairness_toolkit():
    st.header("🛡️ Causal Fairness Toolkit")
    with st.expander("🔍 Friendly Definition"):
        st.write("""
**Causal Analysis** explains *why* disparities arise, enabling targeted fixes rather than treating symptoms.
""")

    if 'causal_report' not in st.session_state:
        st.session_state.causal_report = {}

    tab1, tab2, tab3, tab4, tab5  = st.tabs(["Identification", "Counterfactual Analysis", "Causal Diagram", "Causal Inference","Intersectionality"])

    # Identification
    with tab1:
        st.subheader("Framework for Identifying Discrimination Mechanisms")
        st.info("Identify possible root causes of bias.")
        with st.expander("Direct Discrimination"):
            st.write("Protected attribute directly used in decision.")
        st.text_area("1. Direct influence?", key="causal_q1")
        with st.expander("Indirect Discrimination"):
            st.write("Protected attribute affects a mediating (legitimate) factor.")
        st.text_area("2. Indirect via mediators?", key="causal_q2")
        with st.expander("Proxy Discrimination"):
            st.write("Neutral variable proxies a protected attribute.")
        st.text_area("3. Proxies used?", key="causal_q3")

    # Counterfactual
    with tab2:
        st.subheader("Practical Counterfactual Fairness Methodology")
        st.info("Analyze and mitigate counterfactual bias.")
        with st.expander("💡 Interactive Example: Counterfactual Simulation"):
            base_score = 650; base_decision = "Rejected"
            st.write(f"**Base:** Group B, score **{base_score}** → **{base_decision}**.")
            if st.button("View Counterfactual (Change to Group A)", key="cf_button"):
                st.info("Counterfactual: Group A, score **710** → **Approved**. Suggests problematic dependency.")

        with st.container(border=True):
            st.markdown("##### Step 1: Counterfactual Analysis")
            st.text_area("1.1 Counterfactual Queries", key="causal_q4")
            st.text_area("1.2 Fair vs Unfair Paths", key="causal_q5")
            st.text_area("1.3 Measure Disparities", key="causal_q6")

        with st.container(border=True):
            st.markdown("##### Step 2: Specific Path Analysis")
            st.text_area("2.1 Decompose & Classify Paths", key="causal_q7")
            st.text_area("2.2 Quantify Contribution", key="causal_q8")

        with st.container(border=True):
            st.markdown("##### Step 3: Intervention Design")
            st.selectbox("3.1 Select Approach", ["Data Level", "Model Level", "Post-processing"], key="causal_q9")
            st.text_area("3.2 Implement & Monitor", key="causal_q10")

    # Diagram
    with tab3:
        st.subheader("Initial Causal Diagram Approach")
        st.info("Sketch causal relationships & assumptions.")
        nodes = ["Gender", "Education", "Income", "Loan_Decision"]
        possible_relations = [
            ("Gender", "Education"), ("Gender", "Income"),
            ("Education", "Income"), ("Income", "Loan_Decision"),
            ("Education", "Loan_Decision"), ("Gender", "Loan_Decision")
        ]
        st.multiselect("Select relationships (Cause → Effect):",
                       options=[f"{c} → {e}" for c, e in possible_relations],
                       key="causal_q11_relations")
        if st.session_state.get("causal_q11_relations"):
            dot = "digraph { rankdir=LR; "
            for rel in st.session_state.causal_q11_relations:
                c, e = rel.split(" → ")
                dot += f'"{c}" -> "{e}"; '
            dot += "}"
            st.graphviz_chart(dot)
        st.text_area("Assumptions & Paths", key="causal_q11",
                     placeholder="Path (!): Race -> Income -> Decision...")

    # Inference
    with tab4:
        st.subheader("Causal Inference with Limited Data")
        with st.expander("Matching"):
            st.write("Compare 'twins' across groups.")
        with st.expander("💡 Matching Demo"):
            run_matching_simulation()
        with st.expander("Instrumental Variables (IV)"):
            st.graphviz_chart("""
            digraph {
                rankdir=LR;
                Z [label="Instrument (Z)"];
                A [label="Protected Attribute (A)"];
                Y [label="Outcome (Y)"];
                U [label="Unobserved (U)", style=dashed];
                Z -> A; A -> Y; U -> A [style=dashed]; U -> Y [style=dashed];
            }""")
            st.write("Example: Proximity to university as IV for education → income.")
        with st.expander("Regression Discontinuity (RD)"):
            st.write("Use a cutoff to estimate effect.")
        with st.expander("💡 RD Demo"):
            run_rd_simulation()
        with st.expander("Difference-in-Differences (DiD)"):
            st.write("Compare trends over time.")
        with st.expander("💡 DiD Demo"):
            run_did_simulation()

    # Intersectionality
    with tab5:
        st.subheader("Intersectionality in Causal Analysis")
        with st.expander("🔍 Friendly Definition"):
            st.write("Different subgroups can have distinct causal pathways.")
        with st.expander("💡 Diagram Example"):
            col1, col2 = st.columns(2)
            with col1:
                st.graphviz_chart("""
                digraph { rankdir=LR;
                  Gender -> "Years of Experience";
                  Race -> "Type of Education";
                  "Years of Experience" -> "Decision";
                  "Type of Education" -> "Decision";
                }""")
            with col2:
                st.graphviz_chart("""
                digraph { rankdir=LR;
                  subgraph cluster_0 { label="Intersectional Identity"; "Black Woman" [shape=box]; }
                  "Black Woman" -> "Access to Networks";
                  "Access to Networks" -> "Decision";
                  "Gender" -> "Years of Experience" -> "Decision";
                  "Race" -> "Type of Education" -> "Decision";
                }""")
        st.text_area("Intersectional Paths in your system:", key="causal_intersectional")

    # Report
    st.markdown("---")
    st.header("Generate Causal Toolkit Report")
    if st.button("Generate Causal Report", key="gen_causal_report"):
        report_data = {
            "Identification of Mechanisms": {
                "Direct Discrimination": st.session_state.get('causal_q1', 'Not completed'),
                "Indirect Discrimination": st.session_state.get('causal_q2', 'Not completed'),
                "Proxy Discrimination": st.session_state.get('causal_q3', 'Not completed'),
            },
            "Counterfactual Analysis": {
                "Counterfactual Queries": st.session_state.get('causal_q4', 'Not completed'),
                "Causal Path Identification": st.session_state.get('causal_q5', 'Not completed'),
                "Disparity Measurement": st.session_state.get('causal_q6', 'Not completed'),
                "Path Decomposition": st.session_state.get('causal_q7', 'Not completed'),
                "Contribution Quantification": st.session_state.get('causal_q8', 'Not completed'),
                "Selected Intervention Approach": st.session_state.get('causal_q9', 'Not completed'),
                "Implementation & Monitoring Plan": st.session_state.get('causal_q10', 'Not completed'),
            },
            "Causal Diagram": {
                "Selected Relationships": ", ".join(st.session_state.get('causal_q11_relations', [])),
                "Assumptions Documentation": st.session_state.get('causal_q11', 'Not completed'),
            }
        }
        md = "# Causal Fairness Toolkit Report\n\n"
        for section, content in report_data.items():
            md += f"## {section}\n"
            for k,v in content.items(): md += f"**{k}:**\n{v}\n\n"
        st.session_state.causal_report_md = md
        st.success("✅ Report generated!")

    if st.session_state.get('causal_report_md'):
        st.subheader("Report Preview")
        st.markdown(st.session_state.causal_report_md)
        st.download_button("Download Causal Fairness Report",
                           st.session_state.causal_report_md,
                           "causal_fairness_report.md",
                           "text/markdown")

# ======================================================================
# Pre-processing Fairness Toolkit (kept from your app, condensed)
# ======================================================================

def preprocessing_fairness_toolkit():
    st.header("🧪 Pre-processing Fairness Toolkit")
    with st.expander("🔍 Friendly Definition"):
        st.write("Prepare the data to reduce bias before training (representation, proxies, labels, reweighting/resampling, transformations, generation, intersectionality).")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Representation Analysis", "Correlation Detection", "Label Quality",
        "Re-weighting & Re-sampling", "Transformation", "Data Generation",
        "🌍 Intersectionality", "🔧 Integrated Techniques"
    ])

    # Representation Analysis
    with tab1:
        st.subheader("Multidimensional Representation Analysis")
        with st.expander("💡 Representation Gap"):
            pop_a = 50; pop_b = 50
            col1, col2 = st.columns(2)
            with col1:
                data_a = st.slider("Percentage of Group A in your data", 0, 100, 70)
            data_b = 100 - data_a
            df = pd.DataFrame({'Group':['Group A','Group B'],
                               'Reference Population':[pop_a,pop_b],
                               'Your Data':[data_a,data_b]})
            with col2:
                st.dataframe(df.set_index('Group'))
            if abs(data_a - pop_a) > 10:
                st.warning(f"Group A over/underrepresented by {abs(data_a-pop_a)} pp.")
            else:
                st.success("Representation aligned with reference.")

        st.text_area("1. Compare to population", key="p1")
        st.text_area("2. Intersectional analysis", key="p2")
        st.text_area("3. Across outcomes", key="p3")

    # Correlations / proxies
    with tab2:
        st.subheader("Correlation Pattern Detection")
        with st.expander("💡 Proxy Detection"):
            np.random.seed(1)
            group = np.random.randint(0, 2, 100)
            proxy = group * 20 + np.random.normal(50, 5, 100)
            outcome = proxy * 5 + np.random.normal(100, 20, 100)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(group, proxy, alpha=0.7); ax1.set_title("Group vs Proxy")
            ax2.scatter(proxy, outcome, alpha=0.7); ax2.set_title("Proxy vs Outcome")
            st.pyplot(fig)
        st.text_area("1. Direct correlations", key="p4")
        st.text_area("2. Proxy identification", key="p5")

    # Labels
    with tab3:
        st.subheader("Label Quality Evaluation")
        with st.expander("🔎"):
            st.write("Check historical and annotator bias in labels.")
        st.text_area("1. Historical bias in decisions", key="p6")
        st.text_area("2. Annotator bias", key="p7")

    # Reweighting & resampling
    with tab4:
        st.subheader("Re-weighting and Re-sampling")
        with st.expander("💡 Oversampling Demo"):
            np.random.seed(0)
            data_a = np.random.multivariate_normal([2, 2], [[1, .5], [.5, 1]], 100)
            data_b = np.random.multivariate_normal([4, 4], [[1, .5], [.5, 1]], 20)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(data_a[:,0], data_a[:,1], alpha=0.6, label='A (n=100)')
            ax1.scatter(data_b[:,0], data_b[:,1], alpha=0.6, label='B (n=20)')
            ax1.set_title("Original"); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)
            oversample_idx = np.random.choice(range(20), 80, replace=True)
            data_b_over = np.vstack([data_b, data_b[oversample_idx]])
            ax2.scatter(data_a[:,0], data_a[:,1], alpha=0.6, label='A (n=100)')
            ax2.scatter(data_b_over[:,0], data_b_over[:,1], alpha=0.6, marker='x', label='B (n=100)')
            ax2.set_title("After Oversampling"); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        st.text_area("Decision: reweight or resample?", key="p8")
        st.text_area("Intersectionality consideration", key="p9")

    # Transformations
    with tab5:
        st.subheader("Distribution Transformation")
        st.text_area("1. Disparate Impact Removal", key="p10")
        st.text_area("2. Fair Representations (LFR/LAFTR)", key="p11")
        st.text_area("3. Intersectionality", key="p12")

    # Data generation
    with tab6:
        st.subheader("Fairness-Aware Data Generation")
        st.markdown("**When**: severe underrepresentation or need counterfactuals.")
        st.text_area("Intersectionality focus", key="p13")

    # Intersectionality tab
    with tab7:
        st.subheader("Interseccionalidad en el Pre-procesamiento")
        with st.expander("💡 Re-muestreo Estratificado Interseccional"):
            np.random.seed(1)
            hombres_a = pd.DataFrame({'x1': np.random.normal(2, 1, 80), 'x2': np.random.normal(5, 1, 80), 'Grupo': 'Hombres A'})
            mujeres_a = pd.DataFrame({'x1': np.random.normal(2.5, 1, 20), 'x2': np.random.normal(5.5, 1, 20), 'Grupo': 'Mujeres A'})
            hombres_b = pd.DataFrame({'x1': np.random.normal(6, 1, 50), 'x2': np.random.normal(2, 1, 50), 'Grupo': 'Hombres B'})
            mujeres_b = pd.DataFrame({'x1': np.random.normal(6.5, 1, 50), 'x2': np.random.normal(2.5, 1, 50), 'Grupo': 'Mujeres B'})
            mujeres_b_int = pd.DataFrame({'x1': np.random.normal(7, 1, 10), 'x2': np.random.normal(3, 1, 10), 'Grupo': 'Mujeres B (Intersección)'})
            df_original = pd.concat([hombres_a, mujeres_a, hombres_b, mujeres_b, mujeres_b_int])
            factor = st.slider("Sobremuestreo para 'Mujeres B (Intersección)'", 1, 10, 5, key="inter_remuestreo")
            if factor > 1:
                idx = mujeres_b_int.sample(n=(factor-1)*len(mujeres_b_int), replace=True).index
                df_res = pd.concat([df_original, mujeres_b_int.loc[idx]])
            else:
                df_res = df_original
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6), sharex=True, sharey=True)
            for name, grp in df_original.groupby('Grupo'):
                ax1.scatter(grp['x1'], grp['x2'], label=f"{name} (n={len(grp)})", alpha=0.7)
            ax1.set_title("Original"); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
            for name, grp in df_res.groupby('Grupo'):
                ax2.scatter(grp['x1'], grp['x2'], label=f"{name} (n={len(grp)})", alpha=0.7)
            ax2.set_title("Con Sobremuestreo Interseccional"); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        st.text_area("Aplica a tu caso:", key="p_inter")

    # Integrated techniques
    with tab8:
        st.subheader("Integrated Bias Mitigation Techniques")
        st.info("Pick a technique and get starter code.")
        technique = st.selectbox("Technique:", ["Oversampling", "Undersampling", "Reweighting", "SMOTE", "Data Augmentation"], key="bias_mit_selector")
        if technique == "Oversampling":
            st.code("""
def preprocessing_oversampling(data, target_col, protected_attr):
    results = {}
    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group]
        maj = group_data[group_data[target_col]==0]
        mino = group_data[group_data[target_col]==1]
        mino_up = resample(mino, replace=True, n_samples=len(maj), random_state=42)
        results[group] = pd.concat([maj, mino_up])
    return pd.concat(list(results.values()))
            """, language="python")
        elif technique == "Undersampling":
            st.code("""
def preprocessing_undersampling(data, target_col):
    maj = data[data[target_col]==0]
    mino = data[data[target_col]==1]
    maj_down = resample(maj, replace=False, n_samples=len(mino), random_state=42)
    return pd.concat([maj_down, mino])
            """, language="python")
        elif technique == "Reweighting":
            st.code("""
from sklearn.utils.class_weight import compute_class_weight
def compute_sample_weights(y):
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return np.array([weights[val] for val in y])
            """, language="python")
        elif technique == "SMOTE":
            st.code("""
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
            """, language="python")
        else:
            st.code("# See ImageDataGenerator example in the Augmentation tab", language="python")

# ======================================================================
# NEW: In-Processing Fairness Toolkit
# ======================================================================

def inprocessing_fairness_toolkit():
    st.header("⚙️ In-Processing Fairness Toolkit")
    st.write("Inject fairness during training via constraints, adversaries, or custom penalties.")

    with st.expander("💡 What is it?"):
        st.markdown("""
- **Reductions/Constraints**: wrap a learner with a fairness constraint (Equalized Odds / Demographic Parity).
- **Adversarial Debiasing**: remove sensitive info from representations by training an adversary.
- **Regularization/Penalties**: add soft penalties (e.g., TPR gap) into the loss (Lagrangian idea).
        """)

    if not FAIRLEARN_AVAILABLE:
        st.warning("`fairlearn` not found. Install locally with: `pip install fairlearn`")

    st.subheader("1) Reductions (Constraints) — Template")
    st.code("""
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, DemographicParity

base = LogisticRegression(solver="liblinear", max_iter=1000)
constraint = EqualizedOdds()  # or DemographicParity()
mitigator = ExponentiatedGradient(base, constraint)
mitigator.fit(X_train, y_train, sensitive_features=s_train)
y_pred = mitigator.predict(X_test)
    """, language="python")

    st.subheader("2) Adversarial Debiasing — Concept (PyTorch)")
    st.code("""
# Loss = BCE_y - beta * BCE_adversary
# Alternate updates: predictor tries to predict y; adversary tries to predict s from representation.
    """, language="python")

    st.subheader("3) Soft Fairness Penalty (Lagrangian)")
    st.code("""
# total_loss = pred_loss + lambda_fair * tpr_gap
# where tpr_gap computed per-batch by sensitive group.
    """, language="python")

    with st.expander("📎 When to use which?"):
        st.markdown("""
- **Constraints**: sklearn pipelines, explicit metric control (EO/DP).
- **Adversarial**: deep models, need representation scrubbing.
- **Penalties**: flexible custom trade-offs with fast iteration.
        """)

# ======================================================================
# NEW: Case Studies (Toy + Justice Translation)
# ======================================================================

def case_studies():
    st.header("📚 Case Studies — Intervention End-to-End")
    choice = st.radio("Choose a case:", ["Toy Case (Comprehensive Demo)",
                                         "AI Translation → Indigenous Languages in Justice (Simulated)"])
    if choice.startswith("Toy"):
        toy_case_demo()
    else:
        justice_translation_case()

def toy_case_demo():
    st.subheader("Toy Case: Full Pre → In → Post Pipeline")
    st.caption("Synthetic binary classification with a binary sensitive attribute (Group A/B).")

    rng = np.random.RandomState(0)
    nA, nB = 600, 200
    XA = rng.normal(loc=[0,0], scale=[1.0,1.0], size=(nA,2))
    XB = rng.normal(loc=[0.8,0.8], scale=[1.0,1.0], size=(nB,2))
    X = np.vstack([XA, XB])
    s = np.array(["A"]*nA + ["B"]*nB)
    logits = 0.8*X[:,0] + 0.8*X[:,1] - (s=="B")*0.4
    y = (logits + rng.normal(0,0.8,size=len(X)) > 0.0).astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.3, random_state=42, stratify=s
    )

    st.markdown("### Baseline (no mitigation)")
    base = LogisticRegression(solver="liblinear")
    base.fit(X_train, y_train)
    y_hat_base = (base.predict_proba(X_test)[:,1] >= 0.5).astype(int)
    per_group, gaps = group_fairness_summary(y_test, y_hat_base, s_test)
    _show_group_metrics(per_group, gaps, label="Baseline")

    st.markdown("### Pre-processing: SMOTE + Reweighting")
    try:
        if not IMBLEARN_AVAILABLE:
            raise RuntimeError("imblearn/SMOTE not available.")
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_train, y_train)
        weights = compute_class_weight('balanced', classes=np.unique(y_sm), y=y_sm)
        sw = np.array([weights[val] for val in y_sm])
        base2 = LogisticRegression(solver="liblinear")
        base2.fit(X_sm, y_sm, sample_weight=sw)
        y_hat_pre = (base2.predict_proba(X_test)[:,1] >= 0.5).astype(int)
        per_group_pre, gaps_pre = group_fairness_summary(y_test, y_hat_pre, s_test)
        _show_group_metrics(per_group_pre, gaps_pre, label="Pre-processing")
    except Exception as e:
        st.error(f"Pre-processing demo (SMOTE) unavailable: {e}")

    st.markdown("### In-processing: Constraint (Equalized Odds)")
    if FAIRLEARN_AVAILABLE:
        est = LogisticRegression(solver="liblinear", max_iter=1000)
        mitigator = ExponentiatedGradient(est, EqualizedOdds())
        mitigator.fit(X_train, y_train, sensitive_features=s_train)
        y_hat_in = (mitigator.predict(X_test) >= 0.5).astype(int)
        per_group_in, gaps_in = group_fairness_summary(y_test, y_hat_in, s_test)
        _show_group_metrics(per_group_in, gaps_in, label="In-processing (EO)")
    else:
        st.info("Install `fairlearn` to run the in-processing constraint demo.")

    st.markdown("### Post-processing: Group-Specific Thresholds")
    best = None
    pa_all = base.predict_proba(X_test)[:,1]
    for ta in np.linspace(0.3,0.7,21):
        for tb in np.linspace(0.3,0.7,21):
            maskA = (s_test=="A"); maskB = ~maskA
            yhatA = (pa_all[maskA]>=ta).astype(int)
            yhatB = (pa_all[maskB]>=tb).astype(int)
            yhat = np.empty_like(y_test); yhat[maskA]=yhatA; yhat[maskB]=yhatB
            _, g = group_fairness_summary(y_test, yhat, s_test)
            score = g["tpr"] + g["fpr"]
            if best is None or score < best[0]:
                best = (score, (ta,tb), yhat, g)
    if best:
        (score,(ta,tb), yhat_pp, gaps_pp) = best
        st.write(f"Selected thresholds A={ta:.2f}, B={tb:.2f}")
        per_group_pp, _ = group_fairness_summary(y_test, yhat_pp, s_test)
        _show_group_metrics(per_group_pp, gaps_pp, label="Post-processing (thresholds)")

    st.divider()
    if st.button("Download Case Report (Toy)", key="toy_report_btn"):
        md = _build_case_report_md(
            title="Toy Case — Full Intervention",
            notes=[
                "Baseline shows TPR/FPR gaps due to distribution shift and imbalance.",
                "Pre-processing (SMOTE + weights) reduces minority miss-rate.",
                "In-processing (EO) further aligns error rates.",
                "Post-processing thresholds close residual gaps."
            ]
        )
        st.download_button("Download toy_case_report.md", md, "toy_case_report.md", "text/markdown")

def justice_translation_case():
    st.subheader("AI Translation → Indigenous Languages in Justice — Simulated Evaluation")
    st.caption("Simulate translation adequacy classification across High-Resource (HR) and Indigenous (IND) language groups, then apply interventions.")

    rng = np.random.RandomState(7)
    n_hr = st.slider("Samples (High-resource)", 600, 3000, 1200)
    n_ind = st.slider("Samples (Indigenous)", 200, 2000, 400)

    q_hr  = np.clip(rng.beta(6, 2, size=n_hr), 0, 1)       # higher quality
    q_ind = np.clip(rng.beta(3, 3.5, size=n_ind), 0, 1)    # lower & variable
    y_hr  = (q_hr  >= 0.6).astype(int)
    y_ind = (q_ind >= 0.6).astype(int)

    m_hr  = np.clip(q_hr  + rng.normal(0, 0.07, n_hr), 0, 1)
    m_ind = np.clip(q_ind - 0.08 + rng.normal(0, 0.12, n_ind), 0, 1)

    th = st.slider("Global threshold (baseline)", 0.4, 0.8, 0.6, 0.01)
    yhat_hr  = (m_hr  >= th).astype(int)
    yhat_ind = (m_ind >= th).astype(int)
    y_true = np.concatenate([y_hr, y_ind])
    y_pred = np.concatenate([yhat_hr, yhat_ind])
    s = np.array(["HR"]*n_hr + ["IND"]*n_ind)

    st.markdown("### Baseline fairness")
    per_group_b, gaps_b = group_fairness_summary(y_true, y_pred, s)
    _show_group_metrics(per_group_b, gaps_b, "Baseline")

    st.markdown("### Interventions")
    with st.expander("1) Pre-processing: Domain lexicon & data augmentation"):
        st.write("Simulate +0.06 uplift in IND model score via specialized lexicons/community glossaries/augmented parallel sentences.")
    with st.expander("2) In-processing: Group-aware loss (reduce FNR on IND)"):
        st.write("Simulate improved IND recall by nudging uncertain IND scores +0.03 around the decision boundary.")
    with st.expander("3) Post-processing: Group thresholds & Human-in-the-loop"):
        st.write("Select group thresholds and route uncertain IND cases to interpreters.")

    m_ind2 = np.clip(m_ind + 0.06, 0, 1)  # pre
    band = (m_ind2 > (th-0.05)) & (m_ind2 < (th+0.05))
    m_ind3 = np.clip(m_ind2 + 0.03*band, 0, 1)  # in

    best = None
    for th_hr in np.linspace(th-0.05, th+0.05, 11):
        for th_ind in np.linspace(th-0.10, th+0.10, 21):
            yhat_hr2 = (m_hr >= th_hr).astype(int)
            yhat_ind2 = (m_ind3 >= th_ind).astype(int)
            yhat = np.concatenate([yhat_hr2, yhat_ind2])
            _, g = group_fairness_summary(y_true, yhat, s)
            score = g["fnr"] + g["tpr"] + 0.5*g["fpr"]
            if best is None or score < best[0]:
                best = (score, (th_hr, th_ind), yhat, g)

    (score,(th_hr_sel, th_ind_sel), yhat_best, gaps_best) = best
    st.write(f"Selected thresholds → HR: **{th_hr_sel:.2f}**, IND: **{th_ind_sel:.2f}**")
    per_group_post, _ = group_fairness_summary(y_true, yhat_best, s)
    _show_group_metrics(per_group_post, gaps_best, "After Interventions")

    st.info("Operationalization: define a confidence band to route IND cases for human review; maintain community lexicons; monitor FNR/TPR parity by legal domain (criminal/civil/administrative).")

    if st.button("Download Case Report (Justice Translation)", key="ind_trans_report_btn"):
        md = _build_case_report_md(
            title="Justice Translation — Indigenous Languages (Simulated)",
            notes=[
                "Global threshold under-serves IND (higher FNR).",
                "Lexicons + augmentation uplift IND scores.",
                "Group-aware loss improves recall near boundary.",
                "Group thresholds + HIL close residual parity gaps."
            ]
        )
        st.download_button("Download translation_case_report.md", md, "translation_case_report.md", "text/markdown")

# ======================================================================
# NEW: Sources & Snippets
# ======================================================================

def sources_and_snippets():
    st.header("📖 Sources & Ready-to-Use Snippets")
    st.markdown("""
**Core papers/frameworks**
- Feldman et al. (2015) — *Certifying & Removing Disparate Impact*  
- Hardt et al. (2016) — *Equality of Opportunity in Supervised Learning*  
- Agarwal et al. (2018) — *Reductions Approach to Fair Classification*  
- Kusner et al. (2017) — *Counterfactual Fairness*  
- Pleiss et al. (2017) — *On Fairness and Calibration*  
- Buolamwini & Gebru (2018) — *Gender Shades*
""")
    st.subheader("Snippets")
    st.code("""
# Equalized Odds with fairlearn
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression

mitigator = ExponentiatedGradient(LogisticRegression(solver="liblinear"),
                                  EqualizedOdds())
mitigator.fit(X_train, y_train, sensitive_features=s_train)
y_pred = mitigator.predict(X_test)
    """, language="python")
    st.code("""
# Group-specific thresholds
def group_threshold_predict(probas, sens, th_map):
    probas = np.asarray(probas)
    yhat = np.zeros_like(probas, dtype=int)
    for g, th in th_map.items():
        mask = (sens==g)
        yhat[mask] = (probas[mask] >= th).astype(int)
    return yhat
    """, language="python")

# ======================================================================
# NEW: Internal Doc Generator
# ======================================================================

def generate_full_intervention_doc():
    st.header("📝 Generate Internal Document — Full Intervention Process")
    st.write("Summarize **Pre → In → Post** steps with your selections/notes.")

    pre_notes  = st.text_area("Notes: Pre-processing choices", "")
    in_notes   = st.text_area("Notes: In-processing choices", "")
    post_notes = st.text_area("Notes: Post-processing choices", "")
    eval_notes = st.text_area("Notes: Evaluation & Monitoring", "")

    if st.button("Generate Internal Doc"):
        md = "# Fairness Intervention — Internal Guide\n\n"
        md += "## Pre-processing\n" + (pre_notes or "- (none)\n") + "\n"
        md += "## In-processing\n" + (in_notes or "- (none)\n") + "\n"
        md += "## Post-processing\n" + (post_notes or "- (none)\n") + "\n"
        md += "## Evaluation & Monitoring\n" + (eval_notes or "- (none)\n") + "\n"
        st.download_button("Download internal_intervention.md", md,
                           "internal_intervention.md", "text/markdown")

# ======================================================================
# ROUTER / NAVIGATION
# ======================================================================

def main_navigation():
    st.sidebar.markdown("## 📚 Playbooks & Demos")
    page = st.sidebar.selectbox(
        "Go to:",
        [
            "Pre-processing Fairness Toolkit",
            "Causal Fairness Toolkit",
            "Bias Mitigation Techniques Toolkit",
            "In-processing Fairness Toolkit",
            "Case Studies",
            "Sources & Snippets",
            "Generate Full Intervention Doc",
            "Simulations (Calibration/Threshold/Rejection)"
        ]
    )
    if page == "Pre-processing Fairness Toolkit":
        preprocessing_fairness_toolkit()
    elif page == "Causal Fairness Toolkit":
        causal_fairness_toolkit()
    elif page == "Bias Mitigation Techniques Toolkit":
        bias_mitigation_techniques_toolkit()
    elif page == "In-processing Fairness Toolkit":
        inprocessing_fairness_toolkit()
    elif page == "Case Studies":
        case_studies()
    elif page == "Sources & Snippets":
        sources_and_snippets()
    elif page == "Simulations (Calibration/Threshold/Rejection)":
        st.header("📈 Simulations")
        with st.expander("Calibration"):
            run_calibration_simulation()
        with st.expander("Threshold Optimization"):
            run_threshold_simulation()
        with st.expander("Human-in-the-Loop Rejection"):
            run_rejection_simulation()
    else:
        generate_full_intervention_doc()

# ======================================================================
# ENTRY
# ======================================================================

if __name__ == "__main__":
    main_navigation()
