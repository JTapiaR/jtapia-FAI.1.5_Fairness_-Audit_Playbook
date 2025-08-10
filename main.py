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
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns

# --- Configuración de la Página ---
st.set_page_config(
    page_title="AI Fairness Playbooks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    st.write("See how raw model scores (blue line) can be miscalibrated and how techniques like **Platt Scaling** (logistic) or **Isotonic Regression** adjust them to better align with reality (perfect diagonal line).")

    np.random.seed(0)
    # Generate poorly calibrated model scores
    raw_scores = np.sort(np.random.rand(100))
    true_probs = 1 / (1 + np.exp(-(raw_scores * 4 - 2)))  # Sigmoid curve to simulate reality

    # Platt Scaling
    platt = LogisticRegression()
    platt.fit(raw_scores.reshape(-1, 1), (true_probs > 0.5).astype(int))
    calibrated_platt = platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    # Isotonic Regression
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(raw_scores, true_probs)
    calibrated_isotonic = isotonic.predict(raw_scores)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(raw_scores, true_probs, 'b-', label='Original Scores (Miscalibrated)')
    ax.plot(raw_scores, calibrated_platt, 'g:', label='Platt Scaling Calibrated')
    ax.plot(raw_scores, calibrated_isotonic, 'r-.', label='Isotonic Regression Calibrated')
    ax.set_title("Calibration Techniques Comparison")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Positive Fraction")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.info("The goal is for the score lines to be as close as possible to the dashed diagonal line, which represents perfect calibration.")


def run_rejection_simulation():
    st.markdown("#### Classification with Rejection Simulation")
    st.write("Set a confidence threshold. Predictions with very high or very low confidence (probability) are automated. Those in the 'uncertainty zone' are rejected and sent to a human for review.")

    np.random.seed(1)
    scores = np.random.beta(2, 2, 200)  # Probabilities between 0 and 1

    low_thresh = st.slider("Lower Confidence Threshold", 0.0, 0.5, 0.25)
    high_thresh = st.slider("Upper Confidence Threshold", 0.5, 1.0, 0.75)

    automated_low = scores[scores <= low_thresh]
    automated_high = scores[scores >= high_thresh]
    rejected = scores[(scores > low_thresh) & (scores < high_thresh)]

    fig, ax = plt.subplots()
    ax.hist(automated_low, bins=10, range=(0,1), color='green', alpha=0.7, label=f'Automatic Decision (Low Prob, n={len(automated_low)})')
    ax.hist(rejected, bins=10, range=(0,1), color='orange', alpha=0.7, label=f'Rejected to Human (n={len(rejected)})')
    ax.hist(automated_high, bins=10, range=(0,1), color='blue', alpha=0.7, label=f'Automatic Decision (High Prob, n={len(automated_high)})')
    ax.set_title("Decision Distribution")
    ax.set_xlabel("Model Probability Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    coverage = (len(automated_low) + len(automated_high)) / len(scores)
    st.metric("Coverage Rate (Automation)", f"{coverage:.1%}")
    st.info("Adjust thresholds to see how the number of automated cases vs. those requiring human review changes. A wider rejection range increases fairness in difficult cases at the expense of lower automation.")


def run_matching_simulation():
    st.markdown("#### Matching Simulation")
    st.write("Compare two groups to estimate an effect. Matching seeks 'similar' individuals in both groups to make a fairer comparison.")
    np.random.seed(0)
    x_treat = np.random.normal(5, 1.5, 50)
    y_treat = 2 * x_treat + 5 + np.random.normal(0, 2, 50)
    x_control = np.random.normal(3.5, 1.5, 50)
    y_control = 2 * x_control + np.random.normal(0, 2, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.scatter(x_treat, y_treat, c='red', label='Treatment', alpha=0.7)
    ax1.scatter(x_control, y_control, c='blue', label='Control', alpha=0.7)
    ax1.set_title("Before Matching")
    ax1.set_xlabel("Feature (e.g., Prior Spending)")
    ax1.set_ylabel("Outcome (e.g., Purchases)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    matched_indices = [np.argmin(np.abs(x_c - x_treat)) for x_c in x_control]
    x_treat_matched = x_treat[matched_indices]
    y_treat_matched = y_treat[matched_indices]

    ax2.scatter(x_treat_matched, y_treat_matched, c='red', label='Treatment (Matched)', alpha=0.7)
    ax2.scatter(x_control, y_control, c='blue', label='Control', alpha=0.7)
    ax2.set_title("After Matching")
    ax2.set_xlabel("Feature (e.g., Prior Spending)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)
    st.info("On the left, groups are not directly comparable. On the right, we've selected a subset of the treatment group that is 'similar' to the control group, allowing a fairer estimate of the treatment effect.")

def run_rd_simulation():
    st.markdown("#### Regression Discontinuity (RD) Simulation")
    st.write("RD is used when a treatment is assigned based on a cutoff (e.g., a minimum score for a scholarship). Individuals just above and below the cutoff are compared to estimate the treatment effect.")

    np.random.seed(42)
    cutoff = st.slider("Cutoff Value", 40, 60, 50, key="rd_cutoff")

    x = np.linspace(0, 100, 200)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 200)
    treatment_effect = 15
    y[x >= cutoff] += treatment_effect

    fig, ax = plt.subplots()
    ax.scatter(x[x < cutoff], y[x < cutoff], c='blue', label='Control (No treatment)')
    ax.scatter(x[x >= cutoff], y[x >= cutoff], c='red', label='Treatment')
    ax.axvline(x=cutoff, color='gray', linestyle='--', label=f'Cutoff at {cutoff}')
    ax.set_title("Treatment Effect at the Cutoff")
    ax.set_xlabel("Assignment Variable (e.g., Exam Score)")
    ax.set_ylabel("Outcome (e.g., Future Income)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.info(f"The 'jump' or discontinuity in the outcome line at the cutoff point ({cutoff}) is an estimate of the causal treatment effect. Here, the effect is approximately **{treatment_effect}** units.")


def run_did_simulation():
    st.markdown("#### Difference-in-Differences (DiD) Simulation")
    st.write("DiD compares the change in outcomes over time between a group that receives a treatment and one that does not. It assumes both groups would have followed 'parallel trends' without the treatment.")

    time = ['Before', 'After']
    control_outcomes = [20, 25] 
    treat_outcomes = [15, 28]

    fig, ax = plt.subplots()
    ax.plot(time, control_outcomes, 'bo-', label='Control Group (Observed)')
    ax.plot(time, treat_outcomes, 'ro-', label='Treatment Group (Observed)')

    counterfactual = [treat_outcomes[0], treat_outcomes[0] + (control_outcomes[1] - control_outcomes[0])]
    ax.plot(time, counterfactual, 'r--', label='Treatment Group (Counterfactual)')

    ax.set_title("Treatment Effect Estimation with DiD")
    ax.set_ylabel("Outcome")
    ax.set_ylim(10, 35)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    effect = treat_outcomes[1] - counterfactual[1]
    st.info(f"The dashed line shows the 'parallel trend' the treatment group would have followed without intervention. The vertical difference between the solid red line and the dashed line in the 'After' period is the treatment effect, estimated at **{effect}** units.")

def bias_mitigation_techniques_toolkit():
    """New toolkit for bias mitigation techniques"""
    st.header("🔧 Bias Mitigation Techniques Toolkit")
    
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **Bias Mitigation Techniques** are practical methods to balance your dataset before training. 
        Think of them as different ways to ensure all voices are heard equally in your data - 
        some by amplifying quiet voices (oversampling), others by moderating loud ones (undersampling), 
        and some by creating synthetic but realistic examples (SMOTE).
        """)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resampling Techniques", "Reweighting", "Data Augmentation", 
        "Fair Clustering", "SMOTE", "Interactive Comparison"
    ])

    # TAB 1: Resampling Techniques
    with tab1:
        st.subheader("Resampling Techniques")
        
        with st.expander("💡 Interactive Oversampling Simulation"):
            st.write("See how oversampling balances an imbalanced dataset")
            
            # Generate sample data
            np.random.seed(42)
            majority_size = st.slider("Majority group size", 100, 1000, 800, key="maj_size")
            minority_size = st.slider("Minority group size", 50, 500, 200, key="min_size")
            
            # Original distribution
            original_ratio = minority_size / (majority_size + minority_size)
            
            # After oversampling
            target_ratio = st.radio("Target balance", ["50-50", "60-40", "70-30"], key="target_balance")
            if target_ratio == "50-50":
                new_minority_size = majority_size
            elif target_ratio == "60-40":
                new_minority_size = int(majority_size * 0.67)
            else:  # 70-30
                new_minority_size = int(majority_size * 0.43)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original
            ax1.bar(['Majority', 'Minority'], [majority_size, minority_size], 
                   color=['lightblue', 'lightcoral'])
            ax1.set_title(f"Original Dataset\nRatio: {original_ratio:.1%} minority")
            ax1.set_ylabel("Sample Count")
            
            # After oversampling
            ax2.bar(['Majority', 'Minority'], [majority_size, new_minority_size], 
                   color=['lightblue', 'lightcoral'])
            ax2.set_title(f"After Oversampling\nRatio: {new_minority_size/(majority_size+new_minority_size):.1%} minority")
            ax2.set_ylabel("Sample Count")
            
            st.pyplot(fig)
            
            # Show replication factor
            replication_factor = new_minority_size / minority_size
            st.info(f"Replication factor: {replication_factor:.1f}x (each minority sample duplicated {replication_factor:.1f} times)")

        st.code("""
# Oversampling implementation
from sklearn.utils import resample

def apply_oversampling(data, target_column, minority_class):
    majority = data[data[target_column] != minority_class]
    minority = data[data[target_column] == minority_class]
    
    # Oversample minority class
    minority_upsampled = resample(minority, 
                                 replace=True,
                                 n_samples=len(majority),
                                 random_state=42)
    
    # Combine majority and upsampled minority
    return pd.concat([majority, minority_upsampled])
        """, language="python")

        st.text_area("Apply to your case: Which resampling strategy fits your problem?", 
                     placeholder="Example: Our hiring dataset has 80% male candidates. We'll use oversampling to reach 60-40 balance, avoiding perfect balance to maintain some realism.", 
                     key="resample_plan")

    # TAB 2: Reweighting
    with tab2:
        st.subheader("Sample Reweighting")
        
        with st.expander("💡 Interactive Weighting Simulation"):
            st.write("See how different weighting strategies affect model training focus")
            
            # Sample fraud detection scenario
            legitimate_pct = st.slider("Percentage of legitimate transactions", 80, 99, 95, key="legit_pct")
            fraud_pct = 100 - legitimate_pct
            
            # Calculate inverse weights
            weight_legit = 1 / (legitimate_pct / 100)
            weight_fraud = 1 / (fraud_pct / 100)
            
            # Visualization
            fig, ax = plt.subplots()
            categories = ['Legitimate', 'Fraudulent']
            percentages = [legitimate_pct, fraud_pct]
            weights = [weight_legit, weight_fraud]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, percentages, width, label='Data Percentage', color='lightblue')
            ax.bar(x + width/2, weights, width, label='Assigned Weight', color='lightcoral')
            
            ax.set_ylabel('Value')
            ax.set_title('Data Distribution vs. Assigned Weights')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            
            # Add value labels on bars
            for i, (pct, wt) in enumerate(zip(percentages, weights)):
                ax.text(i - width/2, pct + 1, f'{pct}%', ha='center')
                ax.text(i + width/2, wt + 0.1, f'{wt:.1f}', ha='center')
            
            st.pyplot(fig)
            st.info(f"Fraudulent cases get {weight_fraud:.1f}x more weight, making the model pay {weight_fraud:.1f}x more attention to errors in fraud detection.")

        st.code("""
# Reweighting implementation
from sklearn.utils.class_weight import compute_class_weight

def apply_reweighting(X, y):
    # Calculate balanced weights automatically
    weights = compute_class_weight('balanced', 
                                  classes=np.unique(y), 
                                  y=y)
    
    # Create sample weights array
    sample_weights = np.array([weights[label] for label in y])
    
    return sample_weights

# Usage in training
sample_weights = apply_reweighting(X_train, y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
        """, language="python")

        st.text_area("Apply to your case: What imbalance will you address with reweighting?", 
                     placeholder="Example: Our medical diagnosis dataset has only 3% positive cases. We'll use inverse frequency weighting to ensure the model doesn't ignore rare diseases.", 
                     key="reweight_plan")

    # TAB 3: Data Augmentation
    with tab3:
        st.subheader("Data Augmentation for Fairness")
        
        with st.expander("💡 Interactive Augmentation Visualization"):
            st.write("See how data augmentation can expand underrepresented groups")
            
            augmentation_factor = st.slider("Augmentation factor for minority group", 1, 10, 5, key="aug_factor")
            
            # Original small group
            np.random.seed(1)
            original_samples = 20
            augmented_samples = original_samples * augmentation_factor
            
            # Simulate feature distributions
            original_data = np.random.multivariate_normal([2, 3], [[1, 0.5], [0.5, 1]], original_samples)
            
            # Simulate augmented data (with slight variations)
            augmented_data = []
            for _ in range(augmented_samples - original_samples):
                # Add noise to simulate realistic augmentation
                base_sample = original_data[np.random.randint(0, original_samples)]
                augmented_sample = base_sample + np.random.normal(0, 0.3, 2)
                augmented_data.append(augmented_sample)
            
            if augmented_data:
                augmented_data = np.array(augmented_data)
                combined_data = np.vstack([original_data, augmented_data])
            else:
                combined_data = original_data
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original
            ax1.scatter(original_data[:, 0], original_data[:, 1], 
                       c='blue', alpha=0.7, s=50)
            ax1.set_title(f"Original Minority Group\n(n={original_samples})")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")
            ax1.grid(True, alpha=0.3)
            
            # Augmented
            ax2.scatter(original_data[:, 0], original_data[:, 1], 
                       c='blue', alpha=0.7, s=50, label='Original')
            if len(augmented_data) > 0:
                ax2.scatter(augmented_data[:, 0], augmented_data[:, 1], 
                           c='red', alpha=0.5, s=30, marker='x', label='Augmented')
            ax2.set_title(f"After Augmentation\n(n={len(combined_data)})")
            ax2.set_xlabel("Feature 1")
            ax2.set_ylabel("Feature 2")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.info(f"Data augmentation increased the minority group from {original_samples} to {len(combined_data)} samples, helping the model learn their patterns better.")

        st.code("""
# Data Augmentation for Images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_image_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=15,           # Rotate ±15 degrees
        brightness_range=[0.8, 1.2], # Adjust brightness
        zoom_range=0.1,              # Zoom in/out
        horizontal_flip=True,        # Mirror horizontally
        width_shift_range=0.1,       # Shift horizontally
        height_shift_range=0.1       # Shift vertically
    )
    return datagen

# Generate augmented data
def augment_minority_group(images, labels, minority_label, target_count):
    minority_mask = labels == minority_label
    minority_images = images[minority_mask]
    
    datagen = setup_image_augmentation()
    augmented_images = []
    
    current_count = len(minority_images)
    needed_count = target_count - current_count
    
    for batch in datagen.flow(minority_images, batch_size=32):
        augmented_images.extend(batch)
        if len(augmented_images) >= needed_count:
            break
    
    return np.array(augmented_images[:needed_count])
        """, language="python")

        st.text_area("Apply to your case: What augmentation strategy would work for your data type?", 
                     placeholder="Example: For our facial recognition system, we'll apply rotation, lighting changes, and background substitution to increase representation of underrepresented ethnic groups.", 
                     key="augment_plan")

    # TAB 4: Fair Clustering
    with tab4:
        st.subheader("Fair Clustering Techniques")
        
        with st.expander("💡 Interactive Fair Clustering Demo"):
            st.write("Compare traditional clustering vs. fair clustering that ensures balanced representation")
            
            # Generate sample data with bias
            np.random.seed(10)
            group_a = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 80)
            group_b = np.random.multivariate_normal([6, 6], [[1, 0.3], [0.3, 1]], 20)
            
            all_data = np.vstack([group_a, group_b])
            group_labels = ['A'] * 80 + ['B'] * 20
            
            n_clusters = st.slider("Number of clusters", 2, 5, 3, key="fair_clusters")
            
            # Traditional clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            traditional_clusters = kmeans.fit_predict(all_data)
            
            # Simulate fair clustering (simplified version)
            fair_clusters = traditional_clusters.copy()
            for cluster_id in range(n_clusters):
                cluster_mask = traditional_clusters == cluster_id
                cluster_groups = np.array(group_labels)[cluster_mask]
                
                # If cluster is very imbalanced, reassign some points
                group_a_pct = np.mean(cluster_groups == 'A')
                if group_a_pct > 0.9:  # Too many A's
                    # Find B points to reassign to this cluster
                    b_points = np.where((np.array(group_labels) == 'B') & (traditional_clusters != cluster_id))[0]
                    if len(b_points) > 0:
                        fair_clusters[b_points[:2]] = cluster_id
                elif group_a_pct < 0.1:  # Too many B's
                    # Find A points to reassign
                    a_points = np.where((np.array(group_labels) == 'A') & (traditional_clusters != cluster_id))[0]
                    if len(a_points) > 0:
                        fair_clusters[a_points[:2]] = cluster_id
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Traditional clustering
            colors_trad = ['red', 'blue', 'green', 'purple', 'orange']
            for i in range(n_clusters):
                cluster_points = all_data[traditional_clusters == i]
                if len(cluster_points) > 0:
                    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                               c=colors_trad[i], alpha=0.7, label=f'Cluster {i}')
            ax1.set_title("Traditional K-Means Clustering")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Fair clustering
            for i in range(n_clusters):
                cluster_points = all_data[fair_clusters == i]
                if len(cluster_points) > 0:
                    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                               c=colors_trad[i], alpha=0.7, label=f'Cluster {i}')
            ax2.set_title("Fair Clustering (Balanced)")
            ax2.set_xlabel("Feature 1")
            ax2.set_ylabel("Feature 2")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Show balance metrics
            st.write("#### Cluster Balance Analysis")
            for i in range(n_clusters):
                trad_mask = traditional_clusters == i
                fair_mask = fair_clusters == i
                
                if np.sum(trad_mask) > 0 and np.sum(fair_mask) > 0:
                    trad_a_pct = np.mean(np.array(group_labels)[trad_mask] == 'A')
                    fair_a_pct = np.mean(np.array(group_labels)[fair_mask] == 'A')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"Cluster {i} - Traditional", f"{trad_a_pct:.1%} Group A")
                    with col2:
                        st.metric(f"Cluster {i} - Fair", f"{fair_a_pct:.1%} Group A", 
                                 delta=f"{fair_a_pct - trad_a_pct:+.1%}")

        st.code("""
# Fair Clustering Implementation
def fair_clustering(X, sensitive_features, n_clusters, balance_threshold=0.3):
    # Initial clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    initial_clusters = kmeans.fit_predict(X)
    
    # Check and adjust for fairness
    for cluster_id in range(n_clusters):
        cluster_mask = initial_clusters == cluster_id
        cluster_sensitive = sensitive_features[cluster_mask]
        
        # Calculate group proportions in cluster
        unique_groups, counts = np.unique(cluster_sensitive, return_counts=True)
        proportions = counts / len(cluster_sensitive)
        
        # If cluster is imbalanced beyond threshold
        if np.max(proportions) > (1 - balance_threshold):
            # Reassign some points to achieve better balance
            initial_clusters = rebalance_cluster(initial_clusters, cluster_id, 
                                               sensitive_features, balance_threshold)
    
    return initial_clusters

def rebalance_cluster(clusters, cluster_id, sensitive_features, threshold):
    # Implementation of rebalancing logic
    # (This would involve finding appropriate points to reassign)
    return clusters
        """, language="python")

    # TAB 5: SMOTE
    with tab5:
        st.subheader("SMOTE (Synthetic Minority Oversampling Technique)")
        
        with st.expander("💡 Interactive SMOTE Visualization"):
            st.write("See how SMOTE creates synthetic samples by interpolating between existing minority samples")
            
            # Generate sample minority data
            np.random.seed(5)
            minority_samples = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 1]], 15)
            majority_samples = np.random.multivariate_normal([1, 1], [[1, 0.2], [0.2, 1]], 85)
            
            # Simulate SMOTE process
            k_neighbors = st.slider("Number of neighbors for SMOTE", 1, 5, 3, key="smote_k")
            synthetic_count = st.slider("Synthetic samples to generate", 10, 50, 25, key="smote_count")
            
            # Create synthetic samples (simplified SMOTE simulation)
            synthetic_samples = []
            for _ in range(synthetic_count):
                # Pick a random minority sample
                base_idx = np.random.randint(0, len(minority_samples))
                base_sample = minority_samples[base_idx]
                
                # Find k nearest neighbors
                distances = np.linalg.norm(minority_samples - base_sample, axis=1)
                neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
                
                # Pick a random neighbor and interpolate
                neighbor_idx = np.random.choice(neighbor_indices)
                neighbor_sample = minority_samples[neighbor_idx]
                
                # Linear interpolation
                alpha = np.random.random()
                synthetic_sample = base_sample + alpha * (neighbor_sample - base_sample)
                synthetic_samples.append(synthetic_sample)
            
            synthetic_samples = np.array(synthetic_samples)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot original data
            ax.scatter(majority_samples[:, 0], majority_samples[:, 1], 
                      c='lightblue', alpha=0.6, s=50, label=f'Majority (n={len(majority_samples)})')
            ax.scatter(minority_samples[:, 0], minority_samples[:, 1], 
                      c='red', alpha=0.8, s=80, label=f'Minority Original (n={len(minority_samples)})')
            
            # Plot synthetic samples
            ax.scatter(synthetic_samples[:, 0], synthetic_samples[:, 1], 
                      c='orange', alpha=0.7, s=60, marker='x', 
                      label=f'Synthetic (n={len(synthetic_samples)})')
            
            # Draw lines between some original points to show interpolation
            for i in range(min(5, len(synthetic_samples))):
                # Find closest original points to show interpolation
                distances_to_orig = np.linalg.norm(minority_samples - synthetic_samples[i], axis=1)
                closest_indices = np.argsort(distances_to_orig)[:2]
                
                ax.plot([minority_samples[closest_indices[0], 0], synthetic_samples[i, 0]], 
                       [minority_samples[closest_indices[0], 1], synthetic_samples[i, 1]], 
                       'gray', alpha=0.3, linestyle='--')
                ax.plot([minority_samples[closest_indices[1], 0], synthetic_samples[i, 0]], 
                       [minority_samples[closest_indices[1], 1], synthetic_samples[i, 1]], 
                       'gray', alpha=0.3, linestyle='--')
            
            ax.set_title("SMOTE: Synthetic Sample Generation")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.info("Synthetic samples (orange X's) are created by interpolating between original minority samples (red dots). The dashed lines show some interpolation paths.")

        st.code("""
# SMOTE Implementation
from imblearn.over_sampling import SMOTE

def apply_smote(X, y, sampling_strategy='auto'):
    # Initialize SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, 
                  random_state=42,
                  k_neighbors=5)
    
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Show before/after distribution
    print("Original distribution:", Counter(y))
    print("After SMOTE:", Counter(y_resampled))
    
    return X_resampled, y_resampled

# Usage example
X_balanced, y_balanced = apply_smote(X_train, y_train)

# For multi-class with specific strategy
smote_strategy = {0: 100, 1: 200}  # Specific target counts
X_balanced, y_balanced = apply_smote(X_train, y_train, smote_strategy)
        """, language="python")

    # TAB 6: Interactive Comparison
    with tab6:
        st.subheader("Technique Comparison Tool")
        
        st.write("Compare different bias mitigation techniques on a simulated dataset")
        
        # Dataset parameters
        col1, col2 = st.columns(2)
        with col1:
            majority_size = st.number_input("Majority group size", 100, 2000, 800, key="comp_maj")
            minority_size = st.number_input("Minority group size", 20, 500, 150, key="comp_min")
        with col2:
            target_balance = st.selectbox("Target balance", ["50-50", "60-40", "70-30"], key="comp_balance")
            techniques = st.multiselect("Select techniques to compare", 
                                       ["Oversampling", "Undersampling", "Reweighting", "SMOTE"], 
                                       default=["Oversampling", "SMOTE"], key="comp_techniques")
        
        if techniques:
            results = {}
            
            # Calculate results for each technique
            if "Oversampling" in techniques:
                if target_balance == "50-50":
                    new_minority = majority_size
                elif target_balance == "60-40":
                    new_minority = int(majority_size * 0.67)
                else:  # 70-30
                    new_minority = int(majority_size * 0.43)
                
                results["Oversampling"] = {
                    "Final Majority": majority_size,
                    "Final Minority": new_minority,
                    "Total Samples": majority_size + new_minority,
                    "Information Loss": "None",
                    "Overfitting Risk": "Medium" if new_minority > minority_size * 3 else "Low"
                }
            
            if "Undersampling" in techniques:
                new_majority = minority_size
                results["Undersampling"] = {
                    "Final Majority": new_majority,
                    "Final Minority": minority_size,
                    "Total Samples": new_majority + minority_size,
                    "Information Loss": f"{((majority_size - new_majority) / majority_size * 100):.1f}%",
                    "Overfitting Risk": "Low"
                }
            
            if "Reweighting" in techniques:
                weight_ratio = majority_size / minority_size
                results["Reweighting"] = {
                    "Final Majority": majority_size,
                    "Final Minority": minority_size,
                    "Total Samples": majority_size + minority_size,
                    "Minority Weight": f"{weight_ratio:.1f}x",
                    "Information Loss": "None"
                }
            
            if "SMOTE" in techniques:
                synthetic_needed = max(0, majority_size - minority_size)
                results["SMOTE"] = {
                    "Final Majority": majority_size,
                    "Final Minority": minority_size + synthetic_needed,
                    "Total Samples": majority_size + minority_size + synthetic_needed,
                    "Synthetic Samples": synthetic_needed,
                    "Information Loss": "None"
                }
            
            # Display comparison table
            if results:
                df_comparison = pd.DataFrame(results).T
                st.dataframe(df_comparison)
                
                # Recommendations
                st.subheader("Recommendations")
                if majority_size > 5 * minority_size:
                    st.warning("⚠️ High imbalance detected. Consider combining techniques (e.g., SMOTE + slight undersampling)")
                
                if "Undersampling" in results and int(results["Undersampling"]["Information Loss"].replace('%', '')) > 50:
                    st.warning("⚠️ Undersampling would lose >50% of data. Consider oversampling or SMOTE instead")
                
                if synthetic_needed > minority_size * 5:
                    st.warning("⚠️ SMOTE would create many synthetic samples. Risk of unrealistic data. Consider combining with other techniques")

        st.text_area("Document your technique selection and rationale:", 
                     placeholder="Example: Given our 85-15 imbalance and small minority group (n=150), we'll use SMOTE to generate realistic synthetic samples, combined with slight undersampling of the majority to reach 70-30 balance.", 
                     key="comparison_conclusion")
    # --- Combined Pipeline Section ---
    st.markdown("---")
    st.subheader("🔗 Complete Bias Mitigation Pipeline")
    
    st.code("""
# Complete bias mitigation pipeline combining multiple techniques
def complete_bias_mitigation_pipeline(X, y, sensitive_attr, strategy='balanced'):
    \"\"\"
    Complete pipeline for bias mitigation using multiple techniques
    \"\"\"
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Step 1: Analyze current bias
    print("=== BIAS ANALYSIS ===")
    analyze_bias(X, y, sensitive_attr)
    
    # Step 2: Apply SMOTE for synthetic generation
    print("=== APPLYING SMOTE ===")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"Original distribution: {Counter(y)}")
    print(f"After SMOTE: {Counter(y_balanced)}")
    
    # Step 3: Calculate and apply sample weights for remaining imbalance
    print("=== CALCULATING WEIGHTS ===")
    weights = compute_class_weight('balanced', 
                                  classes=np.unique(y_balanced), 
                                  y=y_balanced)
    sample_weights = np.array([weights[label] for label in y_balanced])
    
    # Step 4: Train model with weights
    print("=== TRAINING MODEL ===")
    model = LogisticRegression(random_state=42)
    model.fit(X_balanced, y_balanced, sample_weight=sample_weights)
    
    # Step 5: Validate fairness
    print("=== FAIRNESS VALIDATION ===")
    X_test_bal, _, y_test_bal, _ = train_test_split(X_balanced, y_balanced, 
                                                   test_size=0.2, random_state=42)
    validate_fairness(model, X_test_bal, y_test_bal, sensitive_attr)
    
    return model, X_balanced, y_balanced

def analyze_bias(X, y, sensitive_attr):
    \"\"\"Analyze current bias in the dataset\"\"\"
    unique_groups = np.unique(sensitive_attr)
    
    for group in unique_groups:
        group_mask = sensitive_attr == group
        group_positive_rate = np.mean(y[group_mask])
        print(f"Group {group}: {np.sum(group_mask)} samples, "
              f"{group_positive_rate:.2%} positive rate")

def validate_fairness(model, X_test, y_test, sensitive_attr):
    \"\"\"Validate fairness metrics after mitigation\"\"\"
    predictions = model.predict(X_test)
    
    unique_groups = np.unique(sensitive_attr)
    
    print("\\n=== FAIRNESS METRICS ===")
    tprs = []
    fprs = []
    
    for group in unique_groups:
        group_mask = sensitive_attr == group
        group_y_true = y_test[group_mask]
        group_y_pred = predictions[group_mask]
        
        # Calculate TPR and FPR
        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
        fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
        fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
        tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
        print(f"Group {group}: TPR={tpr:.3f}, FPR={fpr:.3f}")
    
    # Calculate fairness metrics
    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    
    print(f"\\nTPR Difference: {tpr_diff:.3f}")
    print(f"FPR Difference: {fpr_diff:.3f}")
    
    if tpr_diff < 0.1 and fpr_diff < 0.1:
        print("✅ Good fairness achieved!")
    else:
        print("⚠️ Significant fairness gaps remain")

# Usage example
if __name__ == "__main__":
    # Load your data
    # X, y, sensitive_attr = load_your_data()
    
    # Apply complete pipeline
    fair_model, X_fair, y_fair = complete_bias_mitigation_pipeline(X, y, sensitive_attr)
        """, language="python")
        
    st.text_area("Apply to your case: How would you adapt SMOTE for your specific data type?", 
                     placeholder="Example: For our tabular medical data, we'll use SMOTE with k=5 neighbors and focus on generating synthetic samples for rare disease cases, ensuring clinical realism by constraining feature ranges.", 
                     key="smote_plan")

    # --- Report Generation ---
    st.markdown("---")
    st.header("Generate Bias Mitigation Report")
    if st.button("Generate Bias Mitigation Report", key="gen_bias_mit_report"):
        report_data = {
            "Resampling Strategy": {
                "Selected Approach": st.session_state.get('resample_plan', 'Not completed'),
            },
            "Reweighting Strategy": {
                "Implementation Plan": st.session_state.get('reweight_plan', 'Not completed'),
            },
            "Data Augmentation": {
                "Augmentation Strategy": st.session_state.get('augment_plan', 'Not completed'),
            },
            "SMOTE Application": {
                "SMOTE Adaptation": st.session_state.get('smote_plan', 'Not completed'),
            },
            "Technique Comparison": {
                "Selection Rationale": st.session_state.get('comparison_conclusion', 'Not completed'),
            }
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
        st.download_button(
            label="Download Bias Mitigation Report",
            data=st.session_state.bias_mit_report_md,
            file_name="bias_mitigation_report.md",
            mime="text/markdown"
        )
        

#======================================================================
# --- FAIRNESS INTERVENTION PLAYBOOK ---
#======================================================================

def causal_fairness_toolkit():
    st.header("🛡️ Causal Fairness Toolkit")
    
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **Causal Analysis** goes beyond correlations to understand the *why* behind disparities.
        It’s like being a detective who not only sees two events happen together but reconstructs 
        the cause-and-effect chain that connects them. This helps us apply solutions that target 
        the root of the problem, instead of just treating the symptoms.
        """)
    
    if 'causal_report' not in st.session_state:
        st.session_state.causal_report = {}

    tab1, tab2, tab3, tab4, tab5  = st.tabs(["Identification", "Counterfactual Analysis", "Causal Diagram", "Causal Inference","Intersectionality"])

    # TAB 1: Identification
    with tab1:
        st.subheader("Framework for Identifying Discrimination Mechanisms")
        st.info("Identify the possible root causes of bias in your application.")
        
        with st.expander("Definition: Direct Discrimination"):
            st.write("Occurs when a protected attribute (such as race or gender) is explicitly used to make a decision. This is the most obvious type of bias.")
        st.text_area("1. Does the protected attribute directly influence the decision?", 
                     placeholder="Example: A hiring model explicitly assigns lower scores to female applicants.", 
                     key="causal_q1")
        
        with st.expander("Definition: Indirect Discrimination"):
            st.write("Occurs when a protected attribute affects an intermediate factor that is legitimate for the decision. The bias is transmitted through this mediating variable.")
        st.text_area("2. Does the protected attribute affect legitimate intermediate factors?", 
                     placeholder="Example: Gender can influence career breaks (for childcare), and the model penalizes these breaks, indirectly affecting women.", 
                     key="causal_q2")

        with st.expander("Definition: Proxy Discrimination"):
            st.write("Occurs when a seemingly neutral variable is so correlated with a protected attribute that it acts as a substitute (a 'proxy').")
        st.text_area("3. Do decisions rely on variables correlated with protected attributes?", 
                     placeholder="Example: In a credit model, using zip code as a predictor can be a proxy for race due to historical residential segregation.", 
                     key="causal_q3")

    # TAB 2: Counterfactual Analysis
    with tab2:
        st.subheader("Practical Counterfactual Fairness Methodology")
        st.info("Analyze, quantify, and mitigate counterfactual bias in your model.")
        with st.expander("💡 Interactive Example: Counterfactual Simulation"):
            st.write("See how changing a protected attribute can alter a model's decision, revealing causal bias.")
            base_score = 650
            base_decision = "Rejected"
            st.write(f"**Base Case:** Applicant from **Group B** with a score of **{base_score}**. Model decision: **{base_decision}**.")
            if st.button("View Counterfactual (Change to Group A)", key="cf_button"):
                cf_score = 710
                cf_decision = "Approved"
                st.info(f"**Counterfactual Scenario:** Same applicant, but from **Group A**. The model now predicts a score of **{cf_score}** and the decision is: **{cf_decision}**.")
                st.warning("**Analysis:** Changing the protected attribute altered the decision, suggesting the model has learned a problematic causal dependency.")
        
        with st.container(border=True):
            st.markdown("##### Step 1: Counterfactual Fairness Analysis")
            st.text_area("1.1 Formulate Counterfactual Queries", 
                         placeholder="Example: For a rejected loan applicant, what would the outcome be if their race were different, keeping income and credit history constant?", 
                         key="causal_q4")
            st.text_area("1.2 Identify Causal Paths (Fair vs. Unfair)", 
                         placeholder="Example: Path Race → Zip Code → Loan Decision is unfair because zip code is a proxy. Path Education Level → Income → Loan Decision is considered fair.", 
                         key="causal_q5")
            st.text_area("1.3 Measure Disparities and Document", 
                         placeholder="Example: 15% of applicants from the disadvantaged group would have been approved in the counterfactual scenario. This indicates a counterfactual fairness violation.", 
                         key="causal_q6")
        
        with st.container(border=True):
            st.markdown("##### Step 2: Specific Path Analysis")
            st.text_area("2.1 Decompose and Classify Paths", 
                         placeholder="Example: Path 1 (zip code proxy) classified as UNFAIR. Path 2 (mediated by income) classified as FAIR.", 
                         key="causal_q7")
            st.text_area("2.2 Quantify Contribution and Document", 
                         placeholder="Example: The zip code path accounts for 60% of observed disparity. Reason: Reflects historical residential segregation bias.", 
                         key="causal_q8")
        
        with st.container(border=True):
            st.markdown("##### Step 3: Intervention Design")
            st.selectbox("3.1 Select Intervention Approach", ["Data Level", "Model Level", "Post-processing"], key="causal_q9")
            st.text_area("3.2 Implement and Monitor", 
                         placeholder="Example: Applied a transformation to the zip code feature. Counterfactual disparity reduced by 50%.", 
                         key="causal_q10")

    # TAB 3: Causal Diagram
    with tab3:
        st.subheader("Initial Causal Diagram Approach")
        st.info("Sketch diagrams to visualize causal relationships and document your assumptions.")
        with st.expander("💡 Causal Diagram Simulator"):
            st.write("Build a simple causal diagram by selecting relationships between variables. This helps visualize your hypotheses about how bias operates.")
            
            nodes = ["Gender", "Education", "Income", "Loan_Decision"]
            possible_relations = [
                ("Gender", "Education"), ("Gender", "Income"),
                ("Education", "Income"), ("Income", "Loan_Decision"),
                ("Education", "Loan_Decision"), ("Gender", "Loan_Decision")
            ]
            
            st.multiselect(
                "Select causal relationships (Cause → Effect):",
                options=[f"{cause} → {effect}" for cause, effect in possible_relations],
                key="causal_q11_relations"
            )
            
            if st.session_state.causal_q11_relations:
                dot_string = "digraph { rankdir=LR; "
                for rel in st.session_state.causal_q11_relations:
                    cause, effect = rel.split(" → ")
                    dot_string += f'"{cause}" -> "{effect}"; '
                dot_string += "}"
                st.graphviz_chart(dot_string)

        st.markdown("""
        **Annotation Conventions:**
        - **Nodes (variables):** Protected Attributes, Features, Outcomes.
        - **Causal Arrows (→):** Assumed causal relationship.
        - **Correlation Arrows (<-->):** Correlation without direct known causality.
        - **Uncertainty (?):** Hypothetical or weak causal relationship.
        - **Problematic Path (!):** Path considered a source of inequity.
        """)
        st.text_area("Assumptions and Path Documentation", 
                     placeholder="Path (!): Race -> Income Level -> Decision.\nAssumption: Historical income disparities linked to race affect lending capacity.", 
                     height=200, key="causal_q11")

    # TAB 4: Causal Inference
    with tab4:
        st.subheader("Causal Inference with Limited Data")
        st.info("Practical methods for estimating causal effects when data is imperfect.")
        
        with st.expander("🔍 Definition: Matching"):
            st.write("Compare individuals from a 'treatment' group with very similar individuals from a 'control' group. By comparing statistical 'twins', the treatment effect is isolated. In fairness, the 'treatment' may be belonging to a demographic group.")
        with st.expander("💡 Interactive Example: Matching Simulation"):
            run_matching_simulation()

        with st.expander("🔍 Definition: Instrumental Variables (IV)"):
            st.write("Use an 'instrument' variable that affects the treatment but not the outcome directly to untangle correlation from causation.")
            st.graphviz_chart("""
            digraph {
                rankdir=LR;
                Z [label="Instrument (Z)"];
                A [label="Protected Attribute (A)"];
                Y [label="Outcome (Y)"];
                U [label="Unobserved Confounder (U)", style=dashed];
                Z -> A;
                A -> Y;
                U -> A [style=dashed];
                U -> Y [style=dashed];
            }
            """)
            st.write("**Example:** To measure the causal effect of education (A) on income (Y), proximity to a university (Z) can be used as an instrument. Proximity affects education but not income directly (except through education).")

        with st.expander("🔍 Definition: Regression Discontinuity (RD)"):
            st.write("Takes advantage of a threshold or cutoff in treatment assignment. By comparing those just above and below the cutoff, the treatment effect can be estimated, assuming these individuals are otherwise very similar.")
        with st.expander("💡 Interactive Example: RD Simulation"):
            run_rd_simulation()

        with st.expander("🔍 Definition: Difference-in-Differences (DiD)"):
            st.write("Compares change in outcomes over time between a treatment group and a control group. The 'difference in differences' between groups before and after treatment estimates the causal effect.")
        with st.expander("💡 Interactive Example: DiD Simulation"):
            run_did_simulation()
    # TAB 5: Intersectionality
    with tab5:
        st.subheader("Applying an Intersectional Perspective to Causal Analysis")
        with st.expander("🔍 Friendly Definition"):
            st.write("Intersectionality in causal analysis means recognizing that **the causes of bias are not the same for everyone**. For example, the reason a model is unfair to Black women may differ from why it is unfair to Black men or white women. We must model how the combination of identities creates unique causal pathways of discrimination.")
        
        with st.expander("💡 Interactive Example: Intersectional Causal Diagram"):
            st.write("See how a causal diagram becomes more complex and accurate when an intersectional node is considered.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Simplistic Causal Model**")
                st.graphviz_chart("""
                digraph {
                    rankdir=LR;
                    Gender -> "Years of Experience";
                    Race -> "Type of Education";
                    "Years of Experience" -> "Decision";
                    "Type of Education" -> "Decision";
                }
                """)
            with col2:
                st.write("**Intersectional Causal Model**")
                st.graphviz_chart("""
                digraph {
                    rankdir=LR;
                    subgraph cluster_0 {
                        label = "Intersectional Identity";
                        "Black Woman" [shape=box];
                    }
                    "Black Woman" -> "Access to Professional Networks" [label="Specific Path"];
                    "Access to Professional Networks" -> "Decision";
                    "Gender" -> "Years of Experience" -> "Decision";
                    "Race" -> "Type of Education" -> "Decision";
                }
                """)
            st.info("The intersectional model reveals a new causal path ('Access to Professional Networks') that specifically affects the 'Black Woman' subgroup, a factor simplistic models would ignore.")

        st.text_area(
            "Apply to your case: What unique causal paths might affect intersectional subgroups in your system?", 
            placeholder="Example: In our lending system, the interaction of 'being a woman' and 'living in a rural area' creates a unique causal path through 'lack of history with large banks', which does not affect other groups in the same way.", 
            key="causal_intersectional"
        )

    # --- Report Section ---
    st.markdown("---")
    st.header("Generate Causal Toolkit Report")
    if st.button("Generate Causal Report", key="gen_causal_report"):
        # Gather data from session_state
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

        # Format report in Markdown
        report_md = "# Causal Fairness Toolkit Report\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.causal_report_md = report_md
        st.success("✅ Report successfully generated! You can preview and download it below.")

    if 'causal_report_md' in st.session_state and st.session_state.causal_report_md:
        st.subheader("Report Preview")
        st.markdown(st.session_state.causal_report_md)
        st.download_button(
            label="Download Causal Fairness Report",
            data=st.session_state.causal_report_md,
            file_name="causal_fairness_report.md",
            mime="text/markdown"
        )

def preprocessing_fairness_toolkit():
    st.header("🧪 Pre-processing Fairness Toolkit")
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **Pre-processing** means "cleaning" the data *before* the model learns from it. 
        It’s like preparing ingredients for a recipe: if you know some ingredients are biased 
        (e.g., too salty), you adjust them before cooking to ensure the final dish is balanced.
        """)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8= st.tabs([
        "Representation Analysis", "Correlation Detection", "Label Quality", 
        "Re-weighting and Re-sampling", "Transformation", "Data Generation", 
        "🌍 Intersectionality","🔧 Bias Mitigation Techniques"
    ])

    # TAB 1: Representation Analysis
    with tab1:
        st.subheader("Multidimensional Representation Analysis")
        with st.expander("🔍 Friendly Definition"):
            st.write("This means checking whether all demographic groups are fairly represented in your data. Not just main groups (e.g., men and women), but also intersections (e.g., women of a specific ethnicity).")
        
        with st.expander("💡 Interactive Example: Representation Gap"):
            st.write("Compare the representation of two groups in your dataset with their representation in a reference population (e.g., census).")
            pop_a = 50
            pop_b = 50
            
            col1, col2 = st.columns(2)
            with col1:
                data_a = st.slider("Percentage of Group A in your data", 0, 100, 70)
            data_b = 100 - data_a
            
            df = pd.DataFrame({
                'Group': ['Group A', 'Group B'],
                'Reference Population': [pop_a, pop_b],
                'Your Data': [data_a, data_b]
            })

            with col2:
                st.write("Comparison:")
                st.dataframe(df.set_index('Group'))

            if abs(data_a - pop_a) > 10:
                st.warning(f"Significant representation gap. Group A is overrepresented in your data by {data_a - pop_a} percentage points.")
            else:
                st.success("Representation in your data is similar to the reference population.")

        st.text_area("1. Comparison with Reference Population", 
                     placeholder="E.g.: Our dataset has 70% Group A and 30% Group B, while the real population is 50/50.", 
                     key="p1")
        st.text_area("2. Intersectional Representation Analysis", 
                     placeholder="E.g.: Women from racial minorities make up only 3% of the data, though they represent 10% of the population.", 
                     key="p2")
        st.text_area("3. Representation Across Outcome Categories", 
                     placeholder="E.g.: Group A constitutes 30% of applications but only 10% of approvals.", 
                     key="p3")

    # TAB 2: Correlation Detection
    with tab2:
        st.subheader("Correlation Pattern Detection")
        with st.expander("🔍 Friendly Definition"):
            st.write("We look for seemingly neutral variables that are strongly connected to protected attributes. For example, if a postal code is strongly correlated with race, the model could use the postal code to discriminate indirectly.")
        
        with st.expander("💡 Interactive Example: Proxy Detection"):
            st.write("Visualize how a 'proxy' variable (e.g., Postal Code) can be correlated with both a Protected Attribute (e.g., Demographic Group) and the Outcome (e.g., Credit Score).")
            np.random.seed(1)
            group = np.random.randint(0, 2, 100)  # 0 or 1
            proxy = group * 20 + np.random.normal(50, 5, 100)
            outcome = proxy * 5 + np.random.normal(100, 20, 100)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(group, proxy, c=group, cmap='coolwarm', alpha=0.7)
            ax1.set_title("Protected Attribute vs. Proxy Variable")
            ax1.set_xlabel("Demographic Group (0 or 1)")
            ax1.set_ylabel("Proxy Value (e.g., Postal Code)")
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.scatter(proxy, outcome, c=group, cmap='coolwarm', alpha=0.7)
            ax2.set_title("Proxy Variable vs. Outcome")
            ax2.set_xlabel("Proxy Value (e.g., Postal Code)")
            ax2.set_ylabel("Outcome (e.g., Credit Score)")
            ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            st.info("The left plot shows that the proxy is correlated with the group. The right plot shows that the proxy predicts the outcome. Thus, the model could use the proxy to discriminate.")

        st.text_area("1. Direct Correlations (Protected Attribute ↔ Outcome)", 
                     placeholder="E.g.: In historical data, gender has a correlation of 0.3 with the hiring decision.", 
                     key="p4")
        st.text_area("2. Proxy Variable Identification (Protected Attribute ↔ Feature)", 
                     placeholder="E.g.: The 'chess club attendance' feature is highly correlated with being male.", 
                     key="p5")

    # TAB 3: Label Quality
    with tab3:
        st.subheader("Label Quality Evaluation")
        with st.expander("🔍 Friendly Definition"):
            st.write("Labels are the correct answers in your training data (e.g., 'was hired', 'did not repay the loan'). If these labels come from past human decisions that were biased, your model will learn that same bias.")
        st.text_area("1. Historical Bias in Decisions", 
                     placeholder="Example: 'Promoted' labels in our dataset come from a period when the company had biased promotion policies, so the labels themselves are a bias source.", 
                     key="p6")
        st.text_area("2. Annotator Bias", 
                     placeholder="Example: Annotator agreement analysis shows male annotators rated the same comments as 'toxic' less often than female annotators, indicating label bias.", 
                     key="p7")

    # TAB 4: Re-weighting and Re-sampling
    with tab4:
        st.subheader("Re-weighting and Re-sampling Techniques")
        with st.expander("🔍 Friendly Definition"):
            st.write("**Re-weighting:** Assigns more 'weight' or importance to samples from underrepresented groups. **Re-sampling:** Changes the dataset physically, either by duplicating minority group samples (oversampling) or removing majority group samples (undersampling).")
        with st.expander("💡 Interactive Example: Oversampling Simulation"):
            st.write("See how oversampling can balance a dataset with uneven representation.")
            np.random.seed(0)
            data_a = np.random.multivariate_normal([2, 2], [[1, .5], [.5, 1]], 100)
            data_b = np.random.multivariate_normal([4, 4], [[1, .5], [.5, 1]], 20)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.scatter(data_a[:, 0], data_a[:, 1], c='blue', label='Group A (n=100)', alpha=0.6)
            ax1.scatter(data_b[:, 0], data_b[:, 1], c='red', label='Group B (n=20)', alpha=0.6)
            ax1.set_title("Original Data (Unbalanced)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            oversample_indices = np.random.choice(range(20), 80, replace=True)
            data_b_oversampled = np.vstack([data_b, data_b[oversample_indices]])
            ax2.scatter(data_a[:, 0], data_a[:, 1], c='blue', label='Group A (n=100)', alpha=0.6)
            ax2.scatter(data_b_oversampled[:, 0], data_b_oversampled[:, 1], c='red', label='Group B (n=100)', alpha=0.6, marker='x')
            ax2.set_title("Data with Oversampling of Group B")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            st.info("The right plot shows newly added samples (marked with 'x') from Group B to match Group A’s size, helping the model learn their patterns better.")
        st.text_area("Decision Criteria: Re-weight or Re-sample?", 
                     placeholder="Based on my audit and model, the best strategy is...", 
                     key="p8")
        st.text_area("Intersectionality Consideration", 
                     placeholder="Example: To address underrepresentation of minority women, we will apply stratified oversampling to ensure this subgroup reaches parity with others.", 
                     key="p9")

    with tab5:
        st.subheader("Distribution Transformation Approaches")
        with st.expander("🔍 Friendly Definition"):
            st.write("This technique directly modifies feature values to break problematic correlations with protected attributes. It’s like 'recalibrating' a variable so it means the same for all groups.")
        st.text_area("1. Disparate Impact Removal", 
                     placeholder="E.g.: 'Repair' the 'postal code' feature so its distribution is the same across racial groups, eliminating its use as a proxy.", 
                     key="p10")
        st.text_area("2. Fair Representations (LFR, LAFTR)", 
                     placeholder="E.g.: Use an adversarial autoencoder to learn an applicant profile representation without gender information.", 
                     key="p11")
        st.text_area("3. Intersectionality Considerations", 
                     placeholder="My transformation strategy will focus on intersections of gender and ethnicity...", 
                     key="p12")

    with tab6:
        st.subheader("Fairness-Aware Data Generation")
        with st.expander("🔍 Friendly Definition"):
            st.write("When data is very scarce or biased, we can generate synthetic (artificial) data to fill the gaps. This is especially useful for creating examples of very small intersectional groups or for generating counterfactual scenarios.")
        st.markdown("**When to Generate Data:** When there is severe underrepresentation or counterfactual examples are needed.")
        st.markdown("**Strategies:** Conditional Generation, Counterfactual Augmentation.")
        st.text_area("Intersectionality Considerations", 
                     placeholder="Example: We will use a generative model conditioned on the intersection of age and gender to create synthetic profiles of 'older women in tech', a group absent in our data.", 
                     key="p13")
    with tab7:
        st.subheader("Interseccionalidad en el Pre-procesamiento")
        with st.expander("🔍 Definición Amigable"):
            st.write("""
            La interseccionalidad aquí significa ir más allá de equilibrar los datos para grupos principales (ej. hombres vs. mujeres). Debemos asegurarnos de que los **subgrupos específicos** (ej. mujeres negras, hombres latinos jóvenes) también estén bien representados. Las técnicas de pre-procesamiento deben aplicarse de forma estratificada para corregir desequilibrios en estas intersecciones, que a menudo son las más vulnerables al sesgo.
            """)
        
        with st.expander("💡 Ejemplo Interactivo: Re-muestreo Estratificado Interseccional"):
            st.write("Observa cómo un conjunto de datos puede parecer equilibrado en un eje (Grupo A vs. B), pero no en sus intersecciones. El re-muestreo estratificado soluciona esto.")

            # Datos iniciales
            np.random.seed(1)
            # Grupo A: 100 total (80 Hombres, 20 Mujeres)
            hombres_a = pd.DataFrame({'Característica 1': np.random.normal(2, 1, 80), 'Característica 2': np.random.normal(5, 1, 80), 'Grupo': 'Hombres A'})
            mujeres_a = pd.DataFrame({'Característica 1': np.random.normal(2.5, 1, 20), 'Característica 2': np.random.normal(5.5, 1, 20), 'Grupo': 'Mujeres A'})
            # Grupo B: 100 total (50 Hombres, 50 Mujeres)
            hombres_b = pd.DataFrame({'Característica 1': np.random.normal(6, 1, 50), 'Característica 2': np.random.normal(2, 1, 50), 'Grupo': 'Hombres B'})
            mujeres_b = pd.DataFrame({'Característica 1': np.random.normal(6.5, 1, 50), 'Característica 2': np.random.normal(2.5, 1, 50), 'Grupo': 'Mujeres B'})
            
            # Subgrupo interseccional pequeño
            mujeres_b_interseccional = pd.DataFrame({'Característica 1': np.random.normal(7, 1, 10), 'Característica 2': np.random.normal(3, 1, 10), 'Grupo': 'Mujeres B (Intersección)'})


            df_original = pd.concat([hombres_a, mujeres_a, hombres_b, mujeres_b, mujeres_b_interseccional])
            
            # Aplicar sobremuestreo
            remuestreo_factor = st.slider("Factor de sobremuestreo para 'Mujeres B (Intersección)'", 1, 10, 5, key="inter_remuestreo")
            
            if remuestreo_factor > 1:
                indices_remuestreo = mujeres_b_interseccional.sample(n=(remuestreo_factor-1)*len(mujeres_b_interseccional), replace=True).index
                df_remuestreado = pd.concat([df_original, mujeres_b_interseccional.loc[indices_remuestreo]])
            else:
                df_remuestreado = df_original

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

            # Gráfico Original
            for name, group in df_original.groupby('Grupo'):
                ax1.scatter(group['Característica 1'], group['Característica 2'], label=f"{name} (n={len(group)})", alpha=0.7)
            ax1.set_title("Datos Originales")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)

            # Gráfico Remuestreado
            for name, group in df_remuestreado.groupby('Grupo'):
                 ax2.scatter(group['Característica 1'], group['Característica 2'], label=f"{name} (n={len(group)})", alpha=0.7)
            ax2.set_title("Datos con Sobremuestreo Interseccional")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)

            st.pyplot(fig)
            st.info("El grupo 'Mujeres B (Intersección)' estaba severamente subrepresentado. Al aplicar un sobremuestreo específico para este subgrupo, ayudamos al modelo a aprender sus patrones sin distorsionar el resto de los datos.")
        
        st.text_area("Aplica a tu caso: ¿Qué subgrupos interseccionales están subrepresentados en tus datos y qué estrategia de re-muestreo/re-ponderación estratificada podrías usar?", key="p_inter")
    with tab8:
        st.subheader("Integrated Bias Mitigation Techniques")
        st.info("Apply specific bias mitigation techniques with interactive examples and code templates.")
                # Technique selector
        technique = st.selectbox(
            "Select bias mitigation technique:",
            ["Oversampling", "Undersampling", "Reweighting", "SMOTE", "Data Augmentation"],
            key="bias_mit_selector"
        )
        if technique == "Oversampling":
            st.markdown("### Oversampling Implementation")
            st.code("""
# Oversampling for your preprocessing pipeline
def preprocessing_oversampling(data, target_col, protected_attr):
    results = {}
    # Analyze by protected attribute
    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group]
        
        # Separate majority and minority within group
        majority_class = group_data[target_col].mode()[0]
        majority = group_data[group_data[target_col] == majority_class]
        minority = group_data[group_data[target_col] != majority_class]
        
        # Oversample minority
        if len(minority) > 0:
            minority_upsampled = resample(minority,
                                        replace=True,
                                        n_samples=len(majority),
                                        random_state=42)
            results[group] = pd.concat([majority, minority_upsampled])
    
    return pd.concat(results.values(), ignore_index=True)
            """, language="python")
        
        elif technique == "SMOTE":
            st.markdown("### SMOTE for Preprocessing")
            st.code("""
# SMOTE integration in preprocessing
def preprocessing_smote(X, y, sensitive_features):
    # Apply SMOTE while preserving sensitive attribute information
    smote = SMOTE(random_state=42, k_neighbors=5)
    
    # Get original sensitive feature mapping
    sensitive_mapping = dict(zip(range(len(X)), sensitive_features))
    
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Reconstruct sensitive features for new samples
    # (This is a simplified approach - in practice, you'd need more sophisticated mapping)
    n_original = len(X)
    n_synthetic = len(X_resampled) - n_original
    
    # For synthetic samples, assign sensitive attributes based on nearest neighbors
    sensitive_resampled = list(sensitive_features)
    for i in range(n_synthetic):
        # Find closest original sample
        synthetic_sample = X_resampled[n_original + i]
        distances = np.linalg.norm(X - synthetic_sample, axis=1)
        closest_idx = np.argmin(distances)
        sensitive_resampled.append(sensitive_features[closest_idx])
    
    return X_resampled, y_resampled, np.array(sensitive_resampled)
            """, language="python")
        
        # Integration with existing workflow
        st.markdown("### Integration with Existing Preprocessing Workflow")
        st.text_area(
            "How will you integrate this technique with your existing preprocessing steps?",
            placeholder="Example: 1. First apply correlation analysis to identify proxies. 2. Then use SMOTE to balance underrepresented intersectional groups. 3. Finally apply reweighting for fine-tuning. 4. Validate using fairness metrics.",
            key=f"integration_{technique.lower()}"
        )

# ==================================================================
# COMPLETE INTEGRATION FUNCTION
# ==================================================================

def complete_integration_example():
    """Complete example showing how all components work together"""
    st.header("🎯 Complete Integration Example")
    st.info("This example shows how the bias mitigation techniques integrate with your existing audit and intervention playbooks.")
    
    st.markdown("""
    ## Integration Workflow
    
    ### Phase 1: Audit (Using existing audit playbook)
    1. **Historical Context Assessment** → Identify domain-specific bias patterns
    2. **Fairness Definition Selection** → Choose appropriate fairness criteria
    3. **Bias Source Identification** → Locate where bias enters the system
    4. **Comprehensive Fairness Metrics** → Establish measurement framework
    
    ### Phase 2: Data Analysis (Using bias mitigation techniques)
    5. **Dataset Analysis** → Apply representation analysis and correlation detection
    6. **Bias Quantification** → Measure imbalances and identify mitigation needs
    
    ### Phase 3: Mitigation (Using intervention playbook + new techniques)
    7. **Pre-processing** → Apply resampling, reweighting, SMOTE, or augmentation
    8. **In-processing** → Use fairness constraints during training
    9. **Post-processing** → Adjust thresholds and calibration
    
    ### Phase 4: Validation
    10. **Fairness Validation** → Measure improvement using established metrics
    11. **Intersectional Analysis** → Ensure fairness across subgroups
    12. **Monitoring Setup** → Establish ongoing fairness monitoring
    """)
    
    st.code("""
# Complete integration pipeline
def integrated_fairness_pipeline(data_path, config):
    # Phase 1: Audit
    # (Assumes run_fairness_audit is defined below or elsewhere)
    audit_results = run_fairness_audit(data_path, config)
    
    # Phase 2: Load and analyze data
    # (Assumes these functions are defined elsewhere)
    X, y, sensitive_attr = load_data(data_path)
    bias_analysis = analyze_dataset_bias(X, y, sensitive_attr)
    
    # Phase 3: Apply mitigation based on audit results
    if audit_results['requires_resampling']:
        if bias_analysis['imbalance_ratio'] > 5:
            # High imbalance - use SMOTE
            # (Assumes preprocessing_smote is defined elsewhere)
            X, y, sensitive_attr = preprocessing_smote(X, y, sensitive_attr)
        else:
            # Moderate imbalance - use oversampling
            # (Assumes apply_oversampling is defined elsewhere)
            X, y = apply_oversampling(X, y)
    
    if audit_results['requires_reweighting']:
        # (Assumes apply_reweighting is defined elsewhere)
        sample_weights = apply_reweighting(X, y)
    else:
        sample_weights = None
    
    # Phase 4: Train with fairness constraints
    # (Assumes train_fair_model is defined elsewhere)
    model = train_fair_model(X, y, sensitive_attr,
                           fairness_def=audit_results['selected_fairness_def'],
                           sample_weight=sample_weights)
    
    # Phase 5: Post-process if needed
    if audit_results['requires_post_processing']:
        # (Assumes apply_threshold_optimization is defined elsewhere)
        model = apply_threshold_optimization(model, X, y, sensitive_attr)
    
    # Phase 6: Validate
    # (Assumes validate_complete_fairness is defined elsewhere)
    fairness_metrics = validate_complete_fairness(model, X, y, sensitive_attr)
    
    return model, fairness_metrics

def run_fairness_audit(data_path, config):
    \"\"\"Run the audit playbook programmatically\"\"\"
    # Historical context assessment
    # (Assumes assess_historical_context is defined elsewhere)
    hca_results = assess_historical_context(config['domain'])
    
    # Fairness definition selection
    # (Assumes select_fairness_definition is defined elsewhere)
    fairness_def = select_fairness_definition(hca_results, config['use_case'])
    
    # Bias source identification
    # (Assumes identify_bias_sources is defined elsewhere)
    bias_sources = identify_bias_sources(data_path, hca_results)
    
    return {
        'selected_fairness_def': fairness_def,
        'bias_sources': bias_sources,
        'requires_resampling': 'representation_bias' in bias_sources,
        'requires_reweighting': 'measurement_bias' in bias_sources,
        'requires_post_processing': 'deployment_bias' in bias_sources
    }
    """, language="python")

    # --- Report Section ---
    st.markdown("---")
    st.header("Generate Pre-processing Toolkit Report")
    if st.button("Generate Pre-processing Report", key="gen_preproc_report"):
        report_data = {
            "Representation Analysis": {
                "Comparison with Reference Population": st.session_state.get('p1', 'Not completed'),
                "Intersectional Analysis": st.session_state.get('p2', 'Not completed'),
                "Outcome Representation": st.session_state.get('p3', 'Not completed'),
            },
            "Correlation Detection": {
                "Direct Correlations": st.session_state.get('p4', 'Not completed'),
                "Identified Proxy Variables": st.session_state.get('p5', 'Not completed'),
            },
            "Label Quality": {
                "Historical Label Bias": st.session_state.get('p6', 'Not completed'),
                "Annotator Bias": st.session_state.get('p7', 'Not completed'),
            },
            "Re-weighting and Re-sampling": {
                "Decision and Rationale": st.session_state.get('p8', 'Not completed'),
                "Intersectional Plan": st.session_state.get('p9', 'Not completed'),
            },
            "Distribution Transformation": {
                "Disparate Impact Removal Plan": st.session_state.get('p10', 'Not completed'),
                "Fair Representations Plan": st.session_state.get('p11', 'Not completed'),
                "Intersectional Plan": st.session_state.get('p12', 'Not completed'),
            },
            "Data Generation": {
                "Intersectional Data Generation Plan": st.session_state.get('p13', 'Not completed'),
            },
            "Intersectional Pre-processing Strategy": {
                 "Analysis and Strategy": st.session_state.get('p_inter', 'Not completed'),
            }
        }
        
        report_md = "# Pre-processing Fairness Toolkit Report\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.preproc_report_md = report_md
        st.success("✅ Report successfully generated!")

    if 'preproc_report_md' in st.session_state and st.session_state.preproc_report_md:
        st.subheader("Report Preview")
        st.markdown(st.session_state.preproc_report_md)
        st.download_button(
            label="Download Pre-processing Report",
            data=st.session_state.preproc_report_md,
            file_name="preprocessing_report.md",
            mime="text/markdown"
        )
       

def inprocessing_fairness_toolkit():
    st.header("⚙️ In-processing Fairness Toolkit")
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **In-processing** involves modifying the model's learning algorithm so that fairness is one of its objectives, alongside accuracy. 
        It's like teaching a chef to cook not only so the food is delicious but also nutritionally balanced, making nutrition a central part of the recipe.
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Objectives and Constraints", "Adversarial Debiasing", 
        "Multi-objective Optimization", "Code Patterns",
        "🌍 Intersectionality"
    ])
    
    with tab1:
        st.subheader("Fairness Objectives and Constraints")
        with st.expander("🔍 Friendly Definition"):
            st.write("This means incorporating 'fairness rules' directly into the math the model uses to learn. Instead of only seeking the most accurate answer, the model must also ensure it does not violate these rules.")
        
        st.markdown("**Lagrangian Methods:**")
        with st.expander("🔍 Definition and Example"):
            st.write("A mathematical technique to turn a 'hard constraint' (a rule that cannot be broken) into a 'soft penalty'. Imagine you're training a robot to be fast but it cannot exceed a certain speed. Instead of a strict limit, you give it a penalty every time it gets close to that limit, encouraging it to stay within bounds more flexibly.")
        st.latex(r''' \mathcal{L}(\theta, \lambda) = L(\theta) + \sum_{i=1}^{k} \lambda_i C_i(\theta) ''')
        st.text_area("Apply to your case: What fairness constraint (e.g., max approval rate difference) do you want to implement?", key="in_q1")

        st.markdown("**Feasibility and Trade-offs:**")
        with st.expander("🔍 Definition and Example"):
            st.write("It is not always possible to be perfectly fair and perfectly accurate at the same time. Often there is a 'trade-off'. Improving fairness can slightly reduce overall accuracy, and vice versa. It’s crucial to understand this balance.")
            st.write("**Intersectionality Example:** Forcing equal outcomes for all subgroups (e.g., Latina women, Asian men) may be mathematically impossible or require such a large sacrifice in accuracy that the model becomes unusable.")
        st.text_area("Apply to your case: What trade-off between accuracy and fairness are you willing to accept?", key="in_q2")


    # TAB 2: Adversarial Debiasing
    with tab2:
        st.subheader("Adversarial Debiasing Approaches")
        with st.expander("🔍 Friendly Definition"):
            st.write("Imagine a game between two AIs: a 'Predictor' that tries to do its job (e.g., evaluate resumes) and an 'Adversary' that tries to guess the protected attribute (e.g., candidate gender) based on the Predictor’s decisions. The Predictor wins if it makes good evaluations AND fools the Adversary. Over time, the Predictor learns to make decisions without relying on information related to gender.")
        
        st.markdown("**Architecture:**")
        with st.expander("💡 Adversarial Architecture Simulator"):
            st.graphviz_chart("""
            digraph {
                rankdir=LR;
                node [shape=box, style=rounded];
                "Input Data (X)" -> "Predictor";
                "Predictor" -> "Prediction (Ŷ)";
                "Predictor" -> "Adversary" [label="Tries to fool"];
                "Adversary" -> "Protected Attribute Prediction (Â)";
                "Protected Attribute (A)" -> "Adversary" [style=dashed, label="Compares to learn"];
            }
            """)
        st.text_area("Apply to your case: Describe the architecture you would use.", 
                     placeholder="E.g.: A BERT-based predictor for analyzing CVs and a 3-layer adversary to predict gender from internal representations.", 
                     key="in_q3")

        st.markdown("**Optimization:**")
        with st.expander("🔍 Definition and Example"):
            st.write("Training can be unstable because the Predictor and Adversary have opposing objectives. Special techniques, like 'gradient reversal', are needed so the Predictor actively 'unlearns' bias.")
        st.text_area("Apply to your case: What optimization challenges do you foresee and how would you address them?", 
                     placeholder="E.g.: The adversary could become too strong at the start. We will use a gradual increase in its weight in the loss function.", 
                     key="in_q4")

    # TAB 3: Multi-objective Optimization
    with tab3:
        st.subheader("Multi-objective Optimization for Fairness")
        with st.expander("🔍 Friendly Definition"):
            st.write("Instead of combining accuracy and fairness into a single goal, this approach treats them as two separate objectives to balance. The goal is to find a set of 'Pareto optimal solutions', where you cannot improve fairness without sacrificing some accuracy, and vice versa.")
        with st.expander("💡 Interactive Example: Pareto Frontier"):
            st.write("Explore the **Pareto frontier**, which visualizes the trade-off between a model's accuracy and its fairness. You cannot improve one without worsening the other.")
            
            np.random.seed(10)
            accuracy = np.linspace(0.80, 0.95, 20)
            fairness_score = 1 - np.sqrt(accuracy - 0.79) + np.random.normal(0, 0.02, 20)
            fairness_score = np.clip(fairness_score, 0.5, 1.0)
            
            fig, ax = plt.subplots()
            ax.scatter(accuracy, fairness_score, c=accuracy, cmap='viridis', label='Possible Models')
            ax.set_title("Pareto Frontier: Fairness vs. Accuracy")
            ax.set_xlabel("Model Accuracy")
            ax.set_ylabel("Fairness Score")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            st.info("Each point represents a different model. Models on the top-right edge are 'optimal'. The choice of which point to use depends on your project's priorities.")
        st.text_area("Apply to your case: What multiple objectives do you need to balance?", 
                     placeholder="E.g.: 1. Maximize accuracy in default prediction. 2. Minimize approval rate differences between demographic groups. 3. Minimize false negative rate differences.", 
                     key="in_q5")

    # TAB 4: Code Patterns
    with tab4:
        st.subheader("Implementation Pattern Catalog")
        with st.expander("🔍 Friendly Definition"):
            st.write("These are code or pseudocode snippets showing how in-processing techniques look in practice. They serve as reusable templates for implementing fairness in your own code.")
        st.code("""
# Example of a fairness-regularized loss function
def fairness_regularized_loss(original_loss, predictions, protected_attribute):
    # Calculate a penalty based on prediction disparity
    fairness_penalty = calculate_disparity(predictions, protected_attribute)
    
    # Combine the original loss with the fairness penalty
    # lambda controls how much importance is given to fairness
    return original_loss + lambda * fairness_penalty
        """, language="python")

    # TAB 5: Intersectionality in In-processing
    with tab5:
        st.subheader("Intersectionality in In-processing")
        with st.expander("🔍 Friendly Definition"):
            st.write("""
            Intersectional fairness at this stage means that the 'fairness rules' we add to the model must protect not only main groups but also intersections. 
            A model can be fair for 'women' and for 'minorities' in general but very unfair to 'minority women'. 
            In-processing techniques must be able to handle multiple fairness constraints for these specific subgroups.
            """)

        with st.expander("💡 Interactive Example: Subgroup Constraints"):
            st.write("See how adding a specific constraint for an intersectional subgroup can improve its fairness, sometimes at the cost of overall accuracy.")
            
            np.random.seed(42)
            # Simple simulated data
            X_maj = np.random.normal(1, 1, (100, 2))
            y_maj = (X_maj[:, 0] > 1).astype(int)
            X_min1 = np.random.normal(-1, 1, (50, 2))
            y_min1 = (X_min1[:, 0] > -1).astype(int)
            X_min2 = np.random.normal(0, 1, (50, 2))
            y_min2 = (X_min2[:, 0] > 0).astype(int)
            X_inter = np.random.normal(-2, 1, (20, 2))
            y_inter = (X_inter[:, 0] > -2).astype(int)

            X_total = np.vstack([X_maj, X_min1, X_min2, X_inter])
            y_total = np.concatenate([y_maj, y_min1, y_min2, y_inter])
            
            # Base model without constraints
            model_base = LogisticRegression(solver='liblinear').fit(X_total, y_total)
            acc_base = model_base.score(X_total, y_total)
            acc_inter_base = model_base.score(X_inter, y_inter)

            # Model WITH constraint (simulated)
            lambda_inter = st.slider("Constraint strength for 'Women B'", 0.0, 1.0, 0.5, key="in_inter_lambda")
            
            acc_con = acc_base * (1 - 0.1 * lambda_inter) 
            acc_inter_con = acc_inter_base + (0.95 - acc_inter_base) * lambda_inter 
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Without Intersectional Constraint**")
                st.metric("Overall Accuracy", f"{acc_base:.2%}")
                st.metric("Accuracy for 'Women B'", f"{acc_inter_base:.2%}", delta_color="off")
            with col2:
                st.write("**Model WITH Intersectional Constraint**")
                st.metric("Overall Accuracy", f"{acc_con:.2%}", delta=f"{(acc_con-acc_base):.2%}")
                st.metric("Accuracy for 'Women B'", f"{acc_inter_con:.2%}", delta=f"{(acc_inter_con-acc_inter_base):.2%}")

            st.info("Increasing the constraint strength for the 'Women B' subgroup significantly improves its accuracy. However, this may cause a slight decrease in the model’s overall accuracy. This is the fairness trade-off.")
        
        st.text_area("Apply to your case: What specific fairness constraints for subgroups do you need to incorporate into your model?", key="in_inter")


    # --- Sección de Reporte ---
     # --- Report Section ---
    st.markdown("---")
    st.header("Generate In-processing Toolkit Report")
    if st.button("Generate In-processing Report", key="gen_inproc_report"):
        report_data = {
            "Objectives and Constraints": {
                "Fairness Constraint": st.session_state.get('in_q1', 'Not completed'),
                "Trade-off Analysis": st.session_state.get('in_q2', 'Not completed'),
            },
            "Adversarial Debiasing": {
                "Architecture Description": st.session_state.get('in_q3', 'Not completed'),
                "Optimization Plan": st.session_state.get('in_q4', 'Not completed'),
            },
            "Multi-objective Optimization": {
                "Objectives to Balance": st.session_state.get('in_q5', 'Not completed'),
            },
            "Intersectional In-processing Strategy": {
                "Analysis and Strategy": st.session_state.get('in_inter', 'Not completed'),
            }
        }
        
        report_md = "# In-processing Fairness Toolkit Report\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.inproc_report_md = report_md
        st.success("✅ Report successfully generated!")

    if 'inproc_report_md' in st.session_state and st.session_state.inproc_report_md:
        st.subheader("Report Preview")
        st.markdown(st.session_state.inproc_report_md)
        st.download_button(
            label="Download In-processing Report",
            data=st.session_state.inproc_report_md,
            file_name="inprocessing_report.md",
            mime="text/markdown"
        )

def postprocessing_fairness_toolkit():
    st.header("📊 Post-processing Fairness Toolkit")
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **Post-processing** consists of adjusting a model's predictions *after* it has already been trained. 
        It's like an editor reviewing a written text to correct bias or mistakes. 
        The original model does not change—only its final output is adjusted to make it fairer.
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Threshold Optimization", "Calibration", "Prediction Transformation", 
        "Rejection Classification", "🌍 Intersectionality"
    ])

    # TAB 1: Threshold Optimization
    with tab1:
        st.subheader("Threshold Optimization Techniques")
        with st.expander("💡 Interactive Example"):
             run_threshold_simulation()
        st.info("Adjust classification thresholds after training to meet specific fairness definitions.")
        st.text_area("Apply to your case: What fairness criterion will you use and how do you plan to analyze trade-offs?", 
                     placeholder="1. Criterion: Equal Opportunity.\n2. Calculation: Find thresholds that equalize TPR in a validation set.\n3. Deployment: Use a proxy for demographic group since we cannot use the protected attribute in production.", 
                     key="po_q1")

    # TAB 2: Calibration
    with tab2:
        st.subheader("Practical Calibration Guide for Fairness")
        with st.expander("🔍 Friendly Definition"):
            st.write("**Calibration** ensures that a prediction of '80% probability' means the same thing for all demographic groups. If for one group it means 95% actual probability and for another 70%, the model is miscalibrated and unfair.")
        with st.expander("💡 Interactive Example: Calibration Simulation"):
            run_calibration_simulation()
        
        with st.expander("Definition: Platt Scaling and Isotonic Regression"):
            st.write("**Platt Scaling:** A simple technique that uses a logistic model to 'readjust' your model’s scores into well-calibrated probabilities. Like applying a smooth correction curve.")
            st.write("**Isotonic Regression:** A more flexible, non-parametric method that adjusts scores through a stepwise function. Powerful but may overfit if data is scarce.")
        st.text_area("Apply to your case: How will you evaluate and correct calibration?", 
                     placeholder="1. Evaluation: Use reliability diagrams and ECE metric by group.\n2. Method: Test Platt Scaling by group, as it's robust and easy to implement.", 
                     key="po_q2")

    # TAB 3: Prediction Transformation
    with tab3:
        st.subheader("Prediction Transformation Methods")
        with st.expander("🔍 Friendly Definition"):
            st.write("These are more advanced techniques than simple threshold optimization. They modify the model’s scores in more complex ways to meet fairness criteria, especially when retraining the model is not possible.")
        
        with st.expander("Definition: Learned Transformation Functions"):
            st.write("Instead of a simple adjustment, an optimal mathematical function is 'learned' to transform biased scores into fair scores, minimizing loss of useful information.")
        with st.expander("Definition: Distribution Alignment"):
            st.write("Ensures that the distribution of scores (the 'histogram' of predictions) is similar for all demographic groups. Useful for achieving demographic parity.")
        with st.expander("Definition: Fair Score Transformations"):
            st.write("Adjusts scores to meet fairness requirements while keeping one important rule: the relative order of individuals within the same group must remain. If person A ranked higher than B in a group, it should remain that way after transformation.")
        
        st.text_area("Apply to your case: Which transformation method is most suitable and why?", 
                     placeholder="E.g.: Use distribution alignment via quantile mapping to ensure credit risk score distributions are comparable between groups, as our goal is demographic parity.", 
                     key="po_q3")

    # TAB 4: Rejection Classification
    with tab4:
        st.subheader("Rejection Option Classification")
        with st.expander("🔍 Friendly Definition"):
            st.write("Instead of forcing the model to make a decision in difficult or ambiguous cases (where it is more likely to make unfair errors), this technique identifies those cases and 'rejects' them, sending them to a human expert for a final decision.")
        with st.expander("💡 Interactive Example: Rejection Simulation"):
            run_rejection_simulation()
            
        with st.expander("Definition: Confidence-based rejection thresholds"):
            st.write("Confidence zones are defined. If the model’s predicted probability is very high (e.g., >90%) or very low (e.g., <10%), the decision is automated. If it falls in the middle, it is rejected for human review.")
        with st.expander("Definition: Selective classification"):
            st.write("The formal framework for deciding what percentage of cases to automate. It optimizes the balance between 'coverage' (how many cases are automatically decided) and fairness.")
        with st.expander("Definition: Human-AI collaboration models"):
            st.write("It’s not enough to reject a case. How information is presented to the human must be carefully designed to avoid introducing new biases. The goal is collaboration where AI and human together make fairer decisions than either alone.")
        
        st.text_area("Apply to your case: How would you design a rejection system?", 
                     placeholder="E.g.: Reject loan applications with probabilities between 40% and 60% for manual review. The reviewer interface will display key data without revealing the demographic group to avoid human bias.", 
                     key="po_q4")

    # TAB 5: Intersectionality
    with tab5:
        st.subheader("Intersectionality in Post-processing")
        with st.expander("🔍 Friendly Definition"):
            st.write("""
            Here, intersectionality means we cannot use a single decision threshold or a single calibration curve for everyone. 
            Each **intersectional subgroup** (e.g., young women, older men from another ethnicity) may have its own score distribution and its own relationship with reality. 
            Post-processing techniques must therefore be applied granularly for each relevant subgroup.
            """)

        with st.expander("💡 Interactive Example: Thresholds for Intersectional Subgroups"):
            st.write("Adjust thresholds for four intersectional subgroups to achieve Equal Opportunity (equal TPRs) across all of them. See how the task becomes more complex.")

            np.random.seed(123)
            # Simulated data for 4 subgroups
            groups = {
                "Men-A": (np.random.normal(0.7, 0.15, 50), np.random.normal(0.4, 0.15, 70)),
                "Women-A": (np.random.normal(0.65, 0.15, 40), np.random.normal(0.35, 0.15, 80)),
                "Men-B": (np.random.normal(0.6, 0.15, 60), np.random.normal(0.3, 0.15, 60)),
                "Women-B": (np.random.normal(0.55, 0.15, 30), np.random.normal(0.25, 0.15, 90)),
            }
            dfs = {
                name: pd.DataFrame({
                    'Score': np.concatenate(scores),
                    'Actual': [1]*len(scores[0]) + [0]*len(scores[1])
                }) for name, scores in groups.items()
            }
            
            st.write("#### Threshold Adjustment")
            cols = st.columns(4)
            thresholds = {}
            for i, name in enumerate(dfs.keys()):
                with cols[i]:
                    thresholds[name] = st.slider(f"{name} Threshold", 0.0, 1.0, 0.5, key=f"po_inter_{i}")

            st.write("#### Results (True Positive Rate)")
            tprs = {}
            cols_res = st.columns(4)
            for i, name in enumerate(dfs.keys()):
                df = dfs[name]
                tpr = np.mean(df[df['Actual'] == 1]['Score'] >= thresholds[name])
                tprs[name] = tpr
                with cols_res[i]:
                    st.metric(f"TPR {name}", f"{tpr:.2%}")

            max_tpr_diff = max(tprs.values()) - min(tprs.values())
            if max_tpr_diff < 0.05:
                st.success(f"✅ Great! The maximum TPR difference across subgroups is only {max_tpr_diff:.2%}.")
            else:
                st.warning(f"Adjust thresholds to equalize TPRs. Current max difference: {max_tpr_diff:.2%}")

        st.text_area("Apply to your case: For which intersectional subgroups do you need separate thresholds or calibration curves?", key="po_inter")


    # --- Report Section ---
    st.markdown("---")
    st.header("Generate Post-processing Toolkit Report")
    if st.button("Generate Post-processing Report", key="gen_postproc_report"):
        report_data = {
            "Threshold Optimization": {"Implementation Plan": st.session_state.get('po_q1', 'Not completed')},
            "Calibration": {"Calibration Plan": st.session_state.get('po_q2', 'Not completed')},
            "Prediction Transformation": {"Selected Transformation Method": st.session_state.get('po_q3', 'Not completed')},
            "Rejection Classification": {"Rejection System Design": st.session_state.get('po_q4', 'Not completed')},
            "Intersectional Post-processing Strategy": {"Analysis and Strategy": st.session_state.get('po_inter', 'Not completed')}
        }
        
        report_md = "# Post-processing Fairness Toolkit Report\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.postproc_report_md = report_md
        st.success("✅ Report successfully generated!")

    if 'postproc_report_md' in st.session_state and st.session_state.postproc_report_md:
        st.subheader("Report Preview")
        st.markdown(st.session_state.postproc_report_md)
        st.download_button(
            label="Download Post-processing Report",
            data=st.session_state.postproc_report_md,
            file_name="postprocessing_report.md",
            mime="text/markdown"
        )



def intervention_playbook():
    st.sidebar.title("Intervention Playbook Navigation")
    selection = st.sidebar.radio(
        "Go to:",
        ["Main Playbook", "Causal Toolkit", "Pre-processing Toolkit", "In-processing Toolkit", "Post-processing Toolkit","Bias Mitigation Techniques"],
        key="intervention_nav"
    )
    
    if selection == "Main Playbook":
        st.header("📖 Fairness Intervention Playbook")
        st.info("This playbook integrates the four toolkits into a cohesive workflow, guiding developers from bias identification to the implementation of effective solutions.")
        
        with st.expander("Implementation Guide"):
            st.write("Explains how to use the playbook, with comments on key decision points, supporting evidence, and identified risks.")
        
        with st.expander("Case Study"):
            st.write("Demonstrates the application of the playbook to a typical fairness problem, showing how the results of each component inform the next.")
        
        with st.expander("Validation Framework"):
            st.write("Provides guidance on how implementation teams can verify the effectiveness of their audit process.")
        
        with st.expander("Intersectional Fairness"):
            st.write("Explicit consideration of intersectional fairness in each component of the playbook.")
    
    elif selection == "Causal Toolkit":
        causal_fairness_toolkit()
    
    elif selection == "Pre-processing Toolkit":
        preprocessing_fairness_toolkit()
    
    elif selection == "In-processing Toolkit":
        inprocessing_fairness_toolkit()
    
    elif selection == "Post-processing Toolkit":
        postprocessing_fairness_toolkit()
    elif selection == "Bias Mitigation Techniques":
        bias_mitigation_techniques_toolkit()
    elif playbook_choice == "Complete Integration Example":
        complete_integration_example()


def enhanced_preprocessing_with_bias_mitigation():
    """Enhanced version of your preprocessing toolkit with integrated bias mitigation"""
    st.header("🧪 Enhanced Pre-processing with Bias Mitigation")
    
    # Add bias mitigation techniques as a core component
    with st.expander("🔧 Quick Bias Mitigation Assessment"):
        st.write("Quick assessment to determine which bias mitigation technique to apply")
        
        col1, col2 = st.columns(2)
        with col1:
            dataset_size = st.selectbox("Dataset size", ["Small (<1K)", "Medium (1K-10K)", "Large (>10K)"])
            imbalance_ratio = st.selectbox("Imbalance ratio", ["Low (<2:1)", "Medium (2:1-5:1)", "High (>5:1)"])
        
        with col2:
            data_type = st.selectbox("Data type", ["Tabular", "Images", "Text", "Mixed"])
            quality_concern = st.selectbox("Main quality concern", ["Representation", "Label bias", "Proxy variables"])
        
        # Recommendation logic
        recommendations = []
        if imbalance_ratio == "High (>5:1)":
            if dataset_size == "Small (<1K)":
                recommendations.append("SMOTE - Good for small, highly imbalanced datasets")
            else:
                recommendations.append("Oversampling + Reweighting - Effective for large imbalanced datasets")
        elif imbalance_ratio == "Medium (2:1-5:1)":
            recommendations.append("Reweighting - Simple and effective for moderate imbalance")
        
        if data_type == "Images":
            recommendations.append("Data Augmentation - Ideal for image data")
        elif data_type == "Text":
            recommendations.append("Text Augmentation - Paraphrasing and back-translation")
        
        if quality_concern == "Proxy variables":
            recommendations.append("Feature Transformation - Remove problematic correlations")
        
        if recommendations:
            st.success("Recommended techniques:")
            for rec in recommendations:
                st.write(f"• {rec}")

# ==================================================================
# INTEGRATION HELPER FUNCTIONS
# ==================================================================

def create_integrated_workflow():
    """Helper function to create integrated workflow"""
    st.markdown("""
    ### How to Use This Integration
    
    1. **Start with Audit Playbook** → Understand your bias landscape
    2. **Apply Bias Mitigation Techniques** → Address data-level issues
    3. **Use Intervention Playbook** → Apply algorithmic solutions
    4. **Monitor with Comprehensive Metrics** → Ensure ongoing fairness
    
    Each component feeds into the next, creating a comprehensive approach to AI fairness.
    """)

#======================================================================
# --- FAIRNESS AUDIT PLAYBOOK ---
#======================================================================

def audit_playbook():
    st.sidebar.title("Audit Playbook Navigation")
    page = st.sidebar.radio("Go to", [
        "How to Navigate this Playbook",
        "Historical Context Assessment",
        "Fairness Definition Selection",
        "Bias Source Identification",
        "Comprehensive Fairness Metrics"
    ], key="audit_nav")

    # PAGE 1: How to Navigate
    if page == "How to Navigate this Playbook":
        st.header("How to Navigate This Playbook")
        st.markdown("""
        **The Four-Component Framework** – Follow sequentially through:
        
        1. **Historical Context Assessment (HCA)** – Uncover systemic biases and power imbalances in your domain.
        
        2. **Fairness Definition Selection (FDS)**
         – Choose fairness definitions appropriate to your context and goals.
        
        3. **Bias Source Identification (BSI)** – Identify and prioritize ways bias can enter your system.
        
        4. **Comprehensive Fairness Metrics (CFM)**
         – Implement quantitative metrics for monitoring and reporting.

        **Tips:**
        - Progress through sections in order, but feel free to go back if new insights emerge.
        - Use **Save Summary** buttons in each tool to record your findings.
        - Check the embedded examples in each section to see how others have applied these tools.
        """)       

    # PAGE 2: Historical Context Assessment
    elif page == "Historical Context Assessment":
        st.header("Historical Context Assessment Tool")
        with st.expander("🔍 Friendly Definition"):
            st.write("""
            **Historical Context** is the social and cultural backdrop in which your AI will operate. 
            Biases don’t originate in algorithms, they originate in society. Understanding the history 
            of discrimination in areas like banking or hiring helps anticipate where your AI might fail 
            and perpetuate past injustices.
            """)
        st.subheader("1. Structured Questionnaire")
        st.markdown("This section helps you uncover relevant historical patterns of discrimination.")
        
        q1 = st.text_area("What specific domain will this system operate in (e.g., loans, hiring, healthcare)?", key="audit_q1")
        q2 = st.text_area("What is the specific function or use case of the system within that domain?", key="audit_q2")
        q3 = st.text_area("What documented patterns of historical discrimination exist in this domain?", key="audit_q3")
        q4 = st.text_area("What historical data sources are used or referenced in this system?", key="audit_q4")
        q5 = st.text_area("How were key categories (e.g., gender, credit risk) historically defined and have they evolved?", key="audit_q5")
        q6 = st.text_area("How were variables (e.g., income, education) historically measured? Could they encode biases?", key="audit_q6")
        q7 = st.text_area("Have other technologies served similar roles in this domain? Did they challenge or reinforce inequalities?", key="audit_q7")
        q8 = st.text_area("How could automation amplify past biases or introduce new risks in this domain?", key="audit_q8")

        st.subheader("2. Risk Classification Matrix")
        st.markdown("""
        For each historical pattern identified, estimate:
        - **Severity**: High = impacts rights/life outcomes, Medium = affects opportunities/access to resources, Low = limited material impact.
        - **Probability**: High = likely to appear in similar systems, Medium = possible, Low = rare.
        - **Relevance**: High = directly related to your system, Medium = affects parts, Low = peripheral.
        """)
        matrix = st.text_area("Risk Classification Matrix (Markdown table)", height=200, 
                              placeholder="| Pattern | Severity | Probability | Relevance | Score (S×P×R) | Priority |\n|---|---|---|---|---|---|", 
                              key="audit_matrix")

        if st.button("Save HCA Summary"):
            summary = {
                "Structured Questionnaire": {
                    "Domain": q1, 
                    "Function": q2, 
                    "Historical Patterns": q3, 
                    "Data Sources": q4,
                    "Category Definitions": q5, 
                    "Measurement Risks": q6, 
                    "Previous Systems": q7, 
                    "Automation Risks": q8
                },
                "Risk Matrix": matrix
            }
            summary_md = "# Historical Context Assessment Summary\n"
            for section, answers in summary.items():
                summary_md += f"## {section}\n"
                if isinstance(answers, dict):
                    for k, v in answers.items():
                        summary_md += f"**{k}:** {v}\n\n"
                else:
                    summary_md += f"{answers}\n"
            
            st.subheader("HCA Summary Preview")
            st.markdown(summary_md)
            st.download_button("Download HCA Summary", summary_md, "HCA_summary.md", "text/markdown")
            st.success("Historical Context Assessment summary saved.")

    # PAGE 3: Fairness Definition Selection
    elif page == "Fairness Definition Selection":
        st.header("Fairness Definition Selection Tool")
        with st.expander("🔍 Friendly Definition"):
            st.write("""
            There’s no single “recipe” for fairness. Different situations require different types of justice. 
            This section helps you choose the **fairness definition** most suitable for your project, 
            like a doctor choosing the right treatment for a specific condition. 
            Some definitions aim for equality of outcomes, others for equality of opportunity, 
            and the right choice depends on your goal and the harm you aim to prevent.
            """)
        st.subheader("1. Fairness Definition Catalog")
        st.markdown("""
        | Definition | Formula | When to Use | Example |
        |---|---|---|---|
        | Demographic Parity | P(Ŷ=1|A=a) = P(Ŷ=1|A=b) | Ensure equal positive rates across groups. | University ads shown equally to all genders. |
        | Equal Opportunity | P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b) | Minimize false negatives among qualified individuals. | Equal sensitivity of medical test across races. |
        | Equalized Odds | P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) ∀ y | Balance false positives and negatives across groups. | Recidivism predictions with equal error rates. |
        | Calibration | P(Y=1|ŝ=s,A=a) = s | When predicted scores are exposed to users. | Credit scores calibrated for different demographics. |
        | Counterfactual Fairness | Ŷ(x) = Ŷ(x') if A changes | Require removal of causal bias related to sensitive traits. | Outcome unchanged if only race changes in the profile. |
        """)
        st.subheader("2. Decision Tree for Selection")
        exclusion = st.radio("Did the HCA reveal systemic exclusion of protected groups?", ("Yes", "No"), key="fds1")
        error_harm = st.radio("Which type of error is more harmful in your context?", ("False Negatives", "False Positives", "Both equally"), key="fds2")
        score_usage = st.checkbox("Will the outputs be used as scores (e.g., risk, ranking)?", key="fds3")
        
        st.subheader("Recommended Definitions")
        definitions = []
        if exclusion == "Yes": definitions.append("Demographic Parity")
        if error_harm == "False Negatives": definitions.append("Equal Opportunity")
        elif error_harm == "False Positives": definitions.append("Predictive Equality")
        elif error_harm == "Both equally": definitions.append("Equalized Odds")
        if score_usage: definitions.append("Calibration")
        
        for d in definitions: st.markdown(f"- **{d}**")

    # PAGE 4: Bias Source Identification
    elif page == "Bias Source Identification":
        st.header("Bias Source Identification Tool")

        st.subheader("1. Bias Taxonomy")
        st.markdown("""
        | Bias Type | Description | Example |
        |-----------|-------------|---------|
        | Historical Bias | Arises from social inequities encoded in data. | Resume reviews based on historically biased performance data. |
        | Representation Bias | Under- or over-representation of groups in data. | Quechua speakers missing in training samples. |
        | Measurement Bias | Proxies or measures vary in accuracy across groups. | Education level as proxy for skill. |
        | Aggregation Bias | Models combine heterogeneous populations without adjustment. | Applying one-size-fits-all translation. |
        | Learning Bias | Modeling choices reinforce disparities. | Optimization favors majority dialects. |
        | Evaluation Bias | Testing doesn’t reflect deployment reality. | Evaluation excludes elder dialects. |
        | Deployment Bias | Model used in unintended contexts. | Trained on formal speech, used informally. |
        """)

        st.subheader("2. Detection Methodology")
        st.markdown("""
        Connect with Historical Context and Fairness Definitions:

        - **Historical Bias**: Extract patterns from HCA, compare outputs across protected groups.
        - **Representation Bias**: Audit population groups vs. benchmarks from HCA.
        - **Measurement Bias**: Inspect proxies using FDS choices (e.g., test proxy fairness).
        - **Learning Bias**: Analyze model behavior against FDS fairness constraints.
        - **Evaluation Bias**: Compare test and live data distributions; disaggregate metrics.
        - **Deployment Bias**: Review how system is actually used.

        Begin with lightweight analysis. Use expert review only when automated methods are insufficient.
        """)

        st.subheader("3. Prioritization Framework")
        st.markdown("""
        | Bias Type | Severity (1–5) | Scope (1–5) | Persistence (1–5) | Hist. Align. (1–5) | Feasibility (1–5) | Score |
        |-----------|----------------|-------------|-------------------|---------------------|-------------------|-------|
        | Measurement Bias | 5 | 4 | 4 | 5 | 3 | 4.4 |
        | Deployment Bias | 4 | 5 | 4 | 5 | 4 | 4.4 |
        """)

        st.subheader("4. User Documentation")
        st.markdown("""
        - Use HCA to identify likely bias risks.
        - Use FDS to determine which definitions are threatened by bias.
        - Focus on risks with high priority scores.
        - Document findings with reference to both HCA and FDS.
        """)

        st.subheader("5. Case Study: Language Translation Tool")
        st.markdown("""
        **Context**: Tool for translating Indigenous speech to Spanish in public service use.

        **Detected Biases:**
        - Historical Bias: Biased language data from formal Quechua settings.
        - Measurement Bias: Acoustic proxies distorted for elderly voices.
        - Deployment Bias: Real use occurs in informal legal interviews.

        **Priority Ranking:**
        - Measurement Bias: High (Score 4.4)
        - Deployment Bias: High (Score 4.4)

        **Actions:**
        - Expand acoustic samples.
        - Retrain on informal dialogue.
        - Monitor calibration monthly.
        """)


    # PAGE 5: Comprehensive Fairness Metrics
    elif page == "Comprehensive Fairness Metrics":
        st.header("Comprehensive Fairness Metrics (CFM)")

        st.subheader("1. Purpose and Connection")
        st.markdown("""
        This section complements the Historical Context Assessment, Fairness Definition Selection, and Bias Source Identification tools.

        Use this method to connect fairness definitions (from the FDS) with appropriate metrics for your system type.
        """)

        st.markdown("""
        **Problem Type:**
        - **Classification** → Choose: TPR Difference, FPR Difference, Equalized Odds, Demographic Parity
        - **Regression** → Choose: Group Outcome Difference, Group Error Ratio, Residual Disparities
        - **Ranking** → Choose: Exposure Parity, Representation Ratio, Rank-Consistency Score

        **Examples:**
        - **Equal Opportunity (Classification)** → True Positive Rate (TPR) Difference
        - **Demographic Parity (Classification)** → Demographic Parity Difference
        - **Exposure Parity (Ranking)** → Exposure Ratio
        """)

        st.subheader("2. Statistical Validation")
        st.markdown("""
        **Group Metrics Confidence Intervals (Bootstrap):**
        - Resample your dataset with replacement.
        - Compute metric on each sample.
        - Report 95% confidence interval.

        **Small Sample Groups (< 100 instances):**
        - Use Bayesian methods with weak priors.
        - Report credible intervals.
        - Annotate visualizations with group size warnings.
        """)

        st.subheader("3. Visualization and Reporting Templates")
        st.markdown("""
        **Fairness Disparity Chart:**
        - Bar chart with group-wise metrics and confidence intervals.
        - Use color to indicate significant disparities.

        **Intersectional Heatmap:**
        - Show metric values for intersecting groups (e.g., gender × language).
        - Use color gradients for disparity magnitude.
        - Adjust cell size/opacity based on sample size.
        """)

        st.subheader("4. User Documentation")
        st.markdown("""
        - Match your fairness definitions to metrics using the selection guide.
        - Apply statistical validation to assess robustness.
        - Use the visual templates to communicate findings.
        - Report findings with transparency. Flag any significant disparities and small-sample uncertainty.
        """)

        st.subheader("5. Case Study: Language Translation Tool")
        st.markdown("""
        **System Context:** A model ranks translations of Indigenous language utterances for legal or health-related service delivery.

        **Definitions from FDS:**
        - Equal Opportunity (minimize FN for valid inputs)
        - Calibration (ensure confidence score reliability)

        **Selected Metrics:**
        - True Positive Rate (TPR) Difference
        - False Negative Rate (FNR) Difference
        - Calibration Slope per Group
        - Intersectional Equal Opportunity

        **Results:**
        - TPR Difference: 0.19 (95% CI: 0.13–0.25)
        - FNR Difference: 0.24 (95% CI: 0.16–0.30)
        - Calibration Slope Quechua vs Spanish: 0.73
        - Intersectional Gap (TPR): 0.28

        **Visualization:**
        - Bar charts show larger disparities in rural Quechua speakers.
        - Heatmap reveals issues at intersection of age + language.
        """)



# --- NAVEGACIÓN PRINCIPAL ---
st.sidebar.title("Playbook Selection")
playbook_choice = st.sidebar.selectbox(
    "Choose the toolkit you want to use:",
    ["Fairness Audit Playbook", "Fairness Intervention Playbook","Bias Mitigation Techniques","Complete Integration Example" ]
)

st.title(playbook_choice)

if playbook_choice == "Fairness Audit Playbook":
    audit_playbook()
elif playbook_choice == "Fairness Intervention Playbook":
    intervention_playbook()     
elif playbook_choice == "Bias Mitigation Techniques":
    bias_mitigation_techniques_toolkit()
