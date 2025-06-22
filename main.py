
import streamlit as st
import json

st.set_page_config(page_title="Fairness Audit Playbook", layout="wide")
# Playbook Title and Overview
st.title("Fairness Audit Playbook")
st.markdown(
    """
    Welcome to the **Fairness Audit Playbook**, a hands-on guide to help development and audit teams evaluate, select, and implement equity strategies in artificial intelligence systems.

    **Purpose:**
    - Provide a structured framework for identifying patterns of historical discrimination.
    - Guide the selection of appropriate fairness definitions for your context.
    - Detect and prioritize sources of bias in your system.
    - Implement quantitative metrics to continuously monitor and report fairness.

    This Playbook is organized into four main components, complete with ready-to-use examples and templates.
    """
)

st.sidebar.title("Playbook Sections")

page = st.sidebar.radio("Go to", [
     "How to Navigate this Playbook",
    "Historical Context Assessment",
    "Fairness Definition Selection",
    "Bias Source Identification",
    "Comprehensive Fairness Metrics"
])
if page == "How to Navigate this Playbook":
    st.header("How to Navigate This Playbook")
    st.markdown(
        """
     ** The Four-Component Framework** 
     – Follow sequentially through:\
     
  1. **Historical Context Assessment (HCA)** 

  – Uncover systemic biases and power imbalances in your domain.\
  
  2. **Fairness Definition Selection (FDS)**

    – Choose appropriate fairness definitions based on your context and goals.\
    
  3. **Bias Source Identification (BSI)** 

  – Identify and prioritize the ways bias can enter your system.\
  
  4. **Comprehensive Fairness Metrics (CFM)**

    – Implement quantitative metrics for monitoring and reportin

**Tips:**
- Progress through the sections in order, but feel free to loop back if new insights emerge.\
- Use the **Save Summary** buttons in each tool to record your findings.\
- Refer to the examples embedded in each section to see how others have applied these tools.
"""
    )
elif page ==  "Historical Context Assessment":

    st.header("Historical Context Assessment Tool")

    st.subheader("1. Structured Questionnaire")
    st.markdown("""
    This questionnaire helps you uncover relevant patterns of historical discrimination that may influence your AI system’s development. 
    Answer what you can using your team’s current knowledge. Use expert input **only if necessary**.

    **Section 1: Domain and Application Context**
    """)
    q1 = st.text_area("What specific domain will this system operate in (e.g., lending, hiring, healthcare)?")
    q2 = st.text_area("What is the system’s specific function or use case within that domain?")
    q3 = st.text_area("What are the key documented historical discrimination patterns in this domain (explicit or implicit)?")

    st.markdown("""
    **Section 2: Data and Representation Analysis**
    """)
    q4 = st.text_area("What historical data sources are used or referenced by this system?")
    q5 = st.text_area("How were key categories (e.g., gender, credit risk) historically defined and have they evolved?")
    q6 = st.text_area("How were variables measured historically (e.g., income, education)? Could those encode bias?")

    st.markdown("""
    **Section 3: Technology Transition Patterns**
    - **Prior Systems:** Which past technologies performed similar roles, and did they perpetuate or mitigate inequities?
    - **Automation Risks:** How might automation amplify past biases or introduce new risks compared to previous systems?
            

    """)
    q7 = st.text_area("Have other technological systems served similar roles in this domain? Did they challenge or reinforce inequities?")
    q8 = st.text_area("How could automation amplify past biases or introduce new risks in this domain?")
    q9 = st.text_area("Domain:")
    q10 = st.text_area("Function:")
    q11 = st.text_area("Historical Patterns:")
    q12 = st.text_area("Data Sources:")
    q13 = st.text_area("Category Definitions:")
    q14 = st.text_area("Measurement Risks:")
    q15 = st.text_area("Prior Systems:")
    q16 = st.text_area("Automation Risks:")

    st.subheader("2. Risk Classification Matrix")
    st.markdown("""
    For each identified historical pattern, estimate:
    - **Severity**: High = impacts rights/life outcomes, Medium = affects opportunity/resource access, Low = limited material impact.
    - **Likelihood**: High = likely to appear in similar systems, Medium = possible, Low = rare.
    - **Relevance**: High = directly related to your system, Medium = affects parts, Low = peripheral.
    """)            

    st.markdown("""            
     | Pattern                   | Severity | Likelihood | Relevance | Score (S×L×R) | Priority       |
|---------------------------|:--------:|:----------:|:---------:|:-------------:|:--------------:|
| [Your pattern here]       |    -     |     -      |     -     |       -       | [Critical/High]|
 """)
    
    st.markdown("""  
    **Example:**            

    | Historical Pattern | Severity | Likelihood | Relevance | Priority Score (S×L×R) |
    |--------------------|----------|------------|-----------|------------------------|
    | Redlining in lending data     | High (3) | High (3) | High (3) | 9 – Critical Priority |
    | Gender bias in performance reviews | High (3) | Medium (2) | Medium (2) | 7 – High Priority |
    | Indigenous language exclusion | High (3) | High (3) | High (3) | 9 – Critical Priority |
    """)
    st.markdown("""
    For each identified historical pattern Register Severity, Likelihood, Relevance, Score (S×L×R) and Priority
) 
""")
    matrix = st.text_area("Risk Classification Matrix (Markdown table)", height=200)


    st.subheader("3. Usage Guide")
    st.markdown("""
    **How to implement this assessment efficiently:**
    
    **Step 1: Domain Research (1–2 hours)**
    - Review academic and legal sources.
    - Use only evidence-backed patterns.

    **Step 2: Questionnaire Completion (1–2 hours)**
    - Work through the questions internally first.
    - Use experts only where risk  and relevance are evaluated ad High and information is missing or ambiguous .

    **Step 3: Risk Matrix Scoring (30–60 min)**
    - For each pattern, assign Severity × Likelihood × Relevance.
    - Prioritize risks with scores ≥ 7.

    **Step 4: Documentation (1–2 hours)**
    - Compile results into a short memo.
    - Highlight critical/high-priority risks.
    - Link findings to the Fairness Definition phase.
    """)

    st.subheader("4. Case Study: Translation Tool for Low-Resource Language Speakers")
    st.markdown("""
    **Scenario**: A team builds a translation tool for Indigenous languages (Quechua, Maya) to increase access to public service forms.

    **Key Issues Identified (before expert input):**
    - Spanish-only training corpora.
    - Categorical omission of Indigenous dialects.
    - Systemic lack of translated legal terms.

    **Escalation to Experts (only where risk  and relevance are evaluated ad High and information is missing or ambiguous ):**
    - Linguists validated dialect equivalence for regional forms.

    **Risk Matrix Results:**
    - **Spanish-only data** → Severity: High, Likelihood: High, Relevance: High → Score: 9 (Critical)
    - **Omitted dialects** → Severity: Medium, Likelihood: High, Relevance: High → Score: 6 (High)

    **Minimal Actions:**
    - Added open-source bilingual corpora.
    - Implemented glossary fallback module.

    **Outcome:**
    Improved translation reliability by 12% with minimal cost. No full re-training or new architecture needed.
    """)

    if st.button("Save HCA Summary"):
        summary = {
            "Structured Questionnaire": {
                "Domain": q1,
                "Function": q2,
                "Historical Patterns": q3,
                "Data Sources": q4,
                "Category Definitions": q5,
                "Measurement Risks": q6,
                "Technology Transition - Prior Systems": q7,
                "Technology Transition - Automation Risks": q8,
                "Domain (Repeat)": q9,
                "Function (Repeat)": q10,
                "Historical Patterns (Repeat)": q11,
                "Data Sources (Repeat)": q12,
                "Category Definitions (Repeat)": q13,
                "Measurement Risks (Repeat)": q14,
                "Prior Systems": q15,
                "Automation Risks": q16
            },
            "Risk Matrix": matrix
        }
        summary_md = "# Historical Context Assessment Summary\n"
        # Questionnaire summary to markdown
        for section, answers in summary["Structured Questionnaire"].items():
            summary_md += f"**{section}:** {answers}\n\n"
        summary_md += "## Risk Classification Matrix\n" + matrix + "\n"

        # Display summary in app
        st.subheader("HCA Summary Preview")
        st.markdown(summary_md)
        # Download button
        st.download_button(
            label="Download HCA Summary",
            data=summary_md,
            file_name="HCA_summary.md",
            mime="text/markdown"
        )
        # Also print to console for debugging
        print(json.dumps(summary, indent=2))
        st.success("Historical Context Assessment summary saved.")


elif page == "Fairness Definition Selection":
    st.header("Fairness Definition Selection Tool")

    st.subheader("1. Fairness Definition Catalog")
    st.markdown("""
    This catalog presents commonly used fairness definitions with concise descriptions, mathematical formulations, use cases, and limitations. Choose definitions based on historical context, domain-specific goals, and stakeholder priorities.

| Definition             | Formula                                                            | When to Use                                           | Example                                                   |
|------------------------|--------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| Demographic Parity     | P(Ŷ=1\|A=a) = P(Ŷ=1\|A=b)                                         | Ensure equal positive rates across groups.            | College ads shown equally to all gender groups.           |
| Equal Opportunity      | P(Ŷ=1\|Y=1,A=a) = P(Ŷ=1\|Y=1,A=b)                                 | Minimize false negatives among qualified individuals. | Medical test sensitivity equal across races.              |
| Equalized Odds         | P(Ŷ=1\|Y=y,A=a) = P(Ŷ=1\|Y=y,A=b) ∀ y∈{0,1}                      | Balance false positives and negatives across groups.   | Recidivism predictions with equal error rates.           |
| Calibration            | P(Y=1\|ŝ=s,A=a) = s                                                  | When predicted scores are exposed to users.           | Credit scores calibrated for different demographics.      |
| Counterfactual Fairness| Ŷ(x) = Ŷ(x') if A changes                                           | Require causal de-biasing relative to sensitive traits. | Outcome unchanged if only race in profile changes.        |
    """)

    st.subheader("2. Definition Selection Decision Tree")
    st.markdown("""
    Follow these steps to select definitions efficiently based on your system’s context.
    Select options below to determine the most appropriate fairness definitions for your system. You can print or save the results once completed.

    **Step 1: Historical Context Assessment**
            """
    )
    exclusion = st.radio(
    "Did the HCA reveal systemic exclusion of protected groups? → Include Demographic Parity.",
    ("Yes", "No")
    )

    st.markdown("""
                
    **Step 2: Error Impact Analysis**
    """)
    error_harm = st.radio("Which error type is more harmful in your context?",
    ("False Negatives more harmful", "False Positives more harmful", "Both equally harmful", "Neither specific")
    )
    st.markdown("""
    - Are False Negatives more harmful? → Require Equal Opportunity.
    - Are False Positives more harmful? → Consider Predictive Equality.
    - Are both equally harmful? → Use Equalized Odds.

    **Step 3: Exposure of Probabilistic Scores**
    """)

    score_usage = st.checkbox(
    "Will outputs be used as scores (e.g., risk scores, ranked results)→ Add Calibration to ensure group-wise score consistency."
    )
    st.markdown("""
    **Step 4: Feature Sensitivity Check**
    """)
    feature_sensitive = st.checkbox(
    "Are model inputs influenced by sensitive attributes(e.g., proxies for protected groups)? → Consider Counterfactual Fairness.")
                
            # Determine Definitions
    definitions = []
    if exclusion == "Yes":
        definitions.append("Demographic Parity")
    if error_harm == "False Negatives more harmful":
        definitions.append("Equal Opportunity")
    elif error_harm == "False Positives more harmful":
        definitions.append("Predictive Equality")
    elif error_harm == "Both equally harmful":
        definitions.append("Equalized Odds")
    if score_usage:
        definitions.append("Calibration")
    if feature_sensitive:
        definitions.append("Counterfactual Fairness")

    # Display Results
    st.subheader("Recommended Fairness Definitions")
    if definitions:
        for d in definitions:
            st.markdown(f"- **{d}**")
    else:
        st.info("No definitions selected. Adjust the decision inputs above to generate recommendations.")

    # Save & Print
    if st.button("Save & Print Definition Selection"):
        summary = {
            "Systemic Exclusion": exclusion,
            "Error Impact": error_harm,
            "Score Usage": score_usage,
            "Feature Sensitivity": feature_sensitive,
            "Recommendations": definitions
        }
        summary_md = "# Fairness Definition Selection Summary\n"
        for key, val in summary.items():
            summary_md += f"**{key}:** {val}\n\n"
        st.subheader("Definition Selection Summary Preview")
        st.markdown(summary_md)
        st.download_button(
            label="Download FDS Summary",
            data=summary_md,
            file_name="FDS_summary.md",
            mime="text/markdown"
        )
        print(json.dumps(summary, indent=2))


    st.subheader("3. Trade-Off Analysis Template (Customizable for Any Project)")
    project = st.text_input("Project (e.g., Hiring System, Lending Platform, Language Translation)")
    hist_context = st.text_area("Historical Context: What historical patterns of exclusion or bias are relevant to this domain?")
    sel_defs = st.text_area("Selected Definitions:", value=", ".join(definitions))
    rationale = st.text_area("Selection Rationale: Link your definition choice to system goals, stakeholder concerns, and historical context.")
    discarded = st.text_area("Considered But Not Used: List discarded definitions and why they were ruled out.")
    tradeoffs = st.text_area("Trade-offs Acknowledged: What are you sacrificing or prioritizing in choosing this fairness definition?")
    not_met = st.text_area("Fairness Properties Not Met: List the fairness metrics this system will not satisfy and justify.")
    monitoring = st.text_area("Monitoring Approach: Specify how and how often you will evaluate fairness post-deployment.")

    st.subheader("4. User Documentation")
    st.markdown("""
    **How to apply the Fairness Definition Selection Tool:**

    1. Use the outputs from the Historical Context Assessment to identify domains of concern.
    2. Walk through the decision tree to determine suitable fairness definitions.
    3. Record the rationale, limitations, and monitoring strategy using the Trade-Off Template.
    4. Keep documentation lean. Avoid unnecessary complexity unless model behavior demands it.
    """)

    st.subheader("5. Example Case Study: Language Access System (Optional)")
    st.markdown("""
    **Scenario (Example Only)**: A public service tool designed to translate Indigenous languages into Spanish for use in health, education, or legal domains. (Replace this with your own system context.)

    **Key Insights from Historical Context Assessment:**
    - Exclusion of Quechua speakers in legal settings.
    - Underrepresentation in datasets.

    **Fairness Definitions Chosen:**
    - **Equal Opportunity**: To reduce under-detection of valid inputs.
    - **Calibration**: To ensure translation score confidence aligns across language groups.

    **Trade-Offs Made:**
    - Demographic parity not enforced due to legitimate base rate differences.
    - Focus placed on usability and inclusion without sacrificing interpretability.

    **Impact:**
    - 18% increase in user satisfaction.
    - FN rate dropped 22% for Indigenous language users.
    - Monthly audits monitor translation effectiveness and group-wise calibration.
    """)

    if st.button("Save & Print FDS Summary"):
    
        summary = {
            "Systemic Exclusion": exclusion,
            "Error Impact": error_harm,
            "Score Usage": score_usage,
            "Feature Sensitivity": feature_sensitive,
            "Recommendations": definitions,
            "Project": project,
            "Historical Context": hist_context,
            "Selected Definitions": sel_defs,
            "Selection Rationale": rationale,
            "Considered But Not Used": discarded,
            "Trade-offs Acknowledged": tradeoffs,
            "Fairness Properties Not Met": not_met,
            "Monitoring Approach": monitoring
        }
        summary_md = "# Fairness Definition Selection & Trade-Off Summary\n"
        for key, val in summary.items():
            summary_md += f"**{key}:** {val}\n\n"
        st.subheader("FDS Summary Preview")
        st.markdown(summary_md)
        st.download_button(
            label="Download FDS & Trade-Off Summary",
            data=summary_md,
            file_name="FDS_Tradeoff_summary.md",
            mime="text/markdown"
        )
        print(json.dumps(summary, indent=2))    
        st.success("Fairness Definition Selection summary saved.")

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
        # 1. Bias Taxonomy
    bias_types=["Historical Bias","Representation Bias","Measurement Bias","Aggregation Bias","Learning Bias","Evaluation Bias","Deployment Bias"]
    selected_bias=st.multiselect("Select Bias Types to Assess", bias_types)
    taxonomy={}  
    for b in selected_bias:
        taxonomy[b]=st.text_input(f"Example for {b}")
    # 2. Detection Methodology

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
    st.subheader("2. Detection Methodology")
    methods={}
    for b in selected_bias:
        methods[b]=st.text_area(f"Detection method for {b}")


    st.subheader("3. Prioritization Framework")
    st.markdown("""
    | Bias Type | Severity (1–5) | Scope (1–5) | Persistence (1–5) | Hist. Align. (1–5) | Feasibility (1–5) | Score |
    |-----------|----------------|-------------|-------------------|---------------------|-------------------|-------|
    | Measurement Bias | 5 | 4 | 4 | 5 | 3 | 4.4 |
    | Deployment Bias | 4 | 5 | 4 | 5 | 4 | 4.4 |
    """)

    st.markdown("Assign scores (1-5) for each selected bias type:")
    scores={}
    for b in selected_bias:
        cols=st.columns(6)
        sev=cols[0].number_input(f"{b} - Severity",1,5,3,key=f"{b}_sev")
        scp=cols[1].number_input(f"{b} - Scope",1,5,3,key=f"{b}_scp")
        per=cols[2].number_input(f"{b} - Persistence",1,5,3,key=f"{b}_per")
        hal=cols[3].number_input(f"{b} - Hist Align",1,5,3,key=f"{b}_hal")
        fea=cols[4].number_input(f"{b} - Feasibility",1,5,3,key=f"{b}_fea")
        score=round((sev+scp+per+hal+fea)/5,2)
        cols[5].markdown(f"**Score:** {score}")
        scores[b]=score
    # Save & Print
    if st.button("Save & Print BSI Summary"):
        summary={"Taxonomy":taxonomy, "Detection":methods, "Scores":scores}
        smd="# Bias Source Identification Summary\n"
        for section, content in summary.items():
            smd+=f"## {section}\n"
            for k,v in content.items(): smd+=f"- **{k}:** {v}\n"
            smd+="\n"
        st.subheader("BSI Summary Preview")
        st.markdown(smd)
        st.download_button("Download BSI Summary", data=smd, file_name="BSI_summary.md", mime="text/markdown")
        print(json.dumps(summary, indent=2))

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

elif page == "Comprehensive Fairness Metrics":
    st.header("Comprehensive Fairness Metrics (CFM)")

    st.subheader("1. Purpose and Connection")
    st.markdown("""
    This section complements the Historical Context Assessment, Fairness Definition Selection, and Bias Source Identification tools
    """)

    st.markdown("""
    Use this method to connect fairness definitions (from the FDS) with appropriate metrics for your system type.

    **Problem Type:**
    - **Classification** → Choose: TPR Difference, FPR Difference, Equalized Odds, Demographic Parity
    - **Regression** → Choose: Group Outcome Difference, Group Error Ratio, Residual Disparities
    - **Ranking** → Choose: Exposure Parity, Representation Ratio, Rank-Consistency Score

    **Examples:**
    - **Equal Opportunity (Classification)** → True Positive Rate (TPR) Difference
    - **Demographic Parity (Classification)** → Demographic Parity Difference
    - **Exposure Parity (Ranking)** → Exposure Ratio
    """)

    problem_type = st.selectbox("Problem Type", ["Classification","Regression","Ranking"])
    options = []
    if problem_type == "Classification":
        options = ["TPR Difference","FPR Difference","Equalized Odds","Demographic Parity"]
    elif problem_type == "Regression":
        options = ["Group Outcome Difference","Group Error Ratio","Residual Disparities"]
    else:
        options = ["Exposure Parity","Representation Ratio","Rank-Consistency Score"]
    selected_metrics = st.multiselect("Select Fairness Metrics", options)

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

    bootstrap = st.checkbox("Use bootstrap for confidence intervals?")
    bayesian = st.checkbox("Use Bayesian methods for small samples (< threshold)?")
    small_threshold = st.number_input("Small sample threshold", min_value=1, value=100)

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

    st.subheader("Visualization Templates")
    viz_bar = st.checkbox("Include Fairness Disparity Chart")
    viz_heatmap = st.checkbox("Include Intersectional Heatmap")
    viz_other = st.text_area("Other visualization templates")

    st.subheader("Case Study: Language Translation Tool Template")
    st.markdown("**Context:** Indigenous-to-Spanish translation for public services.")
    tpr = st.number_input("TPR Difference", value=0.0)
    fnr = st.number_input("FNR Difference", value=0.0)
    calib = st.number_input("Calibration Slope", value=0.0)
    inter_gap = st.number_input("Intersectional Gap (TPR)", value=0.0)

    if st.button("Save & Print CFM Summary"):
        summary = {
            "Problem Type": problem_type,
            "Selected Metrics": selected_metrics,
            "Bootstrap CI": bootstrap,
            "Bayesian Methods": bayesian,
            "Small Sample Threshold": small_threshold,
            "Visualization": {
                "Bar Chart": viz_bar,
                "Heatmap": viz_heatmap,
                "Other": viz_other
            },
            "Case Study Metrics": {
                "TPR Difference": tpr,
                "FNR Difference": fnr,
                "Calibration Slope": calib,
                "Intersectional Gap": inter_gap
            }
        }
        md = "# CFM Summary\n"
        for k,v in summary.items():
            if isinstance(v, dict):
                md += f"## {k}\n"
                for subk, subv in v.items(): md += f"- **{subk}:** {subv}\n"
            else:
                md += f"**{k}:** {v}\n"
            md += "\n"
        st.subheader("CFM Summary Preview")
        st.markdown(md)
        st.download_button("Download CFM Summary", data=md, file_name="CFM_summary.md", mime="text/markdown")
        print(json.dumps(summary, indent=2))

else:
    st.info("Select a section from the sidebar to begin.")
