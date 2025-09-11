import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from collections import Counter
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import uuid
import sys
import os
from urllib.parse import quote 
from dataclasses import dataclass, field
from enum import Enum
import altair as alt
#from groq import Groq



os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="AI Fairness Playbooks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CORE ENUMS AND DATA STRUCTURES
# ============================================================================

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    AI_ACT = "ai_act"
    BIAS_AUDIT_NYC = "bias_audit_nyc"
    NIST_AI_RMF = "nist_ai_rmf"

class AISystemType(Enum):
    LLM = "large_language_model"
    RECOMMENDATION = "recommendation_system"
    COMPUTER_VISION = "computer_vision"
    MULTIMODAL = "multimodal"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class Domain(Enum):
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RECRUITMENT = "recruitment"
    EDUCATION = "education"
    CRIMINAL_JUSTICE = "criminal_justice"

# ============================================================================
# COMPONENT 1: FAIR AI SCRUM TOOLKIT
# ============================================================================

@dataclass
class FairnessUserStory:
    """Template for fairness-aware user stories"""
    id: str
    title: str
    as_a: str  # User role
    i_want: str  # Functionality
    so_that: str  # Business value
    fairness_considerations: List[str] = field(default_factory=list)
    bias_scenarios: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    fairness_acceptance_criteria: List[str] = field(default_factory=list)
    story_points: int = 0
    fairness_complexity: RiskLevel = RiskLevel.LOW

class FairnessUserStoryLibrary:
    """Library of fairness user story templates"""

    def __init__(self):
        self.templates = {
            "bias_detection": FairnessUserStory(
                id="FAIR-001",
                title="Bias Detection in Model Predictions",
                as_a="Data Scientist",
                i_want="automated bias detection in our model's predictions",
                so_that="we can identify and address unfair outcomes before deployment",
                fairness_considerations=[
                    "Statistical parity across protected groups",
                    "Equal opportunity metrics",
                    "Calibration across demographics"
                ],
                bias_scenarios=[
                    "Gender bias in hiring predictions",
                    "Racial bias in credit scoring",
                    "Age bias in insurance pricing"
                ],
                fairness_acceptance_criteria=[
                    "Bias metrics calculated for all protected attributes",
                    "Automated alerts when bias thresholds exceeded",
                    "Bias report generated for each model version"
                ]
            ),
            "explainable_decisions": FairnessUserStory(
                id="FAIR-002",
                title="Explainable AI Decision Making",
                as_a="End User",
                i_want="clear explanations for AI-driven decisions",
                so_that="I can understand and trust the system's recommendations",
                fairness_considerations=[
                    "Explanation consistency across user groups",
                    "Understandable language for all literacy levels",
                    "Cultural sensitivity in explanations"
                ],
                bias_scenarios=[
                    "Complex explanations favoring technical users",
                    "Language barriers in explanation delivery",
                    "Biased feature importance highlighting"
                ]
            ),
            "data_representation": FairnessUserStory(
                id="FAIR-003",
                title="Representative Training Data",
                as_a="ML Engineer",
                i_want="representative and balanced training datasets",
                so_that="our models perform fairly across all user populations",
                fairness_considerations=[
                    "Demographic representation in training data",
                    "Geographic and cultural diversity",
                    "Temporal representation across different periods"
                ],
                bias_scenarios=[
                    "Underrepresented minority groups in training data",
                    "Historical bias embedded in legacy datasets",
                    "Sampling bias in data collection methods"
                ]
            )
        }

    def get_template(self, template_key: str) -> Optional[FairnessUserStory]:
        return self.templates.get(template_key)

    def get_all_templates(self) -> Dict[str, FairnessUserStory]:
        return self.templates.copy()

@dataclass
class FairnessDefinitionOfDone:
    """Mandatory fairness validation steps before deployment"""
    bias_testing_complete: bool = False
    fairness_metrics_calculated: bool = False
    demographic_parity_validated: bool = False
    explanation_consistency_verified: bool = False
    regulatory_compliance_checked: bool = False
    stakeholder_review_completed: bool = False
    documentation_updated: bool = False

    def is_deployment_ready(self) -> bool:
        return all([
            self.bias_testing_complete,
            self.fairness_metrics_calculated,
            self.demographic_parity_validated,
            self.explanation_consistency_verified,
            self.regulatory_compliance_checked,
            self.stakeholder_review_completed,
            self.documentation_updated
        ])

    def get_missing_requirements(self) -> List[str]:
        missing = []
        if not self.bias_testing_complete:
            missing.append("Bias testing")
        if not self.fairness_metrics_calculated:
            missing.append("Fairness metrics calculation")
        if not self.demographic_parity_validated:
            missing.append("Demographic parity validation")
        if not self.explanation_consistency_verified:
            missing.append("Explanation consistency verification")
        if not self.regulatory_compliance_checked:
            missing.append("Regulatory compliance check")
        if not self.stakeholder_review_completed:
            missing.append("Stakeholder review")
        if not self.documentation_updated:
            missing.append("Documentation update")
        return missing

class FairnessScrumCeremonies:
    """Adaptation guide for Scrum ceremonies with fairness integration"""

    @staticmethod
    def sprint_planning_fairness_checklist() -> List[str]:
        return [
            "Review fairness user stories in the backlog",
            "Assign fairness complexity scores to stories",
            "Identify potential bias scenarios for each story",
            "Plan bias testing activities for the sprint",
            "Allocate time for fairness metric calculation",
            "Schedule stakeholder reviews for fairness validation"
        ]

    @staticmethod
    def daily_standup_fairness_questions() -> List[str]:
        return [
            "Did any fairness issues arise in yesterday's work?",
            "Are there any fairness blockers preventing progress?",
            "What fairness validation will be completed today?",
            "Do we need fairness expertise support for any tasks?"
        ]

    @staticmethod
    def sprint_retrospective_fairness_topics() -> List[str]:
        return [
            "What fairness challenges did we encounter this sprint?",
            "How effective were our bias detection methods?",
            "Did our fairness metrics provide useful insights?",
            "What fairness processes should we improve?",
            "How can we better integrate fairness into our workflow?"
        ]

class FairAIScrumToolkit:
    """Main toolkit for integrating fairness into Scrum practices"""

    def __init__(self):
        self.user_story_library = FairnessUserStoryLibrary()
        self.ceremonies = FairnessScrumCeremonies()

    def create_fairness_sprint_backlog(self, stories: List[FairnessUserStory]) -> Dict[str, Any]:
        """Create a sprint backlog with fairness prioritization"""
        prioritized_stories = sorted(
            stories,
            key=lambda x: (x.fairness_complexity.value, x.story_points),
            reverse=True
        )

        return {
            "sprint_goal": "Deliver features with embedded fairness validation",
            "stories": prioritized_stories,
            "fairness_capacity": sum(1 for s in stories if s.fairness_complexity != RiskLevel.LOW),
            "total_story_points": sum(s.story_points for s in stories)
        }

    def generate_fairness_acceptance_criteria(self, story: FairnessUserStory) -> List[str]:
        """Generate comprehensive fairness acceptance criteria"""
        base_criteria = [
            "Bias metrics calculated and within acceptable thresholds",
            "Fairness testing completed with documented results",
            "Explanations provided for AI-driven decisions"
        ]

        if story.fairness_complexity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            base_criteria.extend([
                "Independent fairness review completed",
                "Regulatory compliance verified",
                "Stakeholder approval obtained"
            ])

        return base_criteria + story.fairness_acceptance_criteria

# ============================================================================
# COMPONENT 2: ORGANIZATIONAL INTEGRATION TOOLKIT
# ============================================================================

@dataclass
class FairnessRole:
    """Role definition for fairness governance"""
    name: str
    level: str 
    responsibilities: List[str]
    decision_authority: List[str]
    required_skills: List[str]

class FairnessGovernanceFramework:
    """Framework for organizational fairness governance"""

    def __init__(self):
        self.roles = {
            "chief_ai_officer": FairnessRole(
                name="Chief AI Officer",
                level="executive",
                responsibilities=[
                    "Set organizational AI fairness strategy",
                    "Ensure regulatory compliance",
                    "Approve high-risk AI deployments",
                    "Allocate resources for fairness initiatives"
                ],
                decision_authority=[
                    "AI system deployment approvals",
                    "Fairness policy establishment",
                    "Budget allocation for fairness tools"
                ],
                required_skills=[
                    "AI/ML leadership experience",
                    "Regulatory knowledge",
                    "Strategic planning"
                ]
            ),
            "fairness_product_manager": FairnessRole(
                name="Fairness Product Manager",
                level="management",
                responsibilities=[
                    "Translate fairness requirements into product features",
                    "Coordinate fairness across product teams",
                    "Manage fairness user story backlogs",
                    "Communicate fairness metrics to stakeholders"
                ],
                decision_authority=[
                    "Fairness feature prioritization",
                    "Acceptance criteria approval",
                    "Cross-team coordination decisions"
                ],
                required_skills=[
                    "Product management experience",
                    "AI/ML product knowledge",
                    "Stakeholder management"
                ]
            ),
            "fairness_engineer": FairnessRole(
                name="Fairness Engineer",
                level="individual_contributor",
                responsibilities=[
                    "Implement fairness metrics and testing",
                    "Develop bias detection algorithms",
                    "Create fairness monitoring dashboards",
                    "Provide technical fairness expertise"
                ],
                decision_authority=[
                    "Technical implementation decisions",
                    "Fairness metric selection",
                    "Testing strategy definition"
                ],
                required_skills=[
                    "ML/AI engineering",
                    "Bias detection techniques",
                    "Statistical analysis"
                ]
            )
        }

    def get_responsibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """Generate RACI matrix for fairness responsibilities"""
        tasks = [
            "Fairness strategy development",
            "Bias testing implementation",
            "Regulatory compliance monitoring",
            "Fairness metrics reporting",
            "Stakeholder communication",
            "Technical fairness reviews"
        ]

        matrix = {}
        for task in tasks:
            matrix[task] = {
                "Chief AI Officer": "Accountable",
                "Fairness Product Manager": "Responsible",
                "Fairness Engineer": "Consulted",
                "Development Teams": "Informed"
            }

        return matrix

@dataclass
class FairnessDecision:
    """Documentation for fairness decisions and trade-offs"""
    id: str
    timestamp: datetime
    decision: str
    rationale: str
    alternatives_considered: List[str]
    stakeholders_involved: List[str]
    trade_offs: Dict[str, str]
    impact_assessment: Dict[str, Any]
    approval_status: str
    approver: str

class FairnessDocumentationSystem:
    """System for capturing fairness decisions and accountability"""

    def __init__(self):
        self.decisions: List[FairnessDecision] = []
        self.decision_templates = {
            "bias_threshold": {
                "decision_type": "Bias Threshold Setting",
                "required_fields": ["threshold_value", "protected_groups", "metric_type"],
                "approval_level": "management"
            },
            "fairness_trade_off": {
                "decision_type": "Fairness vs Performance Trade-off",
                "required_fields": ["performance_impact", "fairness_benefit", "business_justification"],
                "approval_level": "executive"
            }
        }

    def record_decision(self, decision: FairnessDecision) -> str:
        """Record a fairness decision with full documentation"""
        decision.id = str(uuid.uuid4())
        decision.timestamp = datetime.now()
        self.decisions.append(decision)
        return decision.id

    def get_decisions_by_timeframe(self, days: int) -> List[FairnessDecision]:
        """Retrieve decisions from the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [d for d in self.decisions if d.timestamp >= cutoff]

    def generate_accountability_report(self) -> Dict[str, Any]:
        """Generate accountability report for stakeholders"""
        recent_decisions = self.get_decisions_by_timeframe(30)

        return {
            "reporting_period": "Last 30 days",
            "total_decisions": len(recent_decisions),
            "decisions_by_type": self._count_decisions_by_type(recent_decisions),
            "pending_approvals": len([d for d in recent_decisions if d.approval_status == "pending"]),
            "key_trade_offs": [d.trade_offs for d in recent_decisions if d.trade_offs]
        }

    def _count_decisions_by_type(self, decisions: List[FairnessDecision]) -> Dict[str, int]:
        """Count decisions by type for reporting"""
        counts = {}
        for decision in decisions:
            # Simple categorization based on keywords
            if "bias" in decision.decision.lower():
                counts["bias_related"] = counts.get("bias_related", 0) + 1
            elif "performance" in decision.decision.lower():
                counts["performance_related"] = counts.get("performance_related", 0) + 1
            else:
                counts["other"] = counts.get("other", 0) + 1
        return counts

class OrganizationalIntegrationToolkit:
    """Main toolkit for organizational fairness integration"""

    def __init__(self):
        self.governance_framework = FairnessGovernanceFramework()
        self.documentation_system = FairnessDocumentationSystem()

    def assess_organizational_readiness(self, org_size: str, ai_maturity: str) -> Dict[str, Any]:
        """Assess organization's readiness for fairness integration"""
        readiness_score = 0
        recommendations = []

        # Size-based assessment
        if org_size in ["large", "enterprise"]:
            readiness_score += 30
            recommendations.append("Establish dedicated fairness team")
        elif org_size == "medium":
            readiness_score += 20
            recommendations.append("Assign part-time fairness champions")
        else:
            readiness_score += 10
            recommendations.append("Start with fairness training and basic tools")

        # Maturity-based assessment
        if ai_maturity == "advanced":
            readiness_score += 40
        elif ai_maturity == "intermediate":
            readiness_score += 25
        else:
            readiness_score += 15

        return {
            "readiness_score": readiness_score,
            "readiness_level": self._categorize_readiness(readiness_score),
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(readiness_score)
        }

    def _categorize_readiness(self, score: int) -> str:
        if score >= 60:
            return "Ready for comprehensive implementation"
        elif score >= 40:
            return "Ready for phased implementation"
        else:
            return "Requires foundational work"

    def _generate_next_steps(self, score: int) -> List[str]:
        if score >= 60:
            return [
                "Begin full playbook implementation",
                "Establish governance structure",
                "Deploy advanced fairness tools"
            ]
        elif score >= 40:
            return [
                "Start with pilot teams",
                "Implement basic fairness processes",
                "Build internal expertise"
            ]
        else:
            return [
                "Conduct fairness awareness training",
                "Assess current AI systems for bias",
                "Establish basic documentation practices"
            ]

# ============================================================================
# COMPONENT 3: ADVANCED ARCHITECTURE COOKBOOK
# ============================================================================

@dataclass
class FairnessRecipe:
    """Implementation recipe for fairness in specific AI architectures"""
    name: str
    system_type: AISystemType
    domain: Domain
    bias_risks: List[str]
    fairness_techniques: List[str]
    implementation_steps: List[str]
    metrics: List[str]
    code_examples: Dict[str, str]
    validation_methods: List[str]

class AdvancedArchitectureCookbook:
    """Cookbook with fairness recipes for different AI architectures"""

    def __init__(self):
        self.recipes = {
            "llm_bias_mitigation": FairnessRecipe(
                name="LLM Bias Mitigation",
                system_type=AISystemType.LLM,
                domain=Domain.RECRUITMENT,
                bias_risks=[
                    "Gender bias in language generation",
                    "Racial stereotypes in responses",
                    "Cultural bias in reasoning",
                    "Socioeconomic bias in recommendations"
                ],
                fairness_techniques=[
                    "Prompt engineering for fairness",
                    "Bias-aware fine-tuning",
                    "Demographic parity constraints",
                    "Adversarial debiasing"
                ],
                implementation_steps=[
                    "Audit training data for demographic representation",
                    "Implement bias detection in prompt responses",
                    "Create fairness-aware evaluation datasets",
                    "Deploy bias monitoring in production"
                ],
                metrics=[
                    "Demographic parity across protected groups",
                    "Equal opportunity in positive predictions",
                    "Calibration across demographic groups",
                    "Representation fairness in generated content"
                ],
                code_examples={
                    "bias_detection": """
def detect_bias_in_llm_response(response, protected_attributes):
    bias_indicators = {
        'gender_bias': check_gender_stereotypes(response),
        'racial_bias': check_racial_stereotypes(response),
        'age_bias': check_age_stereotypes(response)
    }
    return bias_indicators
                    """,
                    "fairness_prompt": """
def create_fairness_aware_prompt(base_prompt, fairness_instructions):
    fairness_prefix = "Please ensure your response is fair and unbiased across all demographic groups. "
    return fairness_prefix + base_prompt + " " + fairness_instructions
                    """
                },
                validation_methods=[
                    "A/B testing across demographic groups",
                    "Human evaluation for bias detection",
                    "Automated bias metric calculation",
                    "Adversarial testing with biased prompts"
                ]
            ),
            "recommendation_fairness": FairnessRecipe(
                name="Fair Recommendation Systems",
                system_type=AISystemType.RECOMMENDATION,
                domain=Domain.RECRUITMENT,
                bias_risks=[
                    "Popularity bias favoring mainstream candidates",
                    "Demographic filtering effects",
                    "Historical bias reinforcement",
                    "Cold start bias for new user groups"
                ],
                fairness_techniques=[
                    "Demographic parity constraints in ranking",
                    "Equal exposure across protected groups",
                    "Fairness-aware collaborative filtering",
                    "Multi-stakeholder fairness optimization"
                ],
                implementation_steps=[
                    "Analyze recommendation distributions across demographics",
                    "Implement fairness constraints in ranking algorithms",
                    "Create demographic-aware evaluation metrics",
                    "Deploy fairness monitoring dashboard"
                ],
                metrics=[
                    "Statistical parity in recommendations",
                    "Equal opportunity for visibility",
                    "Individual fairness in similar cases",
                    "Group fairness across protected attributes"
                ],
                code_examples={
                    "fair_ranking": """
def fair_ranking(candidates, protected_attr, fairness_weight=0.3):
    # Balance relevance score with fairness constraints
    relevance_scores = calculate_relevance(candidates)
    fairness_scores = calculate_fairness(candidates, protected_attr)

    final_scores = (1 - fairness_weight) * relevance_scores + fairness_weight * fairness_scores
    return rank_by_scores(candidates, final_scores)
                    """
                },
                validation_methods=[
                    "Statistical parity testing",
                    "Equal opportunity measurement",
                    "Individual fairness validation",
                    "Long-term impact analysis"
                ]
            )
        }

    def get_recipe(self, system_type: AISystemType, domain: Domain) -> Optional[FairnessRecipe]:
        """Get appropriate recipe for system type and domain"""
        for recipe in self.recipes.values():
            if recipe.system_type == system_type and recipe.domain == domain:
                return recipe
        return None

    def get_recipes_by_system_type(self, system_type: AISystemType) -> List[FairnessRecipe]:
        """Get all recipes for a specific system type"""
        return [recipe for recipe in self.recipes.values() if recipe.system_type == system_type]

    def generate_implementation_plan(self, recipe: FairnessRecipe) -> Dict[str, Any]:
        """Generate detailed implementation plan from recipe"""
        return {
            "recipe_name": recipe.name,
            "estimated_timeline": self._estimate_timeline(recipe),
            "required_skills": self._identify_required_skills(recipe),
            "implementation_phases": self._create_implementation_phases(recipe),
            "success_criteria": recipe.metrics,
            "validation_plan": recipe.validation_methods
        }

    def _estimate_timeline(self, recipe: FairnessRecipe) -> str:
        # Simple heuristic based on number of implementation steps
        steps = len(recipe.implementation_steps)
        if steps <= 3:
            return "2-4 weeks"
        elif steps <= 6:
            return "1-2 months"
        else:
            return "2-3 months"

    def _identify_required_skills(self, recipe: FairnessRecipe) -> List[str]:
        base_skills = ["Machine Learning", "Python Programming", "Statistical Analysis"]

        if recipe.system_type == AISystemType.LLM:
            base_skills.extend(["NLP", "Transformer Models", "Prompt Engineering"])
        elif recipe.system_type == AISystemType.RECOMMENDATION:
            base_skills.extend(["Collaborative Filtering", "Ranking Algorithms"])
        elif recipe.system_type == AISystemType.COMPUTER_VISION:
            base_skills.extend(["Computer Vision", "Deep Learning", "Image Processing"])

        return base_skills

    def _create_implementation_phases(self, recipe: FairnessRecipe) -> List[Dict[str, Any]]:
        phases = [
            {
                "phase": "Assessment",
                "duration": "1 week",
                "activities": ["Analyze current system for bias", "Identify fairness requirements"],
                "deliverables": ["Bias assessment report", "Fairness requirements document"]
            },
            {
                "phase": "Implementation",
                "duration": "2-6 weeks",
                "activities": recipe.implementation_steps,
                "deliverables": ["Fairness-enhanced system", "Implementation documentation"]
            },
            {
                "phase": "Validation",
                "duration": "1-2 weeks",
                "activities": recipe.validation_methods,
                "deliverables": ["Validation report", "Fairness metrics dashboard"]
            }
        ]
        return phases

# ============================================================================
# COMPONENT 4: REGULATORY COMPLIANCE GUIDE
# ============================================================================

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement mapping"""
    framework: ComplianceFramework
    requirement_id: str
    description: str
    development_tasks: List[str]
    acceptance_criteria: List[str]
    documentation_needed: List[str]
    evidence_requirements: List[str]

class RegulatoryComplianceGuide:
    """Guide for ensuring regulatory compliance in AI fairness"""

    def __init__(self):
        self.requirements = {
            ComplianceFramework.GDPR: [
                ComplianceRequirement(
                    framework=ComplianceFramework.GDPR,
                    requirement_id="GDPR-22",
                    description="Right to explanation for automated decision-making",
                    development_tasks=[
                        "Implement explainable AI features",
                        "Create user-friendly explanation interfaces",
                        "Log all automated decisions with explanations"
                    ],
                    acceptance_criteria=[
                        "Explanations provided for all automated decisions",
                        "Explanations understandable to average users",
                        "Explanation generation time < 5 seconds"
                    ],
                    documentation_needed=[
                        "Explanation methodology documentation",
                        "User interface screenshots",
                        "Performance benchmarks"
                    ],
                    evidence_requirements=[
                        "Explanation accuracy validation",
                        "User comprehension testing results",
                        "Technical implementation details"
                    ]
                )
            ],
            ComplianceFramework.BIAS_AUDIT_NYC: [
                ComplianceRequirement(
                    framework=ComplianceFramework.BIAS_AUDIT_NYC,
                    requirement_id="NYC-BA-1",
                    description="Annual bias audit for hiring tools",
                    development_tasks=[
                        "Implement bias measurement tools",
                        "Create audit data collection systems",
                        "Develop bias reporting dashboards"
                    ],
                    acceptance_criteria=[
                        "Bias audit completed annually",
                        "Results published publicly",
                        "Remediation plans created for identified biases"
                    ],
                    documentation_needed=[
                        "Audit methodology document",
                        "Bias measurement results",
                        "Public disclosure report"
                    ],
                    evidence_requirements=[
                        "Statistical bias calculations",
                        "Demographic impact analysis",
                        "Third-party audit validation"
                    ]
                )
            ]
        }

    def get_requirements_for_framework(self, framework: ComplianceFramework) -> List[ComplianceRequirement]:
        """Get all requirements for a specific compliance framework"""
        return self.requirements.get(framework, [])

    def assess_compliance_risk(self, system_type: AISystemType, domain: Domain,
                               frameworks: List[ComplianceFramework]) -> Dict[str, Any]:
        """Assess compliance risk for a specific system"""
        risk_factors = []

        # High-risk combinations
        if domain == Domain.RECRUITMENT and system_type in [AISystemType.LLM, AISystemType.CLASSIFICATION]:
            risk_factors.append("High-impact hiring decisions")

        if domain == Domain.HEALTHCARE and system_type in [AISystemType.CLASSIFICATION, AISystemType.RECOMMENDATION]:
            risk_factors.append("Healthcare decision support")

        if domain == Domain.FINANCE and system_type == AISystemType.CLASSIFICATION:
            risk_factors.append("Financial decision automation")

        # Calculate overall risk level
        risk_score = len(risk_factors) * len(frameworks)

        if risk_score >= 6:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 4:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "applicable_frameworks": frameworks,
            "recommended_actions": self._generate_risk_actions(risk_level)
        }

    def _generate_risk_actions(self, risk_level: RiskLevel) -> List[str]:
        """Generate recommended actions based on risk level"""
        actions = {
            RiskLevel.CRITICAL: [
                "Engage legal counsel for compliance review",
                "Implement comprehensive bias testing",
                "Establish ongoing monitoring systems",
                "Create detailed audit trails"
            ],
            RiskLevel.HIGH: [
                "Implement bias detection and mitigation",
                "Create explanation systems",
                "Establish regular compliance reviews"
            ],
            RiskLevel.MEDIUM: [
                "Basic bias testing implementation",
                "Documentation of decision processes",
                "Periodic compliance assessments"
            ],
            RiskLevel.LOW: [
                "Basic fairness considerations",
                "Simple documentation requirements"
            ]
        }
        return actions.get(risk_level, [])

    def generate_compliance_checklist(self, frameworks: List[ComplianceFramework]) -> Dict[str, List[str]]:
        """Generate comprehensive compliance checklist"""
        checklist = {}

        for framework in frameworks:
            requirements = self.get_requirements_for_framework(framework)
            checklist[framework.value] = []

            for req in requirements:
                checklist[framework.value].extend([
                    f"Task: {task}" for task in req.development_tasks
                ])
                checklist[framework.value].extend([
                    f"Criteria: {criteria}" for criteria in req.acceptance_criteria
                ])
                checklist[framework.value].extend([
                    f"Evidence: {evidence}" for evidence in req.evidence_requirements
                ])

        return checklist

# ============================================================================
# INTEGRATION LAYER: FAIRNESS IMPLEMENTATION PLAYBOOK
# ============================================================================

class FairnessImplementationPlaybook:
    """Main playbook integrating all four components with workflows"""

    def __init__(self):
        self.scrum_toolkit = FairAIScrumToolkit()
        self.org_toolkit = OrganizationalIntegrationToolkit()
        self.architecture_cookbook = AdvancedArchitectureCookbook()
        self.compliance_guide = RegulatoryComplianceGuide()

    def create_implementation_workflow(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create integrated workflow combining all four components"""

        # Extract configuration
        system_type = AISystemType(project_config.get("system_type"))
        domain = Domain(project_config.get("domain"))
        org_size = project_config.get("org_size", "medium")
        compliance_frameworks = [ComplianceFramework(f) for f in project_config.get("frameworks", [])]

        # Phase 1: Organizational Assessment and Setup
        org_readiness = self.org_toolkit.assess_organizational_readiness(
            org_size, project_config.get("ai_maturity", "intermediate")
        )

        # Phase 2: Compliance Risk Assessment
        compliance_risk = self.compliance_guide.assess_compliance_risk(
            system_type, domain, compliance_frameworks
        )

        # Phase 3: Architecture Recipe Selection
        recipe = self.architecture_cookbook.get_recipe(system_type, domain)
        if not recipe:
            # Generate generic recipe if specific one not found
            recipe = self._generate_generic_recipe(system_type, domain)

        # Phase 4: Scrum Integration Planning
        fairness_stories = self._generate_fairness_stories(recipe, compliance_risk)
        sprint_backlog = self.scrum_toolkit.create_fairness_sprint_backlog(fairness_stories)

        # Create integrated workflow
        workflow = {
            "project_id": str(uuid.uuid4()),
            "project_config": project_config,
            "phases": {
                "phase_1_setup": {
                    "name": "Organizational Setup",
                    "duration": "1-2 weeks",
                    "activities": [
                        "Assess organizational readiness",
                        "Establish governance framework",
                        "Assign fairness roles and responsibilities",
                        "Set up documentation systems"
                    ],
                    "deliverables": [
                        "Organizational readiness report",
                        "Governance structure document",
                        "Role assignments",
                        "Documentation templates"
                    ],
                    "inputs": [],
                    "outputs": ["governance_framework", "role_assignments"]
                },
                "phase_2_compliance": {
                    "name": "Compliance Planning",
                    "duration": "1 week",
                    "activities": [
                        "Assess regulatory compliance requirements",
                        "Create compliance checklist",
                        "Plan audit trail systems",
                        "Design evidence collection processes"
                    ],
                    "deliverables": [
                        "Compliance risk assessment",
                        "Compliance checklist",
                        "Audit trail design",
                        "Evidence collection plan"
                    ],
                    "inputs": ["governance_framework"],
                    "outputs": ["compliance_requirements", "audit_systems"]
                },
                "phase_3_architecture": {
                    "name": "Architecture Design",
                    "duration": "2-3 weeks",
                    "activities": [
                        "Select appropriate fairness recipe",
                        "Design fairness-aware architecture",
                        "Plan bias detection systems",
                        "Create fairness monitoring framework"
                    ],
                    "deliverables": [
                        "Architecture design document",
                        "Fairness implementation plan",
                        "Bias detection strategy",
                        "Monitoring framework"
                    ],
                    "inputs": ["compliance_requirements"],
                    "outputs": ["architecture_design", "fairness_strategy"]
                },
                "phase_4_development": {
                    "name": "Agile Development with Fairness",
                    "duration": "4-8 weeks",
                    "activities": [
                        "Execute fairness-aware sprints",
                        "Implement bias detection and mitigation",
                        "Develop explanation systems",
                        "Create fairness monitoring dashboards"
                    ],
                    "deliverables": [
                        "Fairness-enhanced AI system",
                        "Bias testing results",
                        "Explanation interfaces",
                        "Monitoring dashboards"
                    ],
                    "inputs": ["architecture_design", "fairness_strategy"],
                    "outputs": ["ai_system", "fairness_validation"]
                },
                "phase_5_validation": {
                    "name": "Validation and Deployment",
                    "duration": "2-3 weeks",
                    "activities": [
                        "Execute comprehensive fairness testing",
                        "Validate compliance requirements",
                        "Conduct stakeholder reviews",
                        "Deploy with monitoring systems"
                    ],
                    "deliverables": [
                        "Fairness validation report",
                        "Compliance certification",
                        "Stakeholder approval",
                        "Production deployment"
                    ],
                    "inputs": ["ai_system", "fairness_validation", "audit_systems"],
                    "outputs": ["deployed_system", "compliance_certification"]
                }
            },
            "organizational_readiness": org_readiness,
            "compliance_assessment": compliance_risk,
            "selected_recipe": recipe.name if recipe else "Generic Recipe",
            "sprint_planning": sprint_backlog,
            "success_metrics": self._define_success_metrics(compliance_risk, recipe),
            "risk_mitigation": self._create_risk_mitigation_plan(org_readiness, compliance_risk)
        }

        return workflow

    def _generate_generic_recipe(self, system_type: AISystemType, domain: Domain) -> FairnessRecipe:
        """Generate generic fairness recipe when specific one not available"""
        return FairnessRecipe(
            name=f"Generic {system_type.value} Fairness Recipe",
            system_type=system_type,
            domain=domain,
            bias_risks=[
                "Demographic bias in predictions",
                "Historical bias reinforcement",
                "Representation bias in training data"
            ],
            fairness_techniques=[
                "Statistical parity measurement",
                "Equal opportunity analysis",
                "Bias detection testing"
            ],
            implementation_steps=[
                "Audit training data for bias",
                "Implement fairness metrics",
                "Create bias monitoring system",
                "Validate fairness across demographics"
            ],
            metrics=[
                "Statistical parity",
                "Equal opportunity",
                "Calibration across groups"
            ],
            code_examples={},
            validation_methods=[
                "Statistical testing",
                "Cross-demographic validation",
                "Expert review"
            ]
        )

    def _generate_fairness_stories(self, recipe: FairnessRecipe, compliance_risk: Dict[str, Any]) -> List[FairnessUserStory]:
        """Generate fairness user stories based on recipe and compliance requirements"""
        stories = []

        # Base fairness stories from library
        base_templates = self.scrum_toolkit.user_story_library.get_all_templates()

        for template_key, template in base_templates.items():
            story = FairnessUserStory(
                id=f"FAIR-{len(stories)+1:03d}",
                title=f"{template.title} for {recipe.system_type.value}",
                as_a=template.as_a,
                i_want=template.i_want,
                so_that=template.so_that,
                fairness_considerations=template.fairness_considerations + recipe.bias_risks,
                bias_scenarios=template.bias_scenarios,
                acceptance_criteria=template.acceptance_criteria,
                fairness_acceptance_criteria=recipe.metrics,
                story_points=5 if compliance_risk["risk_level"] == RiskLevel.HIGH else 3,
                fairness_complexity=compliance_risk["risk_level"]
            )
            stories.append(story)

        # Add recipe-specific stories
        for technique in recipe.fairness_techniques:
            story = FairnessUserStory(
                id=f"FAIR-{len(stories)+1:03d}",
                title=f"Implement {technique}",
                as_a="ML Engineer",
                i_want=f"to implement {technique} in our system",
                so_that="we can ensure fair outcomes across all user groups",
                fairness_considerations=[f"Implementation of {technique}"],
                bias_scenarios=recipe.bias_risks,
                acceptance_criteria=[f"{technique} successfully implemented and tested"],
                fairness_acceptance_criteria=recipe.metrics,
                story_points=8,
                fairness_complexity=RiskLevel.MEDIUM
            )
            stories.append(story)

        return stories

    def _define_success_metrics(self, compliance_risk: Dict[str, Any], recipe: Optional[FairnessRecipe]) -> Dict[str, Any]:
        """Define success metrics for the implementation"""
        base_metrics = [
            "Bias detection accuracy > 90%",
            "Fairness metrics within acceptable thresholds",
            "Stakeholder approval achieved",
            "Documentation completeness > 95%"
        ]

        if recipe:
            base_metrics.extend(recipe.metrics)

        if compliance_risk["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            base_metrics.extend([
                "Regulatory compliance verified",
                "Independent audit passed",
                "Legal review completed"
            ])

        return {
            "fairness_metrics": base_metrics,
            "timeline_metrics": [
                "Implementation completed on time",
                "Budget variance < 10%",
                "Quality gates met at each phase"
            ],
            "business_metrics": [
                "User satisfaction maintained",
                "Performance impact < 5%",
                "Operational efficiency maintained"
            ]
        }

    def _create_risk_mitigation_plan(self, org_readiness: Dict[str, Any], compliance_risk: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create comprehensive risk mitigation plan"""
        mitigation_plan = {
            "organizational_risks": [],
            "technical_risks": [],
            "compliance_risks": [],
            "timeline_risks": []
        }

        # Organizational risk mitigation
        if org_readiness["readiness_score"] < 40:
            mitigation_plan["organizational_risks"] = [
                "Conduct comprehensive fairness training",
                "Engage external fairness experts",
                "Start with pilot implementation",
                "Establish mentorship programs"
            ]

        # Technical risk mitigation
        mitigation_plan["technical_risks"] = [
            "Create comprehensive testing strategy",
            "Implement continuous monitoring",
            "Establish rollback procedures",
            "Plan for gradual deployment"
        ]

        # Compliance risk mitigation
        if compliance_risk["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            mitigation_plan["compliance_risks"] = [
                "Engage legal counsel early",
                "Implement comprehensive audit trails",
                "Plan for regular compliance reviews",
                "Establish emergency response procedures"
            ]

        # Timeline risk mitigation
        mitigation_plan["timeline_risks"] = [
            "Build buffer time into schedules",
            "Plan for parallel work streams",
            "Establish escalation procedures",
            "Create contingency plans"
        ]

        return mitigation_plan

# ============================================================================
# CASE STUDY: MULTI-TEAM AI RECRUITMENT PLATFORM
# ============================================================================

class RecruitmentPlatformCaseStudy:
    """Comprehensive case study for AI recruitment platform fairness implementation"""

    def __init__(self):
        self.playbook = FairnessImplementationPlaybook()

    def execute_case_study(self) -> Dict[str, Any]:
        """Execute complete case study implementation"""

        # Case study configuration
        project_config = {
            "system_type": "large_language_model",
            "domain": "recruitment",
            "org_size": "large",
            "ai_maturity": "advanced",
            "frameworks": ["bias_audit_nyc", "gdpr"],
            "teams": ["nlp_team", "backend_team", "frontend_team", "data_team"],
            "stakeholders": ["hr_leadership", "legal_team", "engineering_leadership"]
        }

        # Execute workflow
        workflow = self.playbook.create_implementation_workflow(project_config)

        # Simulate implementation execution
        execution_results = self._simulate_implementation_execution(workflow)

        # Generate lessons learned
        lessons_learned = self._generate_lessons_learned(execution_results)

        return {
            "case_study_overview": {
                "scenario": "Multi-team AI recruitment platform",
                "challenge": "Implement fair AI across resume screening, candidate matching, and interview scheduling",
                "teams_involved": project_config["teams"],
                "compliance_requirements": project_config["frameworks"]
            },
            "implementation_workflow": workflow,
            "execution_results": execution_results,
            "lessons_learned": lessons_learned,
            "success_metrics_achieved": self._calculate_success_metrics(execution_results),
            "recommendations": self._generate_recommendations(execution_results)
        }

    def _simulate_implementation_execution(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution of implementation workflow"""
        results = {}

        for phase_key, phase in workflow["phases"].items():
            phase_result = {
                "phase_name": phase["name"],
                "status": "completed",
                "duration_actual": phase["duration"],
                "deliverables_completed": len(phase["deliverables"]),
                "challenges_encountered": self._generate_phase_challenges(phase_key),
                "outcomes": self._generate_phase_outcomes(phase_key)
            }
            results[phase_key] = phase_result

        return results

    def _generate_phase_challenges(self, phase_key: str) -> List[str]:
        """Generate realistic challenges for each phase"""
        challenges = {
            "phase_1_setup": [
                "Resistance to new fairness processes",
                "Resource allocation conflicts",
                "Unclear role boundaries"
            ],
            "phase_2_compliance": [
                "Complex regulatory interpretation",
                "Conflicting requirements across frameworks",
                "Documentation overhead concerns"
            ],
            "phase_3_architecture": [
                "Technical complexity of bias detection",
                "Performance impact concerns",
                "Integration with existing systems"
            ],
            "phase_4_development": [
                "Learning curve for fairness techniques",
                "Testing complexity increases",
                "Sprint velocity impact"
            ],
            "phase_5_validation": [
                "Stakeholder alignment on metrics",
                "Validation dataset limitations",
                "Deployment coordination challenges"
            ]
        }
        return challenges.get(phase_key, [])

    def _generate_phase_outcomes(self, phase_key: str) -> List[str]:
        """Generate positive outcomes for each phase"""
        outcomes = {
            "phase_1_setup": [
                "Clear governance structure established",
                "Fairness champions identified across teams",
                "Documentation systems operational"
            ],
            "phase_2_compliance": [
                "Comprehensive compliance framework created",
                "Audit trail systems implemented",
                "Legal requirements mapped to development tasks"
            ],
            "phase_3_architecture": [
                "Bias detection architecture designed",
                "Fairness monitoring framework created",
                "Integration strategy defined"
            ],
            "phase_4_development": [
                "Fairness-enhanced AI system deployed",
                "Comprehensive bias testing completed",
                "Team fairness expertise developed"
            ],
            "phase_5_validation": [
                "Fairness metrics within target ranges",
                "Compliance requirements satisfied",
                "Stakeholder approval achieved"
            ]
        }
        return outcomes.get(phase_key, [])

    def _generate_lessons_learned(self, execution_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate lessons learned from case study execution"""
        return {
            "organizational": [
                "Early stakeholder engagement crucial for success",
                "Dedicated fairness champions accelerate adoption",
                "Clear governance prevents decision bottlenecks",
                "Training investment pays off in implementation quality"
            ],
            "technical": [
                "Bias detection complexity often underestimated",
                "Performance impact requires careful optimization",
                "Monitoring systems essential for production deployment",
                "Incremental implementation reduces risk"
            ],
            "process": [
                "Agile ceremonies effectively incorporate fairness discussions",
                "Definition of done prevents fairness shortcuts",
                "Cross-team coordination requires dedicated effort",
                "Documentation overhead justified by compliance needs"
            ],
            "compliance": [
                "Regulatory requirements drive technical architecture",
                "Audit trail design must be considered early",
                "Legal review timeline impacts deployment schedule",
                "Evidence collection automation saves significant effort"
            ]
        }

    def _calculate_success_metrics(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate achieved success metrics"""
        return {
            "implementation_success": {
                "phases_completed_on_time": 5,
                "total_phases": 5,
                "deliverables_completion_rate": 0.98,
                "stakeholder_satisfaction": 0.92
            },
            "fairness_metrics": {
                "bias_detection_accuracy": 0.94,
                "demographic_parity_achieved": True,
                "equal_opportunity_ratio": 0.96,
                "explanation_quality_score": 0.89
            },
            "business_metrics": {
                "performance_impact": -0.03,  # 3% performance decrease
                "user_satisfaction_maintained": True,
                "compliance_certification_achieved": True,
                "timeline_variance": 0.05  # 5% over planned timeline
            }
        }

    def _generate_recommendations(self, execution_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for future implementations"""
        return [
            "Invest in comprehensive fairness training before implementation begins",
            "Establish dedicated fairness team or champions early in the process",
            "Plan for 10-15% timeline buffer to accommodate fairness complexity",
            "Implement monitoring systems before production deployment",
            "Engage legal and compliance teams from project inception",
            "Create automated testing pipelines for bias detection",
            "Document all fairness decisions and trade-offs for audit purposes",
            "Plan for ongoing fairness maintenance and monitoring post-deployment"
        ]

# ============================================================================
# VALIDATION FRAMEWORK
# ============================================================================

class ValidationFramework:
    """Framework for validating fairness implementation effectiveness"""

    def __init__(self):
        self.validation_criteria = {
            "technical_effectiveness": {
                "bias_detection_accuracy": {"target": 0.90, "critical": True},
                "fairness_metric_coverage": {"target": 0.95, "critical": True},
                "monitoring_system_reliability": {"target": 0.99, "critical": True},
                "performance_impact": {"target": -0.05, "critical": False}  # Max 5% degradation
            },
            "process_effectiveness": {
                "stakeholder_satisfaction": {"target": 0.85, "critical": True},
                "team_adoption_rate": {"target": 0.80, "critical": True},
                "documentation_completeness": {"target": 0.95, "critical": True},
                "timeline_adherence": {"target": 0.90, "critical": False}
            },
            "organizational_impact": {
                "fairness_awareness_increase": {"target": 0.70, "critical": True},
                "governance_effectiveness": {"target": 0.85, "critical": True},
                "decision_accountability": {"target": 0.90, "critical": True},
                "cross_team_collaboration": {"target": 0.75, "critical": False}
            },
            "compliance_effectiveness": {
                "regulatory_requirement_coverage": {"target": 1.0, "critical": True},
                "audit_trail_completeness": {"target": 0.98, "critical": True},
                "evidence_collection_efficiency": {"target": 0.85, "critical": False},
                "legal_review_satisfaction": {"target": 0.90, "critical": True}
            }
        }

    def validate_implementation(self, implementation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate implementation effectiveness against defined criteria"""

        validation_results = {}
        overall_success = True
        critical_failures = []

        for category, criteria in self.validation_criteria.items():
            category_results = {}
            category_success = True

            for metric, requirements in criteria.items():
                actual_value = implementation_data.get(metric, 0)
                target_value = requirements["target"]
                is_critical = requirements["critical"]

                # Determine if metric passes (handle negative targets for performance impact)
                if metric == "performance_impact":
                    passed = actual_value >= target_value  # Less negative is better
                else:
                    passed = actual_value >= target_value

                category_results[metric] = {
                    "actual": actual_value,
                    "target": target_value,
                    "passed": passed,
                    "critical": is_critical
                }

                if not passed and is_critical:
                    critical_failures.append(f"{category}.{metric}")
                    category_success = False
                    overall_success = False

            validation_results[category] = {
                "metrics": category_results,
                "category_success": category_success
            }

        return {
            "overall_success": overall_success,
            "critical_failures": critical_failures,
            "validation_results": validation_results,
            "recommendations": self._generate_validation_recommendations(validation_results),
            "next_steps": self._generate_next_steps(critical_failures, validation_results)
        }

    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        for category, results in validation_results.items():
            failed_metrics = [
                metric for metric, data in results["metrics"].items()
                if not data["passed"]
            ]

            if failed_metrics:
                if category == "technical_effectiveness":
                    recommendations.extend([
                        "Improve bias detection algorithms and validation methods",
                        "Enhance monitoring system reliability and coverage",
                        "Optimize performance while maintaining fairness"
                    ])
                elif category == "process_effectiveness":
                    recommendations.extend([
                        "Increase stakeholder engagement and communication",
                        "Provide additional training and support for teams",
                        "Streamline documentation and process workflows"
                    ])
                elif category == "organizational_impact":
                    recommendations.extend([
                        "Strengthen fairness governance and accountability",
                        "Improve cross-team collaboration mechanisms",
                        "Enhance fairness awareness programs"
                    ])
                elif category == "compliance_effectiveness":
                    recommendations.extend([
                        "Review and enhance compliance coverage",
                        "Improve audit trail and evidence collection",
                        "Strengthen legal and regulatory alignment"
                    ])

        return list(set(recommendations))  # Remove duplicates

    def _generate_next_steps(self, critical_failures: List[str], validation_results: Dict[str, Any]) -> List[str]:
        """Generate specific next steps for addressing failures"""
        if not critical_failures:
            return ["Continue monitoring and maintain current implementation"]

        next_steps = [
            f"Immediate attention required for: {', '.join(critical_failures)}",
            "Conduct root cause analysis for critical failures",
            "Develop remediation plan with specific timelines",
            "Re-validate after remediation implementation"
        ]

        return next_steps

    def create_validation_dashboard(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create dashboard view of validation results"""

        dashboard = {
            "summary": {
                "overall_status": "PASS" if validation_results["overall_success"] else "FAIL",
                "critical_failures_count": len(validation_results["critical_failures"]),
                "categories_passed": sum(1 for r in validation_results["validation_results"].values() if r["category_success"]),
                "total_categories": len(validation_results["validation_results"])
            },
            "category_scores": {},
            "trending_metrics": self._calculate_trending_metrics(),
            "action_items": validation_results["recommendations"]
        }

        # Calculate category scores
        for category, results in validation_results["validation_results"].items():
            passed_metrics = sum(1 for m in results["metrics"].values() if m["passed"])
            total_metrics = len(results["metrics"])
            score = passed_metrics / total_metrics if total_metrics > 0 else 0

            dashboard["category_scores"][category] = {
                "score": score,
                "status": "PASS" if results["category_success"] else "FAIL",
                "passed_metrics": passed_metrics,
                "total_metrics": total_metrics
            }

        return dashboard

    def _calculate_trending_metrics(self) -> Dict[str, str]:
        """Calculate trending indicators for key metrics"""
        # Placeholder for trending calculation
        return {
            "bias_detection_accuracy": "improving",
            "stakeholder_satisfaction": "stable",
            "compliance_coverage": "improving",
            "team_adoption": "stable"
        }

# ============================================================================
# ADAPTABILITY GUIDELINES
# ============================================================================

class AdaptabilityGuidelines:
    """Guidelines for adapting the playbook across different domains and problem types"""

    def __init__(self):
        self.domain_adaptations = {
            Domain.HEALTHCARE: {
                "specific_considerations": [
                    "Patient safety as primary concern",
                    "Medical ethics compliance",
                    "Health equity requirements",
                    "FDA/regulatory approval processes"
                ],
                "bias_risks": [
                    "Treatment disparities across demographics",
                    "Diagnostic accuracy variations",
                    "Access to care inequities"
                ],
                "compliance_frameworks": [ComplianceFramework.GDPR, ComplianceFramework.NIST_AI_RMF],
                "validation_requirements": [
                    "Clinical validation studies",
                    "Health outcome monitoring",
                    "Medical ethics review"
                ]
            },
            Domain.FINANCE: {
                "specific_considerations": [
                    "Financial regulatory compliance",
                    "Credit fairness requirements",
                    "Anti-discrimination laws",
                    "Systemic risk management"
                ],
                "bias_risks": [
                    "Credit scoring disparities",
                    "Insurance pricing discrimination",
                    "Investment recommendation bias"
                ],
                "compliance_frameworks": [ComplianceFramework.GDPR, ComplianceFramework.CCPA],
                "validation_requirements": [
                    "Regulatory stress testing",
                    "Fair lending analysis",
                    "Consumer protection validation"
                ]
            },
            Domain.EDUCATION: {
                "specific_considerations": [
                    "Student privacy protection",
                    "Educational equity requirements",
                    "Developmental appropriateness",
                    "Family consent processes"
                ],
                "bias_risks": [
                    "Academic assessment bias",
                    "Resource allocation inequities",
                    "Career guidance stereotyping"
                ],
                "compliance_frameworks": [ComplianceFramework.GDPR],
                "validation_requirements": [
                    "Educational outcome studies",
                    "Student development monitoring",
                    "Family stakeholder review"
                ]
            }
        }

        self.system_type_adaptations = {
            AISystemType.CLASSIFICATION: {
                "fairness_techniques": [
                    "Threshold optimization across groups",
                    "Calibration-based fairness",
                    "Equalized odds enforcement"
                ],
                "metrics": [
                    "Demographic parity",
                    "Equal opportunity",
                    "Predictive parity"
                ],
                "implementation_focus": [
                    "Decision boundary fairness",
                    "Feature importance analysis",
                    "Prediction confidence calibration"
                ]
            },
            AISystemType.REGRESSION: {
                "fairness_techniques": [
                    "Error rate equalization",
                    "Residual analysis across groups",
                    "Fairness-constrained optimization"
                ],
                "metrics": [
                    "Mean squared error parity",
                    "Residual distribution analysis",
                    "Prediction interval fairness"
                ],
                "implementation_focus": [
                    "Error distribution analysis",
                    "Prediction accuracy across groups",
                    "Uncertainty quantification fairness"
                ]
            }
        }

    def adapt_playbook(self, target_domain: Domain, target_system_type: AISystemType,
                       base_workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt the base playbook for specific domain and system type"""

        # Get domain-specific adaptations
        domain_adaptations = self.domain_adaptations.get(target_domain, {})
        system_adaptations = self.system_type_adaptations.get(target_system_type, {})

        # Create adapted workflow
        adapted_workflow = base_workflow.copy()

        # Update phases with domain-specific considerations
        for phase_key, phase in adapted_workflow["phases"].items():
            # Add domain-specific activities
            if domain_adaptations.get("specific_considerations"):
                phase["activities"].extend([
                    f"Consider {consideration}" for consideration in domain_adaptations["specific_considerations"][:2]
                ])

            # Add system-type specific activities
            if system_adaptations.get("implementation_focus") and phase_key == "phase_3_architecture":
                phase["activities"].extend(system_adaptations["implementation_focus"])

        # Update compliance requirements
        if domain_adaptations.get("compliance_frameworks"):
            adapted_workflow["compliance_frameworks"] = domain_adaptations["compliance_frameworks"]

        # Update fairness techniques and metrics
        if system_adaptations.get("fairness_techniques"):
            adapted_workflow["recommended_techniques"] = system_adaptations["fairness_techniques"]
            adapted_workflow["recommended_metrics"] = system_adaptations["metrics"]

        # Add domain-specific validation requirements
        if domain_adaptations.get("validation_requirements"):
            adapted_workflow["validation_requirements"] = domain_adaptations["validation_requirements"]

        # Update risk assessment
        adapted_workflow["domain_specific_risks"] = domain_adaptations.get("bias_risks", [])

        return {
            "adapted_workflow": adapted_workflow,
            "adaptation_summary": {
                "target_domain": target_domain.value,
                "target_system_type": target_system_type.value,
                "domain_considerations": domain_adaptations.get("specific_considerations", []),
                "system_considerations": system_adaptations.get("implementation_focus", []),
                "compliance_updates": domain_adaptations.get("compliance_frameworks", []),
                "validation_updates": domain_adaptations.get("validation_requirements", [])
            },
            "implementation_guidance": self._generate_adaptation_guidance(domain_adaptations, system_adaptations)
        }

    def _generate_adaptation_guidance(self, domain_adaptations: Dict[str, Any],
                                      system_adaptations: Dict[str, Any]) -> List[str]:
        """Generate specific guidance for adapted implementation"""
        guidance = []

        if domain_adaptations:
            guidance.append(f"Focus on domain-specific considerations: {', '.join(domain_adaptations.get('specific_considerations', [])[:3])}")
            guidance.append(f"Pay special attention to bias risks: {', '.join(domain_adaptations.get('bias_risks', [])[:2])}")

        if system_adaptations:
            guidance.append(f"Implement system-specific fairness techniques: {', '.join(system_adaptations.get('fairness_techniques', [])[:2])}")
            guidance.append(f"Monitor system-specific metrics: {', '.join(system_adaptations.get('metrics', [])[:2])}")

        guidance.extend([
            "Engage domain experts early in the implementation process",
            "Validate adaptations through pilot implementations",
            "Document adaptation decisions and rationales",
            "Plan for domain-specific stakeholder reviews"
        ])

        return guidance

    def generate_cross_domain_insights(self) -> Dict[str, Any]:
        """Generate insights about adapting across different domains"""
        return {
            "common_patterns": [
                "All domains require stakeholder engagement strategies",
                "Regulatory compliance drives architectural decisions",
                "Domain expertise essential for effective bias identification",
                "Validation requirements vary significantly by domain impact"
            ],
            "adaptation_complexity": {
                "low": [Domain.EDUCATION.value],
                "medium": [Domain.RECRUITMENT.value, Domain.FINANCE.value],
                "high": [Domain.HEALTHCARE.value, Domain.CRIMINAL_JUSTICE.value]
            },
            "cross_domain_lessons": [
                "Privacy requirements are universal but implementation varies",
                "Explanation needs differ based on user technical sophistication",
                "Bias types are domain-specific but detection techniques transferable",
                "Governance structures must reflect domain regulatory environment"
            ]
        }

# ============================================================================
# FUTURE ITERATIONS AND IMPROVEMENT INSIGHTS
# ============================================================================

class PlaybookEvolutionInsights:
    """Insights on how the playbook could be improved through future iterations"""

    def __init__(self):
        self.current_limitations = [
            "Limited automated bias detection capabilities",
            "Manual compliance mapping processes",
            "Static fairness thresholds across contexts",
            "Reactive rather than proactive bias prevention",
            "Limited real-time fairness monitoring",
            "Insufficient integration with MLOps pipelines"
        ]

        self.improvement_opportunities = {
            "automation": {
                "priority": "high",
                "description": "Increased automation of fairness processes",
                "specific_improvements": [
                    "Automated bias detection in CI/CD pipelines",
                    "Dynamic fairness threshold adjustment",
                    "Automated compliance requirement mapping",
                    "Real-time bias monitoring and alerting"
                ],
                "estimated_impact": "30-50% reduction in manual effort",
                "implementation_timeline": "6-12 months"
            },
            "ai_integration": {
                "priority": "high",
                "description": "AI-powered fairness assistance",
                "specific_improvements": [
                    "ML-based bias pattern recognition",
                    "Automated fairness metric selection",
                    "Intelligent compliance requirement analysis",
                    "Predictive bias risk assessment"
                ],
                "estimated_impact": "Improved accuracy and coverage",
                "implementation_timeline": "12-18 months"
            },
            "real_time_capabilities": {
                "priority": "medium",
                "description": "Real-time fairness monitoring and adjustment",
                "specific_improvements": [
                    "Streaming bias detection",
                    "Dynamic model adjustment based on fairness drift",
                    "Real-time stakeholder alerting",
                    "Continuous compliance validation"
                ],
                "estimated_impact": "Faster response to fairness issues",
                "implementation_timeline": "9-15 months"
            },
            "advanced_metrics": {
                "priority": "medium",
                "description": "More sophisticated fairness metrics",
                "specific_improvements": [
                    "Intersectional fairness metrics",
                    "Long-term fairness impact measurement",
                    "Causal fairness analysis",
                    "Multi-stakeholder fairness optimization"
                ],
                "estimated_impact": "More comprehensive fairness assessment",
                "implementation_timeline": "12-24 months"
            }
        }

    def generate_evolution_roadmap(self) -> Dict[str, Any]:
        """Generate roadmap for playbook evolution"""

        # Prioritize improvements
        high_priority = [k for k, v in self.improvement_opportunities.items() if v["priority"] == "high"]
        medium_priority = [k for k, v in self.improvement_opportunities.items() if v["priority"] == "medium"]

        roadmap = {
            "version_2.0": {
                "timeline": "6-12 months",
                "focus": "Automation and AI Integration",
                "key_improvements": [
                    self.improvement_opportunities[area]["description"]
                    for area in high_priority
                ],
                "expected_benefits": [
                    "Reduced manual effort in fairness implementation",
                    "Improved accuracy of bias detection",
                    "Faster implementation cycles"
                ]
            },
            "version_3.0": {
                "timeline": "12-24 months",
                "focus": "Advanced Analytics and Real-time Capabilities",
                "key_improvements": [
                    self.improvement_opportunities[area]["description"]
                    for area in medium_priority
                ],
                "expected_benefits": [
                    "Proactive bias prevention",
                    "Sophisticated fairness analysis",
                    "Real-time fairness assurance"
                ]
            },
            "version_4.0": {
                "timeline": "24+ months",
                "focus": "Ecosystem Integration and Industry Standards",
                "key_improvements": [
                    "Integration with major ML platforms",
                    "Industry-standard fairness protocols",
                    "Cross-organizational fairness coordination",
                    "Regulatory automation interfaces"
                ],
                "expected_benefits": [
                    "Industry-wide fairness standardization",
                    "Seamless regulatory compliance",
                    "Cross-organizational fairness coordination"
                ]
            }
        }

        return roadmap

    def identify_research_needs(self) -> Dict[str, Any]:
        """Identify research areas needed for future improvements"""
        return {
            "technical_research": [
                "Causal fairness in complex AI systems",
                "Intersectional bias detection algorithms",
                "Fairness in federated learning environments",
                "Dynamic fairness threshold optimization"
            ],
            "methodological_research": [
                "Multi-stakeholder fairness reconciliation",
                "Long-term fairness impact measurement",
                "Fairness-performance trade-off optimization",
                "Cultural adaptation of fairness concepts"
            ],
            "empirical_research": [
                "Effectiveness studies of fairness interventions",
                "Industry adoption patterns and barriers",
                "Cost-benefit analysis of fairness implementations",
                "User experience with fairness explanations"
            ],
            "policy_research": [
                "Harmonization of fairness regulations across jurisdictions",
                "Liability frameworks for AI fairness",
                "International fairness standards development",
                "Fairness in AI procurement processes"
            ]
        }

    def generate_improvement_recommendations(self) -> List[str]:
        """Generate specific recommendations for playbook improvement"""
        return [
            "Establish partnerships with research institutions for advanced fairness techniques",
            "Create open-source community around fairness implementation tools",
            "Develop industry-specific fairness certification programs",
            "Build comprehensive fairness implementation case study database",
            "Establish fairness implementation maturity assessment framework",
            "Create automated fairness testing and validation platforms",
            "Develop fairness-aware MLOps integration standards",
            "Establish cross-industry fairness implementation working groups"
        ]

# ============================================================================
# MAIN IMPLEMENTATION GUIDE AND USAGE EXAMPLES
# ============================================================================

class PlaybookImplementationGuide:
    """Comprehensive guide for using the Fairness Implementation Playbook"""

    def __init__(self):
        self.playbook = FairnessImplementationPlaybook()
        self.case_study = RecruitmentPlatformCaseStudy()
        self.validation_framework = ValidationFramework()
        self.adaptability_guidelines = AdaptabilityGuidelines()
        self.evolution_insights = PlaybookEvolutionInsights()

    def get_implementation_guide(self) -> Dict[str, Any]:
        """Get comprehensive implementation guide"""
        return {
            "overview": {
                "purpose": "Systematic approach to implementing fairness in AI systems",
                "scope": "End-to-end fairness integration across development lifecycle",
                "target_users": ["AI teams", "Product managers", "Compliance officers", "Engineering leadership"],
                "expected_outcomes": ["Fair AI systems", "Regulatory compliance", "Organizational fairness capability"]
            },
            "getting_started": {
                "prerequisites": [
                    "Basic understanding of AI/ML concepts",
                    "Familiarity with agile development practices",
                    "Organizational commitment to fairness",
                    "Access to relevant stakeholders"
                ],
                "initial_steps": [
                    "Assess organizational readiness",
                    "Identify key stakeholders",
                    "Define fairness goals and requirements",
                    "Select appropriate implementation approach"
                ],
                "quick_start_checklist": [
                    "□ Complete organizational readiness assessment",
                    "□ Identify compliance requirements",
                    "□ Select system type and domain",
                    "□ Generate implementation workflow",
                    "□ Begin Phase 1 activities"
                ]
            },
            "key_decision_points": {
                "organizational_structure": {
                    "decision": "Centralized vs distributed fairness responsibility",
                    "considerations": [
                        "Organization size and complexity",
                        "AI system distribution across teams",
                        "Existing governance structures",
                        "Resource availability"
                    ],
                    "guidance": "Large organizations benefit from centralized expertise with distributed implementation"
                },
                "implementation_approach": {
                    "decision": "Big bang vs phased implementation",
                    "considerations": [
                        "Organizational risk tolerance",
                        "Resource constraints",
                        "System criticality",
                        "Stakeholder readiness"
                    ],
                    "guidance": "Phased approach recommended for most organizations"
                },
                "fairness_metrics": {
                    "decision": "Which fairness metrics to prioritize",
                    "considerations": [
                        "System type and use case",
                        "Stakeholder values and priorities",
                        "Regulatory requirements",
                        "Technical feasibility"
                    ],
                    "guidance": "Start with statistical parity and equal opportunity, expand based on domain needs"
                }
            },
            "common_pitfalls": [
                "Underestimating organizational change management needs",
                "Focusing only on technical solutions without process changes",
                "Insufficient stakeholder engagement and buy-in",
                "Treating fairness as one-time implementation vs ongoing process",
                "Ignoring performance-fairness trade-offs in planning",
                "Inadequate documentation and audit trail preparation"
            ],
            "success_factors": [
                "Strong leadership commitment and sponsorship",
                "Clear governance structure and accountability",
                "Comprehensive stakeholder engagement",
                "Adequate resource allocation and timeline planning",
                "Focus on both technical and process improvements",
                "Continuous monitoring and improvement mindset"
            ]
        }

    def demonstrate_full_workflow(self) -> Dict[str, Any]:
        """Demonstrate complete workflow with example project"""

        # Example project configuration
        example_config = {
            "system_type": "classification",
            "domain": "recruitment",
            "org_size": "medium",
            "ai_maturity": "intermediate",
            "frameworks": ["bias_audit_nyc", "gdpr"],
            "project_timeline": "3 months",
            "team_size": 8,
            "budget_available": "$200,000"
        }

        print("=== FAIRNESS IMPLEMENTATION PLAYBOOK DEMONSTRATION ===\n")

        # Step 1: Generate workflow
        print("STEP 1: Generating Implementation Workflow")
        workflow = self.playbook.create_implementation_workflow(example_config)
        print(f"✓ Workflow created with {len(workflow['phases'])} phases")
        print(f"✓ Organizational readiness: {workflow['organizational_readiness']['readiness_level']}")
        print(f"✓ Compliance risk level: {workflow['compliance_assessment']['risk_level'].value}")
        print()

        # Step 2: Execute case study
        print("STEP 2: Executing Recruitment Platform Case Study")
        case_study_results = self.case_study.execute_case_study()
        print(f"✓ Case study completed with {len(case_study_results['lessons_learned'])} lesson categories")
        print(f"✓ Success metrics: {case_study_results['success_metrics_achieved']['implementation_success']['deliverables_completion_rate']:.1%} completion rate")
        print()

        # Step 3: Validate implementation
        print("STEP 3: Validating Implementation Effectiveness")

        # Mock implementation data for validation
        mock_implementation_data = {
            "bias_detection_accuracy": 0.92,
            "fairness_metric_coverage": 0.96,
            "monitoring_system_reliability": 0.98,
            "performance_impact": -0.04,
            "stakeholder_satisfaction": 0.88,
            "team_adoption_rate": 0.85,
            "documentation_completeness": 0.97,
            "timeline_adherence": 0.93,
            "fairness_awareness_increase": 0.75,
            "governance_effectiveness": 0.87,
            "decision_accountability": 0.92,
            "cross_team_collaboration": 0.78,
            "regulatory_requirement_coverage": 1.0,
            "audit_trail_completeness": 0.99,
            "evidence_collection_efficiency": 0.86,
            "legal_review_satisfaction": 0.91
        }

        validation_results = self.validation_framework.validate_implementation(mock_implementation_data)
        print(f"✓ Validation completed: {'PASS' if validation_results['overall_success'] else 'FAIL'}")
        print(f"✓ Categories passed: {sum(1 for r in validation_results['validation_results'].values() if r['category_success'])}/4")
        print()

        # Step 4: Generate adaptability guidance
        print("STEP 4: Generating Adaptability Guidance")
        adaptation_example = self.adaptability_guidelines.adapt_playbook(
            Domain.HEALTHCARE, AISystemType.CLASSIFICATION, workflow
        )
        print(f"✓ Adaptation generated for {adaptation_example['adaptation_summary']['target_domain']} domain")
        print(f"✓ Added {len(adaptation_example['adaptation_summary']['domain_considerations'])} domain considerations")
        print()

        # Step 5: Future evolution insights
        print("STEP 5: Generating Future Evolution Insights")
        evolution_roadmap = self.evolution_insights.generate_evolution_roadmap()
        print(f"✓ Evolution roadmap created with {len(evolution_roadmap)} future versions")
        research_needs = self.evolution_insights.identify_research_needs()
        print(f"✓ Identified {sum(len(areas) for areas in research_needs.values())} research areas")
        print()

        print("=== DEMONSTRATION COMPLETE ===")

        return {
            "workflow": workflow,
            "case_study": case_study_results,
            "validation": validation_results,
            "adaptation_example": adaptation_example,
            "evolution_insights": {
                "roadmap": evolution_roadmap,
                "research_needs": research_needs
            },
            "implementation_guide": self.get_implementation_guide()
        }


#======================================================================
# --- WIDGETS, SIMULATIONS, and TOOLKIT/PAGE FUNCTIONS ---
#======================================================================

def get_llm_response_groq(api_key: str, question: str) -> str | None:
    """
    Gets a response from the Groq API for a given question.

    Args:
        api_key: The Groq API key.
        question: The user's question.

    Returns:
        The response from the LLM as a string, or None if an error occurs.
    """
    try:
        client = Groq(api_key=api_key)
        
        # System prompt to define the AI's role and context.
        system_prompt = """
        You are an expert AI assistant for a technical playbook. 
        Your role is to provide clear, concise, and helpful guidance based on the playbook's principles.
        When a user asks a question, answer it directly and helpfully.
        The playbook covers topics like pre-processing, in-processing, and post-processing for machine learning, 
        the Causal Toolkit, and explaining fairness trade-offs to stakeholders.
        Keep your answers focused on these topics.
        """

        completion = client.chat.completions.create(
            # Using a recommended model available on Groq
            model="openai/gpt-oss-20b", 
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=False,  # Set to False to get the full response at once
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")
        return None

def ai_playbook_assistant():
    """An AI assistant to guide users through the playbook."""
    st.sidebar.divider()
    st.sidebar.header("🤖 AI Playbook Assistant")
    st.sidebar.markdown("Get help navigating and applying the playbook.")

    questions = {
        "--- Select a question ---": "Please choose a question from the dropdown.",
        "Where should I start?": "Always start with the **Causal Analysis Toolkit**. Understanding *why* bias might exist in your specific context is the most critical first step. Your findings here will guide all subsequent decisions in the other toolkits.",
        "What's the difference between pre-, in-, and post-processing?": "It's about *when* you intervene.\n\n- **Pre-processing**: You fix the **data** before training (e.g., rebalancing).\n- **In-processing**: You change the **algorithm** during training to be fairness-aware.\n- **Post-processing**: You adjust the model's **predictions** after training.",
        "When should I use the Causal Toolkit?": "Use the **Causal Toolkit** at the very beginning of your project, before you write any mitigation code. It's a strategic tool for project planning, risk assessment, and stakeholder discussions. It helps you build a shared understanding of the problem.",
        "How do I explain fairness trade-offs to my manager?": "Use the concept of a **Pareto Frontier**, shown in the In-processing toolkit. Explain it like this: 'We can't always maximize both accuracy and fairness simultaneously. This chart shows us the set of best possible models. We can choose a model that is slightly less accurate but significantly fairer. Our job is to pick the right balance for our company's values and legal requirements.'",
        "Which mitigation technique is best?": "There's no single best technique. The right choice depends on your answers to these questions:\n\n1.  **What is the source of bias?** (Causal Analysis)\n2.  **Can you modify the data?** (If yes, consider Pre-processing)\n3.  **Can you change the model's training?** (If yes, consider In-processing)\n4.  **Do you only have access to the model's predictions?** (If yes, use Post-processing)",
    }

    question = st.sidebar.selectbox("How can I help you?", list(questions.keys()), key="ai_assist_question")
    st.sidebar.info(questions[question])

def research_assistant():
    """A tool to help users search for external resources."""
    st.sidebar.divider()
    st.sidebar.header("🔍 Research Assistant")
    query = st.sidebar.text_input("Search for a fairness concept...", placeholder="e.g., Adversarial Debiasing")
    if st.sidebar.button("Search Web", key="web_search_button"):
        if query:
            encoded_query = quote(query)
            google_url = f"https://www.google.com/search?q={encoded_query}"
            scholar_url = f"https://scholar.google.com/scholar?q={encoded_query}"
            st.sidebar.markdown(f"[Search on Google]({google_url})", unsafe_allow_html=True)
            st.sidebar.markdown(f"[Search on Google Scholar]({scholar_url})", unsafe_allow_html=True)
        else:
            st.sidebar.warning("Please enter a search term.")
# Set the page configuration for a wider layout

# Add a title to your Streamlit app
st.title("👩‍💻 Developer's Guide to the Fairness Intervention Framework")
st.write("Click the link below to open the developer's guide in a new tab.")

# The URL of the website you want to embed
documentation_url = "https://www.canva.com/design/DAGyppdoWK4/ltpUa4raME22W25ne6PSRg/view?utm_content=DAGyppdoWK4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=ha5f7eedea9"
st.markdown(f"""
<a href="{documentation_url}" target="_blank">
    Developer's Guide to the Fairness Intervention Framework
</a>
""", unsafe_allow_html=True)

st.info("The link will open in a new browser tab.")
# Use st.components.v1.iframe to embed the website
def run_threshold_simulation():
    st.markdown("##### Simulation: Achieving Equal Opportunity")
    st.write("Adjust the decision thresholds for two demographic groups. Your goal is to make the **True Positive Rate (TPR)** as equal as possible for both groups, which satisfies the Equal Opportunity fairness criterion.")
    np.random.seed(42)
    scores_a_pos = np.random.normal(0.7, 0.15, 80)
    scores_a_neg = np.random.normal(0.4, 0.15, 120)
    scores_b_pos = np.random.normal(0.6, 0.15, 50)
    scores_b_neg = np.random.normal(0.3, 0.15, 150)
    df_a = pd.DataFrame({'Score': np.concatenate([scores_a_pos, scores_a_neg]), 'Actual': [1]*80 + [0]*120})
    df_b = pd.DataFrame({'Score': np.concatenate([scores_b_pos, scores_b_neg]), 'Actual': [1]*50 + [0]*150})
    col1, col2 = st.columns(2)
    with col1:
        threshold_a = st.slider("Threshold for Group A", 0.0, 1.0, 0.55, key="sim_thresh_a")
    with col2:
        threshold_b = st.slider("Threshold for Group B", 0.0, 1.0, 0.45, key="sim_thresh_b")
    tpr_a = np.mean(df_a[df_a['Actual'] == 1]['Score'] >= threshold_a)
    fpr_a = np.mean(df_a[df_a['Actual'] == 0]['Score'] >= threshold_a)
    tpr_b = np.mean(df_b[df_b['Actual'] == 1]['Score'] >= threshold_b)
    fpr_b = np.mean(df_b[df_b['Actual'] == 0]['Score'] >= threshold_b)
    tpr_diff = abs(tpr_a - tpr_b)
    st.markdown("###### Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric(label="TPR (Group A)", value=f"{tpr_a:.2%}")
        st.metric(label="FPR (Group A)", value=f"{fpr_a:.2%}")
    with res_col2:
        st.metric(label="TPR (Group B)", value=f"{tpr_b:.2%}")
        st.metric(label="FPR (Group B)", value=f"{fpr_b:.2%}")
    with res_col3:
        st.metric(label="TPR Difference", value=f"{tpr_diff:.2%}")
        if tpr_diff < 0.02:
            st.success("Fairness Goal Met!")
        else:
            st.warning("Adjust Thresholds")

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
    """Renders an interactive simulation for propensity score matching."""
    st.write("This simulation demonstrates how matching can reduce bias from a confounding variable (age).")
    np.random.seed(0)
    
    # Generate synthetic data
    n = 200
    # Confounder: age. The treated group is older on average.
    age_control = np.random.normal(45, 10, n)
    age_treat = np.random.normal(55, 10, n)
    
    # Outcome: salary. Affected by age (confounder) and treatment.
    # The true treatment effect is +$5,000.
    outcome_control = 20000 + 1000 * age_control + np.random.normal(0, 5000, n)
    outcome_treat = 20000 + 1000 * age_treat + 5000 + np.random.normal(0, 5000, n) 
    
    data_control = pd.DataFrame({'group': 'Control', 'age': age_control, 'salary': outcome_control})
    data_treat = pd.DataFrame({'group': 'Treated', 'age': age_treat, 'salary': outcome_treat})
    
    df = pd.concat([data_control, data_treat]).reset_index(drop=True)

    # Raw difference
    raw_diff = df[df['group'] == 'Treated']['salary'].mean() - df[df['group'] == 'Control']['salary'].mean()
    st.metric("Naive (Biased) Effect", f"${raw_diff:,.2f}", "Higher salary for treated group")
    st.write("This estimate is biased because the treated group is older on average, and age also increases salary.")

    # Matching
    if st.button("Perform Matching to Control for Age"):
        control_X = data_control[['age']]
        treat_X = data_treat[['age']]
        
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_X)
        distances, indices = nn.kneighbors(treat_X)
        
        # Get the matched control units
        matched_control = data_control.iloc[indices.flatten()]
        
        # The matched effect is the difference in means between treated and their matched controls
        matched_diff = data_treat['salary'].mean() - matched_control['salary'].mean()
        st.metric("Matched (Less Biased) Effect", f"${matched_diff:,.2f}", f"True effect is ~$5,000")
        st.write("After matching individuals with similar ages, the estimated effect is much closer to the true causal effect we built into the data.")

def run_rd_simulation():
    """Renders an interactive simulation for Regression Discontinuity."""
    st.write("This simulation shows how a 'jump' at a cutoff point can reveal a causal effect.")
    np.random.seed(42)
    
    # Generate synthetic data
    x = np.linspace(0, 100, 200)
    cutoff = 50
    # Treatment is applied if x >= 50
    treatment = (x >= cutoff).astype(int)
    # Outcome has a baseline trend and a jump of +15 at the cutoff
    y = 10 + 0.5 * x + 15 * treatment + np.random.normal(0, 5, 200)
    
    df = pd.DataFrame({'score': x, 'outcome': y, 'group': np.where(treatment, 'Treated', 'Control')})

    # `type='quantitative'` argument. This is more robust across different Altair versions.
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('score', type='quantitative', title='Eligibility Score'),
        y=alt.Y('outcome', type='quantitative', title='Outcome (e.g., Earnings)'),
        color=alt.Color('group', type='nominal', title='Group')
    ).properties(
        title="Outcome vs. Score with a Cutoff at 50"
    )
    
    st.altair_chart(chart, use_container_width=True)
    st.write("Imagine a scholarship is awarded to students with a score of 50 or more. We can see a clear jump in the 'outcome' right at the cutoff. This jump is the estimated causal effect of the scholarship, as students just below and just above the cutoff are assumed to be very similar in all other respects.")

def run_did_simulation():
    """Renders an interactive simulation for Difference-in-Differences."""
    st.write("This simulation shows how comparing trends before and after a treatment reveals its causal effect.")
    
    # Generate data
    df = pd.DataFrame({
        'group': ['Control', 'Control', 'Treated', 'Treated'],
        'time': ['Before', 'After', 'Before', 'After'],
        'outcome': [100, 110, 105, 135] # Control goes up by 10, Treated goes up by 30
    })
    
    chart = alt.Chart(df).mark_line(point=True, size=3).encode(
        x=alt.X('time', type='nominal', title='Time Period'),
        y=alt.Y('outcome', type='quantitative', title='Outcome'),
        color=alt.Color('group', type='nominal', title='Group'),
        strokeDash=alt.StrokeDash('group', type='nominal', title='Group')
    ).properties(
        title="Outcome Change Over Time for Treated and Control Groups"
    )
    st.altair_chart(chart, use_container_width=True)

    # Calculate DiD
    control_before = 100
    control_after = 110
    treat_before = 105
    treat_after = 135
    
    diff_control = control_after - control_before
    diff_treat = treat_after - treat_before
    did_effect = diff_treat - diff_control

    st.markdown(f"**Calculations:**")
    st.markdown(f"- Change in Control group (the 'natural' trend): `{control_after} - {control_before} = {diff_control}`")
    st.markdown(f"- Change in Treated group: `{treat_after} - {treat_before} = {diff_treat}`")
    st.metric("Difference-in-Differences (Estimated Causal Effect)", did_effect)
    st.write("The control group's trend is subtracted from the treated group's trend. The remaining difference is the estimated causal effect of the treatment.")

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
        Before mitigating, an engineer must hypothesize the causal pathways that lead to unfair outcomes.
        This toolkit helps you map out real-world biases that your data might reflect.
        """)
    with st.expander("🔍 **Source & Further Reading**"):
        st.markdown(
            """
            - **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference.* A foundational text on causal modeling.
            - **Kusner, M. J., et al. (2017).** *Counterfactual Fairness.* Introduces a definition of fairness based on what would have happened in a counterfactual world.
            """
        )    
    if 'causal_report' not in st.session_state:
        st.session_state.causal_report = {}

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Discrimination Mechanisms", "Counterfactual Fairness", "Causal Diagramming",
            "Causal Inference", "Intersectionality", "Mitigation Strategies"
        ])
    # TAB 1: Identification
    with tab1:
        st.subheader("Framework for Identifying Discrimination Mechanisms")
        st.info("Identify the possible root causes of bias in your application.")
        
        with st.expander("Definition: Direct Discrimination"):
            st.write("""Occurs when a protected attribute (such as race or gender) is explicitly used to make a decision. This is the most obvious type of bias.
                     Case Study Example: A past loan model explicitly used gender as a variable, assigning a lower weight to female applicants. This is a direct causal path.""")
        with st.expander("📚 Suggested Reading"):
             st.markdown("""
             **Paper**: [Big Data's Disparate Impact](https://www.californialawreview.org/print/big-datas-disparate-impact/)  
             **Authors**: Barocas, S., & Selbst, A. D. (2016).   
             **Relevance**: This paper provides a foundational explanation of the legal and technical distinction between **disparate treatment** (direct discrimination) and **disparate impact** (indirect discrimination) in the context of algorithms.""")

        st.text_area("1. Does the protected attribute directly influence the decision?", 
                     placeholder="Example: A hiring model explicitly assigns lower scores to female applicants.", 
                     key="causal_q1")      
        with st.expander("Definition: Indirect Discrimination"):
             st.write("""Occurs when a protected attribute affects an intermediate factor that is legitimate for the decision. The bias is transmitted through this mediating variable.
                     Case Study Example: Gender influences career interruptions (e.g., for childcare), which reduces 'years of experience'. The model then penalizes lower experience, indirectly discriminating against women.""")
        with st.expander("📚 Suggested Reading"):    
             st.markdown("""
             **Paper**: [Counterfactual Fairness](https://arxiv.org/abs/1703.06856)  
             **Authors**: Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017).   
             ****Relevance**: This paper introduces a formal causal model for fairness. It directly addresses indirect discrimination by asking if a decision would be the same in a "counterfactual world" where the person's protected attribute was different but all other independent factors were unchanged.""")  
        st.text_area("2. Does the protected attribute affect legitimate intermediate factors?", 
                     placeholder="Example: Gender can influence career breaks (for childcare), and the model penalizes these breaks, indirectly affecting women.", 
                     key="causal_q2")
        with st.expander("Definition: Proxy Discrimination"):
            st.write("""Occurs when a seemingly neutral variable is so correlated with a protected attribute that it acts as a substitute (a 'proxy').
                     Case Study Example: In our city, zip code is a strong proxy for race due to historical segregation. Using zip code in a loan model can perpetuate this bias.""")
        with st.expander("📚 Suggested Reading"):
            st.markdown("""
             **Paper**: [Algorithmic transparency via quantitative input influence](https://arxiv.org/abs/1606.00219)  
             **Authors**: Datta, A., Sen, S., & Zick, Y. (2016).   
             ****Relevance**: This work offers a technical approach for detecting proxies. It presents methods to measure the influence of each input variable on a model's outcome, which can reveal when a feature like zip code is having an outsized, proxy-like effect.""")  
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
            st.info("Practical methods for estimating causal effects when observational data is imperfect or limited.")
            
            with st.expander("🔍 Definition: Matching"):
                st.write("Compare individuals from a 'treatment' group with very similar individuals from a 'control' group. By comparing statistical 'twins' on key covariates, the treatment effect is isolated. In fairness, the 'treatment' may be belonging to a demographic group.")
            with st.expander("💡 Interactive Example: Matching Simulation"):
                run_matching_simulation()

            with st.expander("🔍 Definition: Instrumental Variables (IV)"):
                st.write("Use an 'instrument' variable that affects the 'treatment' (e.g., a protected attribute) but not the outcome directly, except through the treatment. This helps untangle correlation from causation when unobserved confounders are present.")
                st.graphviz_chart("""
                digraph {
                    rankdir=LR;
                    Z [label="Instrument (Z)"];
                    A [label="Protected Attribute (A)"];
                    Y [label="Outcome (Y)"];
                    U [label="Unobserved Confounder (U)", style=dashed, fontcolor=gray];
                    Z -> A;
                    A -> Y;
                    U -> A [style=dashed, color=gray];
                    U -> Y [style=dashed, color=gray];
                }
                """)
                st.write("**Example:** To measure the causal effect of education (A) on income (Y) while accounting for unobserved ability (U), 'proximity to a university' (Z) can be an instrument. Proximity affects education but not income directly.")

            with st.expander("🔍 Definition: Regression Discontinuity (RD)"):
                st.write("Takes advantage of a sharp threshold or cutoff in treatment assignment. By comparing those just above and below the cutoff, the treatment effect can be estimated, assuming these individuals are otherwise very similar.")
            with st.expander("💡 Interactive Example: RD Simulation"):
                run_rd_simulation()

            with st.expander("🔍 Definition: Difference-in-Differences (DiD)"):
                st.write("Compares the change in outcomes over time between a treatment group and a control group. The 'difference in the differences' between groups before and after treatment estimates the causal effect, controlling for trends that would have happened anyway.")
            with st.expander("💡 Interactive Example: DiD Simulation"):
                run_did_simulation()
    # TAB 5: Intersectionality
        with tab5:
            st.subheader("Applying an Intersectional Perspective to Causal Analysis")
            st.info("Recognize that the causes and effects of bias can be unique to overlapping identities.")
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
                        node [shape=box, style=rounded];
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
                        node [shape=box, style=rounded];
                        subgraph cluster_0 {
                            label = "Intersectional Identity";
                            style=filled;
                            color=lightgrey;
                            "Black Woman";
                        }
                        "Black Woman" -> "Access to Prof. Networks" [label="Unique Path"];
                        "Access to Prof. Networks" -> "Decision";
                        "Gender" -> "Years of Experience" -> "Decision";
                        "Race" -> "Type of Education" -> "Decision";
                    }
                    """)
                st.info("The intersectional model reveals a new causal path ('Access to Professional Networks') that specifically affects the 'Black Woman' subgroup—a factor that simplistic, single-axis models would completely ignore.")

            st.text_area(
                "Apply to your case: What unique causal paths might affect intersectional subgroups in your system?", 
                placeholder="Example: In our lending system, the interaction of 'being a woman' and 'living in a rural area' creates a unique causal path through 'lack of history with large banks', which does not affect other groups in the same way.", 
                key="causal_intersectional"
            )
        
        with tab6:
            st.subheader("Intervention & Mitigation Strategies")
            st.info("Select and document the strategies you will use to address the biases identified in the previous steps.")

            # --- Pre-processing ---
            st.markdown("##### 1. Pre-processing (Data-Level Interventions)")
            with st.expander("🔍 Definition: Reweighing"):
                st.markdown("""
                Adjusts the weight of samples in the training data to balance the representation of different demographic groups. The goal is to make the model pay more attention to the under-represented or disadvantaged groups during training.
                - **Pros:** Model-agnostic, directly addresses data imbalance.
                - **Cons:** Can distort the original data distribution, potentially affecting overall model accuracy.
                """)
            with st.expander("💡 Interactive Example: Reweighing Simulation"):
                st.write("This simulation shows how sample weights can influence a simple decision boundary.")
                # Create sample data
                group_a_size = st.slider("Size of Group A (Privileged)", 50, 200, 150)
                group_b_size = st.slider("Size of Group B (Disadvantaged)", 10, 100, 30)
                
                # Assign weights
                weight_a = 1.0
                weight_b = st.slider("Weight for Group B", 1.0, 10.0, 5.0, help="Higher weights make the model treat Group B samples as more important.")

                # Simple logic for outcome
                outcome = "Balanced" if (group_b_size * weight_b) / (group_a_size * weight_a) > 0.8 else "Biased towards Group A"
                st.metric("Model Outcome with Current Weights", outcome)
                st.write(f"Effective representation ratio (Group B / Group A): **{((group_b_size * weight_b) / (group_a_size * weight_a)):.2f}**")
                st.info("By increasing the weight for Group B, you force the model to treat it as more prevalent, leading to a more balanced outcome even when the group is smaller in number.")

            # --- In-processing ---
            st.markdown("##### 2. In-processing (Model-Level Interventions)")
            with st.expander("🔍 Definition: Adversarial Debiasing"):
                st.markdown("""
                Trains two models simultaneously: a main **Predictor** model that tries to predict the outcome, and an **Adversary** model that tries to predict the protected attribute from the Predictor's output. The Predictor is trained to maximize its accuracy while also "fooling" the Adversary.
                - **Pros:** Can achieve a good balance between fairness and accuracy.
                - **Cons:** Increases training complexity significantly, can be unstable to train.
                """)
                st.graphviz_chart("""
                digraph {
                    rankdir=LR;
                    "Input Data" -> "Predictor Model" -> "Prediction (Y)";
                    "Predictor Model" -> "Adversary Model" [label="tries to predict Prot. Attr."];
                    "Adversary Model" -> "Predictor Model" [label="sends back loss to penalize"];
                }
                """)

            # --- Post-processing ---
            st.markdown("##### 3. Post-processing (Prediction-Level Interventions)")
            with st.expander("🔍 Definition: Calibrated Equalized Odds"):
                st.markdown("""
                Adjusts the model's output scores (probabilities) for different groups to satisfy a fairness constraint, like equal opportunity or equalized odds, without retraining the model. This is often done by setting different classification thresholds for each group.
                - **Pros:** Easy to implement, model-agnostic (works on black-box models).
                - **Cons:** Does not fix the underlying bias in the model; it only corrects the outputs.
                """)
            with st.expander("💡 Interactive Example: Threshold Adjustment"):
                st.write("Adjust decision thresholds to see how they impact approval rates and fairness.")
                
                # Simulate some scores
                np.random.seed(10)
                scores_group_a = np.random.normal(0.6, 0.15, 100)
                scores_group_b = np.random.normal(0.45, 0.15, 100)

                thresh_a = st.slider("Threshold for Group A", 0.0, 1.0, 0.5)
                thresh_b = st.slider("Threshold for Group B", 0.0, 1.0, 0.5)

                approved_a = np.sum(scores_group_a >= thresh_a)
                approved_b = np.sum(scores_group_b >= thresh_b)

                col1, col2 = st.columns(2)
                col1.metric("Group A Approval Rate", f"{approved_a}%")
                col2.metric("Group B Approval Rate", f"{approved_b}%")

                if abs(approved_a - approved_b) < 5:
                    st.success("The approval rates are now nearly equal (Demographic Parity is high).")
                else:
                    st.warning("The approval rates are unequal.")
                st.info("Notice that to achieve equal approval rates, you may need to set a lower threshold for the group with the lower average score distribution.")

            st.text_area(
                "Document Your Chosen Mitigation Strategy",
                placeholder="Example: We will apply Reweighing during pre-processing to give more importance to applicants from underrepresented zip codes. We will monitor for a 10% drop in overall accuracy.",
                key="mitigation_doc"
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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Define a specific, measurable fairness rule that your model must follow.
            **Step 1: Choose a Fairness Metric**
            Select a metric that aligns with your fairness goals (e.g., Demographic Parity, Equalized Odds).
            **Step 2: Formulate the Constraint**
            Express the metric as a mathematical constraint. For Demographic Parity, the constraint is that the difference in the rate of positive outcomes between groups should be below a small threshold, ε.
            `|P(Ŷ=1|A=0) - P(Ŷ=1|A=1)| ≤ ε`
            **Step 3: Implement as a Penalty**
            In your code, create a function that calculates this disparity. This function will be your `C(θ)` in the Lagrangian formula.
            **Python Example (Conceptual):**
            ```python
            def demographic_parity_constraint(y_pred, sensitive_features):
                # P(Y_hat=1 | A=1)
                rate_privileged = y_pred[sensitive_features==1].mean()
                # P(Y_hat=1 | A=0)
                rate_unprivileged = y_pred[sensitive_features==0].mean()
                
                # The penalty is the difference
                return abs(rate_privileged - rate_unprivileged)

            # During training loop:
            # loss = original_loss + lambda * demographic_parity_constraint(...)
            ```
            """)

        st.markdown("**Feasibility and Trade-offs:**")
        with st.expander("🔍 Definition and Example"):
            st.write("It is not always possible to be perfectly fair and perfectly accurate at the same time. Often there is a 'trade-off'. Improving fairness can slightly reduce overall accuracy, and vice versa. It’s crucial to understand this balance.")
            st.write("**Intersectionality Example:** Forcing equal outcomes for all subgroups (e.g., Latina women, Asian men) may be mathematically impossible or require such a large sacrifice in accuracy that the model becomes unusable.")
        st.text_area("Apply to your case: What trade-off between accuracy and fairness are you willing to accept?", key="in_q2")
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Quantify and document the impact of fairness interventions on model performance.
            **Step 1: Establish Baselines**
            Train a model *without* fairness constraints. Record its accuracy, precision, recall, and fairness metrics. This is your baseline.
            **Step 2: Train Constrained Models**
            Train several versions of your model, each with a different fairness constraint strength (i.e., varying the `λ` parameter).
            **Step 3: Plot the Trade-off Curve**
            Create a scatter plot with the fairness metric on one axis and the accuracy metric on the other. Each point represents one of your trained models. This visualizes the Pareto frontier.
            **Step 4: Make a Decision**
            Based on the plot, select the model that offers the best compromise for your specific use case. For example, you might accept a 2% drop in accuracy to achieve a 50% reduction in fairness disparity. Document this decision and its rationale.
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Build a neural network architecture for adversarial training.
            **Method:** You need two distinct models that share some layers.
            1.  **Predictor Model:** Takes the input data and produces the main prediction.
            2.  **Adversary Model:** Takes the *output* (or an intermediate representation) from the predictor and tries to predict the sensitive attribute.
            **Python Example (PyTorch):**
            ```python
            import torch.nn as nn

            class Predictor(nn.Module):
                def __init__(self):
                    super(Predictor, self).__init__()
                    self.layer1 = nn.Linear(10, 32)
                    self.layer2 = nn.Linear(32, 1) # Predicts the main outcome
                def forward(self, x):
                    x = torch.relu(self.layer1(x))
                    return self.layer2(x)

            class Adversary(nn.Module):
                def __init__(self):
                    super(Adversary, self).__init__()
                    self.layer1 = nn.Linear(1, 16) # Takes predictor's output
                    self.layer2 = nn.Linear(16, 1) # Predicts sensitive attr
                def forward(self, x):
                    x = torch.relu(self.layer1(x))
                    return torch.sigmoid(self.layer2(x))
            
            predictor = Predictor()
            adversary = Adversary()
            ```
            """)

        st.markdown("**Optimization:**")
        with st.expander("🔍 Definition and Example"):
            st.write("Training can be unstable because the Predictor and Adversary have opposing objectives. Special techniques, like 'gradient reversal', are needed so the Predictor actively 'unlearns' bias.")
        st.text_area("Apply to your case: What optimization challenges do you foresee and how would you address them?", 
                     placeholder="E.g.: The adversary could become too strong at the start. We will use a gradual increase in its weight in the loss function.", 
                     key="in_q4")
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Train the predictor and adversary in a stable way.
            **Method:** The key is the **Gradient Reversal Layer (GRL)**. During the backward pass of training, this "layer" reverses the gradient flowing from the adversary to the predictor. This means that while the adversary is learning to get *better* at predicting the sensitive attribute, the predictor receives a signal to get *worse* at providing that information.
            **Python Example (Conceptual Training Loop):**
            ```python
            # alpha is the adversarial strength parameter
            for data, labels, sensitive in dataloader:
                # 1. Train the predictor
                predictor_optimizer.zero_grad()
                predictions = predictor(data)
                predictor_loss = predictor_criterion(predictions, labels)
                
                # 2. Train the adversary
                # Detach to avoid gradients flowing back to predictor here
                adversary_preds = adversary(predictions.detach())
                adversary_loss = adversary_criterion(adversary_preds, sensitive)
                adversary_loss.backward()
                adversary_optimizer.step()

                # 3. Train predictor to fool adversary (with gradient reversal)
                adversary_preds_for_predictor = adversary(predictions)
                adversary_loss_for_predictor = adversary_criterion(adversary_preds_for_predictor, sensitive)
                
                # The "trick": multiply by -alpha to reverse the gradient
                total_predictor_loss = predictor_loss - alpha * adversary_loss_for_predictor
                total_predictor_loss.backward()
                predictor_optimizer.step()
            ```
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Define and track multiple, sometimes competing, goals for your model.
            **Step 1: Define Your Objective Functions**
            Create separate Python functions for each objective you want to track.
            - `objective_1 = calculate_accuracy(y_true, y_pred)`
            - `objective_2 = calculate_demographic_parity(y_pred, sensitive_features)`
            - `objective_3 = calculate_equalized_odds(y_true, y_pred, sensitive_features)`
            **Step 2: Choose a Multi-Objective Algorithm**
            Specialized algorithms are needed to find the Pareto front. These are often found in libraries dedicated to multi-objective optimization.
            - **Scalarization:** The simplest method. Combine all objectives into a single formula with weights: `Loss = w1*Acc + w2*Fair1 + w3*Fair2`. By trying many different weights, you can trace out the Pareto front.
            - **Evolutionary Algorithms (e.g., NSGA-II):** More advanced methods that evolve a population of models to find the optimal set directly. Libraries like `DEAP` or `Pymoo` in Python can implement this.
            **Step 3: Document the Chosen Solution**
            Once you have the Pareto front, select a final model and document *why* you chose that specific trade-off point.
            """)

    # TAB 4: Code Patterns
    with tab4:
        st.subheader("Implementation Pattern Catalog")
        with st.expander("🔍 Friendly Definition"):
            st.write("These are code or pseudocode snippets showing how in-processing techniques look in practice. They serve as reusable templates for implementing fairness in your own code.")
        
        st.markdown("**Pattern 1: Regularized Loss Function**")
        st.code("""
# Example of a fairness-regularized loss function in a training loop
# (Works with any framework: TF, PyTorch, Scikit-learn)

lambda_fairness = 0.8 # Hyperparameter to tune

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(data)
    
    # Calculate standard performance loss
    performance_loss = standard_loss_function(predictions, labels)
    
    # Calculate fairness penalty
    fairness_penalty = demographic_parity_disparity(predictions, sensitive_attrs)
    
    # Combine the losses
    total_loss = performance_loss + lambda_fairness * fairness_penalty
    
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()
        """, language="python")

        st.markdown("**Pattern 2: Custom Fairlearn Reductions Model**")
        st.code("""
# Using the Fairlearn library for exponentiated gradient reduction
# This method finds a stochastic classifier that satisfies the constraint.
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# 1. Define the base estimator (your model)
estimator = LogisticRegression(solver='liblinear')

# 2. Define the fairness constraint
constraint = DemographicParity()

# 3. Create the fairness-aware model using the reduction
mitigator = ExponentiatedGradient(estimator, constraint)

# 4. Fit the model as usual
mitigator.fit(X_train, y_train, sensitive_features=A_train)

# 5. Make predictions
predictions = mitigator.predict(X_test)
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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Extend your fairness constraints to cover multiple intersectional subgroups.
            **Step 1: Define Intersectional Subgroups**
            Create a new feature in your data that represents the intersection of two or more sensitive attributes.
            ```python
            # Example: Creating an intersectional feature
            df['race_gender'] = df['race'] + '_' + df['gender']
            # Now you have groups like 'Black_Woman', 'White_Man', etc.
            ```
            **Step 2: Modify Your Constraint Function**
            Your fairness constraint function needs to calculate the disparity across *all* pairs of these new subgroups.
            ```python
            def max_intersectional_disparity(y_pred, intersectional_features):
                subgroup_rates = {}
                for group in intersectional_features.unique():
                    mask = (intersectional_features == group)
                    subgroup_rates[group] = y_pred[mask].mean()
                
                # The penalty is the difference between the max and min rates
                if not subgroup_rates:
                    return 0
                max_rate = max(subgroup_rates.values())
                min_rate = min(subgroup_rates.values())
                return max_rate - min_rate

            # Use this new function in your regularized loss
            # loss = perf_loss + lambda * max_intersectional_disparity(...)
            ```
            **Step 3: Monitor Trade-offs**
            Be aware that enforcing fairness across many small subgroups can be very difficult and may significantly impact overall accuracy. Track performance for each subgroup individually.
            """)


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
        with st.expander("Implementation Guide: Your Step-by-Step Manual"):
            st.markdown("""
            This guide is your primary resource for putting the playbook into action. It provides a practical, hands-on walkthrough of the entire audit process, from initial setup to final reporting. Think of it as a detailed roadmap for your team.

            **Who is this for?** Project managers, engineers, data scientists, and anyone directly involved in executing the fairness audit.
            
            **What you'll find inside:**
            - **Actionable Checklists:** Step-by-step instructions for each phase of the playbook.
            - **Key Decision Points:** Clear explanations of critical choices you'll need to make (e.g., selecting fairness metrics), including the pros and cons of each option.
            - **Supporting Evidence:** The rationale and research behind our recommendations, so you can confidently justify your decisions.
            - **Risk Mitigation:** A proactive look at common pitfalls and identified risks, with strategies to help you avoid them.

            ➡️ **Start here if you are ready to begin your fairness audit and need to know exactly what to do.**
            """)
       
        with st.expander("Implementation Guide"):
            st.markdown("""
            This guide is your primary resource for putting the playbook into action. It provides a practical, hands-on walkthrough of the entire audit process, from initial setup to final reporting. Think of it as a detailed roadmap for your team.

            **Who is this for?** Project managers, engineers, data scientists, and anyone directly involved in executing the fairness audit.
            
            **What you'll find inside:**
            - **Actionable Checklists:** Step-by-step instructions for each phase of the playbook.
            - **Key Decision Points:** Clear explanations of critical choices you'll need to make (e.g., selecting fairness metrics), including the pros and cons of each option.
            - **Supporting Evidence:** The rationale and research behind our recommendations, so you can confidently justify your decisions.
            - **Risk Mitigation:** A proactive look at common pitfalls and identified risks, with strategies to help you avoid them.

            ➡️ **Start here if you are ready to begin your fairness audit and need to know exactly what to do.**
            """)
       
        with st.expander("Case Study"):
            st.markdown("""
            See the playbook in action from start to finish. This case study demonstrates a real-world application of the audit process on a typical fairness problem. It connects the dots, showing how the output of one component directly informs the next, creating a cohesive and impactful analysis.

            **Who is this for?** Anyone who wants to understand how the playbook's concepts translate into tangible results. It's especially useful for team training and stakeholder buy-in.
            
            **What you'll find inside:**
            - **An End-to-End Walkthrough:** Follow a hypothetical project from problem definition to the final validation of fairness interventions.
            - **Concrete Examples:** See sample code, visualizations, and report snippets that you can adapt for your own work.
            - **Narrative Insights:** Understand the "why" behind each step, learning from the challenges and successes of the example project.

            ➡️ **Read this to see a complete, practical example before you start your own implementation or to explain the process to others.**
            """)
        
        with st.expander("Validation Framework: Measuring Your Success"):
            st.markdown("""
            How do you know your fairness audit was effective? This framework provides the tools and guidance to answer that question. It helps your team verify that the audit process was implemented correctly and is achieving its intended goals.

            **Who is this for?** Team leads, quality assurance specialists, and compliance managers responsible for verifying the integrity and effectiveness of the audit.
            
            **What you'll find inside:**
            - **Success Metrics:** Quantitative and qualitative criteria for evaluating the effectiveness of your audit process.
            - **Verification Checklists:** A set of questions and observable evidence to confirm that each component of the playbook was executed properly.
            - **Benchmarking Guidance:** Advice on how to compare your results over time and against industry standards to ensure continuous improvement.

            ➡️ **Use this framework after your audit to measure its impact, report on its effectiveness, and build confidence in your results.**
            """)
        
        with st.expander("Intersectional Fairness"):
            st.markdown("""
            Fairness is not one-dimensional. This section explains our commitment to intersectionality—the principle that individuals can face overlapping and interdependent systems of discrimination based on multiple factors (e.g., race, gender, age). This isn't a separate step; it's a critical lens applied throughout every component of the playbook.

            **Who is this for?** All users of the playbook. It is essential for conducting a thorough and meaningful fairness analysis.
            
            **How it's integrated:**
            - **Data Analysis:** Guidance on identifying and analyzing subgroups at the intersection of different demographic axes.
            - **Metric Selection:** Recommendations for using metrics that can capture intersectional disparities.
            - **Throughout the Playbook:** Look for callout boxes and specific subsections that explicitly address intersectional considerations within the Implementation Guide and Case Study.
            
            ➡️ **Keep this principle in mind as you navigate all sections to ensure your audit addresses the complex, multifaceted nature of fairness.**
            """)
      
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

# ... [Paste your other correct simulation and helper functions here] ...
# run_calibration_simulation, run_rejection_simulation, run_matching_simulation,
# run_rd_simulation, run_did_simulation, etc.
# These functions appeared to be correct and are omitted for brevity.


# --- PLAYBOOK PAGE FUNCTIONS ---

def audit_playbook():
    st.title("Fairness Audit Playbook")
    st.sidebar.title("Audit Playbook Navigation")
    page = st.sidebar.radio("Go to", [
        "How to Navigate this Playbook",
        "Historical Context Assessment",
        "Fairness Definition Selection",
        "Bias Source Identification",
        "Comprehensive Fairness Metrics"
    ], key="audit_nav") # This key is now safe because this function is only called once.
    # ... [Rest of your audit_playbook function code] ...
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


def intervention_playbook():
    st.title("Fairness Intervention Playbook")
    st.sidebar.title("Intervention Playbook Navigation")
    selection = st.sidebar.radio(
        "Go to:",
        ["Main Playbook", "Causal Toolkit", "Pre-processing Toolkit", "In-processing Toolkit", "Post-processing Toolkit"],
        key="intervention_nav" # This key is now safe.
    )

    if selection == "Main Playbook":
        st.header("📖 Fairness Intervention Playbook")
        st.info("This playbook integrates the four toolkits into a cohesive workflow, guiding developers from bias identification to the implementation of effective solutions.")
        
        with st.expander("Implementation Guide: Your Step-by-Step Manual"):
            st.markdown("""
            This guide is your primary resource for putting the playbook into action. It provides a practical, hands-on walkthrough of the entire audit process, from initial setup to final reporting. Think of it as a detailed roadmap for your team.
            **Who is this for?** Project managers, engineers, data scientists, and anyone directly involved in executing the fairness audit.
            **What you'll find inside:**
            - **Actionable Checklists:** Step-by-step instructions for each phase of the playbook.
            - **Key Decision Points:** Clear explanations of critical choices you'll need to make (e.g., selecting fairness metrics), including the pros and cons of each option.
            - **Supporting Evidence:** The rationale and research behind our recommendations, so you can confidently justify your decisions.
            - **Risk Mitigation:** A proactive look at common pitfalls and identified risks, with strategies to help you avoid them.
            ➡️ **Start here if you are ready to begin your fairness audit and need to know exactly what to do.**
            """)

        with st.expander("Case Study: Theory into Practice"):
            st.markdown("""
            See the playbook in action from start to finish. This case study demonstrates a real-world application of the audit process on a typical fairness problem. It connects the dots, showing how the output of one component directly informs the next, creating a cohesive and impactful analysis.
            **Who is this for?** Anyone who wants to understand how the playbook's concepts translate into tangible results. It's especially useful for team training and stakeholder buy-in.
            **What you'll find inside:**
            - **An End-to-End Walkthrough:** Follow a hypothetical project from problem definition to the final validation of fairness interventions.
            - **Concrete Examples:** See sample code, visualizations, and report snippets that you can adapt for your own work.
            - **Narrative Insights:** Understand the "why" behind each step, learning from the challenges and successes of the example project.
            ➡️ **Read this to see a complete, practical example before you start your own implementation or to explain the process to others.**
            """)

        with st.expander("Validation Framework: Measuring Your Success"):
            st.markdown("""
            How do you know your fairness audit was effective? This framework provides the tools and guidance to answer that question. It helps your team verify that the audit process was implemented correctly and is achieving its intended goals.
            **Who is this for?** Team leads, quality assurance specialists, and compliance managers responsible for verifying the integrity and effectiveness of the audit.
            **What you'll find inside:**
            - **Success Metrics:** Quantitative and qualitative criteria for evaluating the effectiveness of your audit process.
            - **Verification Checklists:** A set of questions and observable evidence to confirm that each component of the playbook was executed properly.
            - **Benchmarking Guidance:** Advice on how to compare your results over time and against industry standards to ensure continuous improvement.
            ➡️ **Use this framework after your audit to measure its impact, report on its effectiveness, and build confidence in your results.**
            """)

        with st.expander("Guiding Principle: Intersectional Fairness"):
            st.markdown("""
            Fairness is not one-dimensional. This section explains our commitment to intersectionality—the principle that individuals can face overlapping and interdependent systems of discrimination based on multiple factors (e.g., race, gender, age). This isn't a separate step; it's a critical lens applied throughout every component of the playbook.
            **Who is this for?** All users of the playbook. It is essential for conducting a thorough and meaningful fairness analysis.
            **How it's integrated:**
            - **Data Analysis:** Guidance on identifying and analyzing subgroups at the intersection of different demographic axes.
            - **Metric Selection:** Recommendations for using metrics that can capture intersectional disparities.
            - **Throughout the Playbook:** Look for callout boxes and specific subsections that explicitly address intersectional considerations within the Implementation Guide and Case Study.
            ➡️ **Keep this principle in mind as you navigate all sections to ensure your audit addresses the complex, multifaceted nature of fairness.**
            """)
     # ... [Rest of your 'Main Playbook' section code] ...
    elif selection == "Causal Toolkit":
        causal_fairness_toolkit()
    elif selection == "Pre-processing Toolkit":
        preprocessing_fairness_toolkit()
    elif selection == "In-processing Toolkit":
        inprocessing_fairness_toolkit()
    elif selection == "Post-processing Toolkit":
        postprocessing_fairness_toolkit()
    # which would have caused a NameError. The main router handles this now.

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

def fairness_implementation_playbook():
    st.title("Fairness Implementation Playbook")
    # ... [Rest of your fairness_implementation_playbook function code] ...
    # FIX: Implemented the UI for this function.
    """Función principal del Fairness Implementation Playbook"""
    st.header("🛡️ Fairness Implementation Playbook")

    # Inicializar componentes
    if 'playbook' not in st.session_state:
        st.session_state.playbook = FairnessImplementationPlaybook()
        st.session_state.implementation_guide = PlaybookImplementationGuide()
        st.session_state.validation_framework = ValidationFramework()

    with st.expander("🔍 About the Fairness Implementation Playbook"):
        st.write("""
        **The Fairness Implementation Playbook** integrates four comprehensive components to ensure
        systematic fairness implementation across your AI development lifecycle:

        1. **Fair AI Scrum Toolkit** - Integrates fairness into agile development
        2. **Organizational Integration Toolkit** - Establishes governance and accountability
        3. **Advanced Architecture Cookbook** - Provides technical implementation recipes
        4. **Regulatory Compliance Guide** - Ensures legal and policy compliance
        """)

    # Navegación por tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Quick Start", "📋 Project Configuration", "📊 Implementation Dashboard",
        "📚 Case Study", "✅ Validation", "🔄 Adaptability"
    ])

    with tab1:
        quick_start_guide()

    with tab2:
        project_configuration()

    with tab3:
        implementation_dashboard()

    with tab4:
        case_study_demo()

    with tab5:
        validation_interface()

    with tab6:
        adaptability_interface()


def quick_start_guide():
    """Guía de inicio rápido"""
    st.subheader("🚀 Quick Start Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prerequisites")
        prerequisites = [
            "Basic understanding of AI/ML concepts",
            "Familiarity with agile development practices",
            "Organizational commitment to fairness",
            "Access to relevant stakeholders"
        ]

        for i, prereq in enumerate(prerequisites, 1):
            checked = st.checkbox(f"{prereq}", key=f"prereq_{i}")
            if checked:
                st.success("✅ Complete")

    with col2:
        st.markdown("### Quick Assessment")

        org_size = st.selectbox(
            "Organization Size:",
            ["small", "medium", "large", "enterprise"],
            key="org_size_selector"
        )

        ai_maturity = st.selectbox(
            "AI Maturity Level:",
            ["beginner", "intermediate", "advanced"],
            key="ai_maturity_selector"
        )

        if st.button("Assess Readiness"):
            playbook = st.session_state.playbook
            assessment = playbook.org_toolkit.assess_organizational_readiness(org_size, ai_maturity)

            st.metric("Readiness Score", f"{assessment['readiness_score']}/70")
            st.info(f"**Status:** {assessment['readiness_level']}")

            st.markdown("**Next Steps:**")
            for step in assessment['next_steps']:
                st.write(f"• {step}")

def project_configuration():
    """Configuración del proyecto"""
    st.subheader("📋 Project Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### System Details")

        system_type = st.selectbox(
            "AI System Type:",
            [e.value for e in AISystemType],
            help="Select the type of AI system you're implementing"
        )

        domain = st.selectbox(
            "Application Domain:",
            [e.value for e in Domain],
            help="Select the domain where your AI system will be deployed"
        )

        org_size = st.selectbox(
            "Organization Size:",
            ["small", "medium", "large", "enterprise"]
        )

        ai_maturity = st.selectbox(
            "AI Maturity:",
            ["beginner", "intermediate", "advanced"]
        )

    with col2:
        st.markdown("### Compliance Requirements")

        frameworks = st.multiselect(
            "Compliance Frameworks:",
            [f.value for f in ComplianceFramework],
            help="Select applicable compliance frameworks"
        )

        timeline = st.selectbox(
            "Project Timeline:",
            ["1-2 months", "3-4 months", "5-6 months", "6+ months"]
        )

        team_size = st.number_input(
            "Team Size:",
            min_value=1,
            max_value=50,
            value=8
        )

        budget = st.number_input(
            "Budget (USD):",
            min_value=10000,
            max_value=1000000,
            value=200000,
            step=10000
        )

    if st.button("Generate Implementation Workflow", type="primary"):
        project_config = {
            "system_type": system_type,
            "domain": domain,
            "org_size": org_size,
            "ai_maturity": ai_maturity,
            "frameworks": frameworks,
            "project_timeline": timeline,
            "team_size": team_size,
            "budget_available": budget
        }

        # Generar workflow
        workflow = st.session_state.playbook.create_implementation_workflow(project_config)
        st.session_state.current_workflow = workflow

        st.success("✅ Implementation Workflow Generated!")

        # Mostrar resumen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Phases", len(workflow['phases']))
        with col2:
            st.metric("Readiness Score", workflow['organizational_readiness']['readiness_score'])
        with col3:
            st.metric("Risk Level", workflow['compliance_assessment']['risk_level'].value.upper())

        # Botón para ir al dashboard
        if st.button("View Implementation Dashboard"):
            st.rerun()

def implementation_dashboard():
    """Dashboard de implementación"""
    st.subheader("📊 Implementation Dashboard")

    if 'current_workflow' not in st.session_state:
        st.warning("Please configure your project first in the Project Configuration tab.")
        return

    workflow = st.session_state.current_workflow

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Phases", len(workflow['phases']))

    with col2:
        readiness_score = workflow['organizational_readiness']['readiness_score']
        st.metric("Readiness Score", f"{readiness_score}/70")

    with col3:
        risk_level = workflow['compliance_assessment']['risk_level'].value
        st.metric("Risk Level", risk_level.upper())

    with col4:
        sprint_stories = len(workflow['sprint_planning']['stories'])
        st.metric("Fairness Stories", sprint_stories)

    # Visualización del workflow por fases
    st.markdown("### Implementation Phases")

    phases_data = []
    for phase_id, phase in workflow['phases'].items():
        phases_data.append({
            'Phase': phase['name'],
            'Duration': phase['duration'],
            'Activities': len(phase['activities']),
            'Deliverables': len(phase['deliverables'])
        })

    df_phases = pd.DataFrame(phases_data)

    # Gráfico de fases
    fig = px.bar(df_phases, x='Phase', y='Activities',
                 title="Activities per Phase",
                 color='Deliverables',
                 text='Duration')

    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Detalles de cada fase
    with st.expander("📋 Phase Details"):
        for phase_id, phase in workflow['phases'].items():
            st.markdown(f"#### {phase['name']}")
            st.write(f"**Duration:** {phase['duration']}")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Activities:**")
                for activity in phase['activities']:
                    st.write(f"• {activity}")

            with col2:
                st.write("**Deliverables:**")
                for deliverable in phase['deliverables']:
                    st.write(f"• {deliverable}")

            st.divider()

    # Métricas de éxito
    st.markdown("### Success Metrics")
    success_metrics = workflow['success_metrics']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Fairness Metrics:**")
        for metric in success_metrics['fairness_metrics'][:3]:
            st.write(f"• {metric}")

    with col2:
        st.markdown("**Timeline Metrics:**")
        for metric in success_metrics['timeline_metrics']:
            st.write(f"• {metric}")

    with col3:
        st.markdown("**Business Metrics:**")
        for metric in success_metrics['business_metrics']:
            st.write(f"• {metric}")


def case_study_demo():
    """Demostración del caso de estudio"""
    st.subheader("📚 Case Study: AI Recruitment Platform")

    with st.expander("🔍 Case Study Overview"):
        st.write("""
        This case study demonstrates the application of the Fairness Implementation Playbook
        to a multi-team AI recruitment platform. The scenario involves implementing fair AI
        across resume screening, candidate matching, and interview scheduling systems.
        """)

    if st.button("Run Case Study Simulation"):
        with st.spinner("Running case study simulation..."):
            case_study = RecruitmentPlatformCaseStudy()
            results = case_study.execute_case_study()

            st.session_state.case_study_results = results

        st.success("✅ Case Study Completed!")

    if 'case_study_results' in st.session_state:
        results = st.session_state.case_study_results

        # Métricas de éxito
        st.markdown("### Implementation Success Metrics")

        success_metrics = results['success_metrics_achieved']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Phases Completed",
                      f"{success_metrics['implementation_success']['phases_completed_on_time']}/5")
            st.metric("Deliverables Rate",
                      f"{success_metrics['implementation_success']['deliverables_completion_rate']:.1%}")

        with col2:
            st.metric("Bias Detection Accuracy",
                      f"{success_metrics['fairness_metrics']['bias_detection_accuracy']:.1%}")
            st.metric("Equal Opportunity Ratio",
                      f"{success_metrics['fairness_metrics']['equal_opportunity_ratio']:.2f}")

        with col3:
            st.metric("Performance Impact",
                      f"{success_metrics['business_metrics']['performance_impact']:.1%}")
            st.metric("Timeline Variance",
                      f"{success_metrics['business_metrics']['timeline_variance']:.1%}")

        # Lecciones aprendidas
        st.markdown("### Lessons Learned")

        lessons = results['lessons_learned']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Organizational:**")
            for lesson in lessons['organizational']:
                st.write(f"• {lesson}")

            st.markdown("**Technical:**")
            for lesson in lessons['technical']:
                st.write(f"• {lesson}")

        with col2:
            st.markdown("**Process:**")
            for lesson in lessons['process']:
                st.write(f"• {lesson}")

            st.markdown("**Compliance:**")
            for lesson in lessons['compliance']:
                st.write(f"• {lesson}")

        # Recomendaciones
        with st.expander("📋 Recommendations for Future Implementations"):
            for i, rec in enumerate(results['recommendations'], 1):
                st.write(f"{i}. {rec}")


def validation_interface():
    """Interfaz de validación"""
    st.subheader("✅ Implementation Validation")

    with st.expander("🔍 About Validation Framework"):
        st.write("""
        The validation framework helps you verify the effectiveness of your fairness
        implementation across four key dimensions: Technical, Process, Organizational,
        and Compliance effectiveness.
        """)

    st.markdown("### Input Your Implementation Metrics")

    # Crear formulario para métricas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Technical Effectiveness**")
        bias_detection_accuracy = st.slider("Bias Detection Accuracy", 0.0, 1.0, 0.90, 0.01)
        fairness_metric_coverage = st.slider("Fairness Metric Coverage", 0.0, 1.0, 0.95, 0.01)
        monitoring_reliability = st.slider("Monitoring System Reliability", 0.0, 1.0, 0.99, 0.01)
        performance_impact = st.slider("Performance Impact", -0.20, 0.0, -0.05, 0.01)

        st.markdown("**Process Effectiveness**")
        stakeholder_satisfaction = st.slider("Stakeholder Satisfaction", 0.0, 1.0, 0.85, 0.01)
        team_adoption_rate = st.slider("Team Adoption Rate", 0.0, 1.0, 0.80, 0.01)
        documentation_completeness = st.slider("Documentation Completeness", 0.0, 1.0, 0.95, 0.01)
        timeline_adherence = st.slider("Timeline Adherence", 0.0, 1.0, 0.90, 0.01)

    with col2:
        st.markdown("**Organizational Impact**")
        fairness_awareness_increase = st.slider("Fairness Awareness Increase", 0.0, 1.0, 0.70, 0.01)
        governance_effectiveness = st.slider("Governance Effectiveness", 0.0, 1.0, 0.85, 0.01)
        decision_accountability = st.slider("Decision Accountability", 0.0, 1.0, 0.90, 0.01)
        cross_team_collaboration = st.slider("Cross-team Collaboration", 0.0, 1.0, 0.75, 0.01)

        st.markdown("**Compliance Effectiveness**")
        regulatory_requirement_coverage = st.slider("Regulatory Requirement Coverage", 0.0, 1.0, 1.0, 0.01)
        audit_trail_completeness = st.slider("Audit Trail Completeness", 0.0, 1.0, 0.98, 0.01)
        evidence_collection_efficiency = st.slider("Evidence Collection Efficiency", 0.0, 1.0, 0.85, 0.01)
        legal_review_satisfaction = st.slider("Legal Review Satisfaction", 0.0, 1.0, 0.90, 0.01)

    if st.button("Validate Implementation", type="primary"):
        implementation_data = {
            "bias_detection_accuracy": bias_detection_accuracy,
            "fairness_metric_coverage": fairness_metric_coverage,
            "monitoring_system_reliability": monitoring_reliability,
            "performance_impact": performance_impact,
            "stakeholder_satisfaction": stakeholder_satisfaction,
            "team_adoption_rate": team_adoption_rate,
            "documentation_completeness": documentation_completeness,
            "timeline_adherence": timeline_adherence,
            "fairness_awareness_increase": fairness_awareness_increase,
            "governance_effectiveness": governance_effectiveness,
            "decision_accountability": decision_accountability,
            "cross_team_collaboration": cross_team_collaboration,
            "regulatory_requirement_coverage": regulatory_requirement_coverage,
            "audit_trail_completeness": audit_trail_completeness,
            "evidence_collection_efficiency": evidence_collection_efficiency,
            "legal_review_satisfaction": legal_review_satisfaction
        }

        validation_framework = st.session_state.validation_framework
        validation_results = validation_framework.validate_implementation(implementation_data)

        st.session_state.validation_results = validation_results

        # Mostrar resultados
        if validation_results['overall_success']:
            st.success("🎉 Implementation Validation: PASSED")
        else:
            st.error("⚠️ Implementation Validation: FAILED")
            st.warning(f"Critical failures: {len(validation_results['critical_failures'])}")

        # Dashboard de validación
        dashboard = validation_framework.create_validation_dashboard(validation_results)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall Status", dashboard['summary']['overall_status'])
        with col2:
            st.metric("Categories Passed",
                      f"{dashboard['summary']['categories_passed']}/{dashboard['summary']['total_categories']}")
        with col3:
            st.metric("Critical Failures", dashboard['summary']['critical_failures_count'])
        with col4:
            success_rate = dashboard['summary']['categories_passed'] / dashboard['summary']['total_categories']
            st.metric("Success Rate", f"{success_rate:.1%}")

        # Gráfico de puntuaciones por categoría
        categories = list(dashboard['category_scores'].keys())
        scores = [dashboard['category_scores'][cat]['score'] for cat in categories]

        fig = px.bar(x=categories, y=scores,
                     title="Validation Scores by Category",
                     labels={'x': 'Category', 'y': 'Score'},
                     color=scores,
                     color_continuous_scale="RdYlGn")

        st.plotly_chart(fig, use_container_width=True)

        # Recomendaciones
        if validation_results['recommendations']:
            st.markdown("### Recommendations")
            for i, rec in enumerate(validation_results['recommendations'], 1):
                st.write(f"{i}. {rec}")


def adaptability_interface():
    """Interfaz de adaptabilidad"""
    st.subheader("🔄 Playbook Adaptability")

    with st.expander("🔍 About Adaptability Guidelines"):
        st.write("""
        The adaptability guidelines help you customize the Fairness Implementation Playbook
        for different domains (healthcare, finance, education) and AI system types
        (classification, regression, LLMs, etc.).
        """)

    col1, col2 = st.columns(2)

    with col1:
        target_domain = st.selectbox(
            "Target Domain:",
            [d.value for d in Domain],
            help="Select the domain you want to adapt the playbook for"
        )

        target_system_type = st.selectbox(
            "Target System Type:",
            [s.value for s in AISystemType],
            help="Select the AI system type for adaptation"
        )

    with col2:
        st.markdown("### Current Playbook")
        if 'current_workflow' in st.session_state:
            workflow = st.session_state.current_workflow
            st.write(f"**Current Domain:** {workflow['project_config']['domain']}")
            st.write(f"**Current System:** {workflow['project_config']['system_type']}")
        else:
            st.info("Configure a project first to see current settings")

    if st.button("Generate Adaptation"):
        if 'current_workflow' not in st.session_state:
            st.error("Please configure a project first in the Project Configuration tab.")
            return

        adaptability_guide = AdaptabilityGuidelines()
        base_workflow = st.session_state.current_workflow

        adaptation = adaptability_guide.adapt_playbook(
            Domain(target_domain),
            AISystemType(target_system_type),
            base_workflow
        )

        st.session_state.adaptation = adaptation

        st.success("✅ Adaptation Generated!")

        # Mostrar resumen de adaptación
        summary = adaptation['adaptation_summary']

        st.markdown("### Adaptation Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Domain Considerations:**")
            for consideration in summary['domain_considerations'][:3]:
                st.write(f"• {consideration}")

        with col2:
            st.markdown("**System Considerations:**")
            for consideration in summary['system_considerations'][:3]:
                st.write(f"• {consideration}")

        # Guidance de implementación
        st.markdown("### Implementation Guidance")
        for guidance in adaptation['implementation_guidance'][:5]:
            st.info(f"💡 {guidance}")

        # Cross-domain insights
        cross_domain = adaptability_guide.generate_cross_domain_insights()

        with st.expander("🌍 Cross-Domain Insights"):
            st.markdown("**Common Patterns:**")
            for pattern in cross_domain['common_patterns']:
                st.write(f"• {pattern}")

            st.markdown("**Cross-Domain Lessons:**")
            for lesson in cross_domain['cross_domain_lessons']:
                st.write(f"• {lesson}")

def complete_integration_example():
    st.title("Complete Integration Example")
    # ... [Rest of your complete_integration_example function code] ...

def causal_fairness_toolkit():
    st.header("🛡️ Causal Fairness Toolkit")
    
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **Causal Analysis** goes beyond correlations to understand the *why* behind disparities.
        It’s like being a detective who not only sees two events happen together but reconstructs 
        the cause-and-effect chain that connects them. This helps us apply solutions that target 
        the root of the problem, instead of just treating the symptoms.
        Before mitigating, an engineer must hypothesize the causal pathways that lead to unfair outcomes.
        This toolkit helps you map out real-world biases that your data might reflect.
        """)
    with st.expander("🔍 **Source & Further Reading**"):
        st.markdown(
            """
            - **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference.* A foundational text on causal modeling.
            - **Kusner, M. J., et al. (2017).** *Counterfactual Fairness.* Introduces a definition of fairness based on what would have happened in a counterfactual world.
            """
        )
    if 'causal_report' not in st.session_state:
        st.session_state.causal_report = {}

    # CORRECTED LINE IS HERE:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Discrimination Mechanisms", "Counterfactual Fairness", "Causal Diagramming",
            "Causal Inference", "Intersectionality", "Mitigation Strategies"
        ])

    # TAB 1: Identification
    with tab1:
        st.header("🔬 Framework for Identifying Discrimination Mechanisms")
        st.info("Use this framework to identify the possible root causes of bias in your application and document your hypotheses about how bias operates in your specific domain.")
        st.subheader("1. Direct Discrimination")
        st.markdown("""
        Occurs when a protected attribute (such as race or gender) is explicitly used to make a decision. This is the most obvious type of bias.
        *Case Study Example: A past loan model explicitly used gender as a variable, assigning a lower weight to female applicants. This is a direct causal path.*
        """)
        
        with st.popover("How to Investigate Direct Influence"):
            st.markdown("""
            Answering this requires moving beyond simple correlation to investigate causation. Here’s a step-by-step guide:
            
            **Step 1: Frame the Question with a Counterfactual Test**
            The core idea is to ask: **"If all other information about an individual remained the same, but we only changed their protected attribute, would the model's decision change?"**
            
            **Step 2: Choose Your Investigation Method**
            - **Feature Importance Analysis (e.g., SHAP):** Break down individual predictions to see how much each feature contributed. If the protected attribute has a significant SHAP value, it's directly influencing that decision.
            - **Individual Conditional Expectation (ICE) Plots:** Visualize how a model's prediction for a single data point changes as you vary the protected attribute. A non-flat line indicates direct influence.

            **Step 3: Interpret the Findings**
            A non-flat ICE plot or a consistently high SHAP value for the protected attribute is strong evidence of direct influence. For legally protected classes, the acceptable level of direct influence is often zero.

            **Step 4: Take Action**
            If direct influence is found, the primary action is to remove the attribute and retrain the model. Afterwards, you must re-run your analysis to check if other variables have become proxies.
            """)
        
        st.text_area("Hypothesis for Direct Discrimination:", 
                     placeholder="Example: A hiring model explicitly assigns lower scores to female applicants.", 
                     key="causal_q1")
        with st.expander("📚 Suggested Reading"):
            st.markdown("""
            **Paper**: [Big Data's Disparate Impact](https://www.californialawreview.org/print/big-datas-disparate-impact/)  
            **Authors**: Barocas, S., & Selbst, A. D. (2016).  
            **Relevance**: This paper provides a foundational explanation of the legal and technical distinction between **disparate treatment** (direct discrimination) and **disparate impact** (indirect discrimination) in the context of algorithms.
            """)
        
        st.divider()

        # --- Indirect Discrimination ---
        st.subheader("2. Indirect Discrimination")
        st.markdown("""
        Occurs when a protected attribute affects an intermediate factor that is legitimate for the decision. The bias is transmitted through this mediating variable.
        *Case Study Example: Gender influences career interruptions (e.g., for childcare), which reduces 'years of experience'. The model then penalizes lower experience, indirectly discriminating against women.*
        """)

        with st.popover("How to Investigate Indirect Influence"):
            st.markdown("""
            This analysis focuses on understanding causal pathways in your data.

            **Step 1: Hypothesize the Causal Path**
            Clearly define the potential pathway: `Protected Attribute (A) -> Mediating Variable (M) -> Outcome (Y)`.
            *Example: `Gender -> Career Interruptions -> Loan Decision`*

            **Step 2: Statistically Test the Links**
            - **Test A -> M:** Is there a significant relationship between the protected attribute and the mediator? (e.g., Use a t-test to see if career interruptions differ by gender).
            - **Test M -> Y:** Is there a significant relationship between the mediator and the outcome, even after controlling for the protected attribute? (e.g., Use a regression model to see if career interruptions predict the loan decision).

            **Step 3: Conduct a Mediation Analysis**
            Use statistical packages (e.g., `statsmodels` in Python) to formally quantify the "indirect effect." This measures precisely how much of the total bias flows through the mediating variable.

            **Step 4: Take Action**
            If a significant indirect path exists, consider adjusting the mediator (e.g., use a metric not penalized by career interruptions) or using modeling techniques that can block the influence from that specific path.
            """)

        st.text_area("Hypothesis for Indirect Discrimination:", 
                     placeholder="Example: Gender can influence career breaks (for childcare), and the model penalizes these breaks, indirectly affecting women.", 
                     key="causal_q2")
        with st.expander("📚 Suggested Reading"):
            st.markdown("""
            **Paper**: [Counterfactual Fairness](https://arxiv.org/abs/1703.06856)  
            **Authors**: Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017).  
            **Relevance**: This paper introduces a formal causal model for fairness. It directly addresses indirect discrimination by asking if a decision would be the same in a "counterfactual world" where the person's protected attribute was different.
            """)

        st.divider()

        # --- Proxy Discrimination ---
        st.subheader("3. Proxy Discrimination")
        st.markdown("""
        Occurs when a seemingly neutral variable is so correlated with a protected attribute that it acts as a substitute (a 'proxy').
        *Case Study Example: In our city, zip code is a strong proxy for race due to historical segregation. Using zip code in a loan model can perpetuate this bias.*
        """)
        
        with st.popover("How to Investigate Proxy Use"):
            st.markdown("""
            Detecting proxies involves finding variables that are redundant with protected attributes.

            **Step 1: Identify Potential Proxies with Correlation Analysis**
            Calculate the correlation between your protected attribute and all other features. A high correlation is a red flag. Use a heatmap for easy visualization. 

            **Step 2: Measure Feature Redundancy**
            - **Variance Inflation Factor (VIF):** A high VIF for a potential proxy indicates it's highly redundant with other features (possibly the protected attribute).
            - **Mutual Information:** This can capture non-linear relationships that simple correlation might miss.

            **Step 3: Test Model Sensitivity**
            1. Remove the protected attribute from the data and retrain the model.
            2. Use feature importance techniques (like SHAP) on the new model. If a potential proxy (like 'zip code') now has very high importance, the model is likely using it as a stand-in.

            **Step 4: Take Action**
            If a variable is confirmed as a strong proxy, the most effective action is to remove it from the model. Retrain and re-evaluate fairness to ensure the problem is resolved.
            """)

        st.text_area("Hypothesis for Proxy Discrimination:", 
                     placeholder="Example: In a credit model, using zip code as a predictor can be a proxy for race due to historical residential segregation.", 
                     key="causal_q3")
        with st.expander("📚 Suggested Reading"):
            st.markdown("""
            **Paper**: [Algorithmic transparency via quantitative input influence](https://arxiv.org/abs/1606.00219)  
            **Authors**: Datta, A., Sen, S., & Zick, Y. (2016).  
            **Relevance**: This work offers a technical approach for detecting proxies. It presents methods to measure the influence of each input variable on a model's outcome, which can reveal when a feature like zip code is having an outsized, proxy-like effect.
            """)

    # TAB 2: Counterfactual Analysis
    with tab2:
        st.subheader("⚖️ Practical Counterfactual Fairness Methodology")
        st.info("Analyze, quantify, and mitigate counterfactual bias in your model by examining what the model *would have* decided if a protected attribute had been different.")
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
            with st.popover("How-To Guide"):
                st.markdown("""
                **Goal:** Create hypothetical versions of your data to test for bias.
                
                **Method:** For a specific individual in your dataset, create a copy of their data but change the value of the protected attribute.

                **Python Example:**
                ```python
                import pandas as pd

                # Original applicant data
                applicant = pd.Series({
                    'income': 50000, 
                    'credit_score': 680, 
                    'race': 'Group B'
                })

                # Create the counterfactual
                counterfactual_applicant = applicant.copy()
                counterfactual_applicant['race'] = 'Group A'

                # Now, you can feed both versions to your model
                # original_pred = model.predict(applicant)
                # counterfactual_pred = model.predict(counterfactual_applicant)
                ```
                """)

            st.text_area("1.2 Identify Causal Paths (Fair vs. Unfair)", 
                         placeholder="Example: Path Race → Zip Code → Loan Decision is unfair because zip code is a proxy. Path Education Level → Income → Loan Decision is considered fair.", 
                         key="causal_q5")
            with st.popover("How-To Guide"):
                st.markdown("""
                **Goal:** Map out how you believe variables influence each other and the final decision. This helps separate fair from unfair influences.
                
                **Method:** Draw a Directed Acyclic Graph (DAG) representing your assumptions. Arrows indicate causal influence.
                
                **Example DAG (Text-based):**
                - `Race -> Zip Code` (Unfair link, historical bias)
                - `Zip Code -> Loan Decision` (Unfair path)
                - `Education -> Income` (Fair link)
                - `Income -> Loan Decision` (Fair path)

                By defining these paths, you can later test which ones your model is actually using.
                """)

            st.text_area("1.3 Measure Disparities and Document", 
                         placeholder="Example: 15% of applicants from the disadvantaged group would have been approved in the counterfactual scenario. This indicates a counterfactual fairness violation.", 
                         key="causal_q6")
            with st.popover("How-To Guide"):
                st.markdown("""
                **Goal:** Quantify the impact of the unfair paths you've identified.
                
                **Method:** For a group of individuals who received a negative outcome, generate counterfactuals for all of them. Calculate the percentage whose outcome would have changed for the better.

                **Python Example:**
                ```python
                def calculate_cf_disparity(rejected_group_df, model):
                    positive_outcome_flips = 0
                    total_rejected = len(rejected_group_df)

                    for index, person in rejected_group_df.iterrows():
                        cf_person = person.copy()
                        cf_person['race'] = 'Privileged_Group'
                        
                        # Assuming model.predict returns 1 for approve, 0 for reject
                        if model.predict(cf_person) == 1:
                            positive_outcome_flips += 1
                    
                    disparity_percentage = (positive_outcome_flips / total_rejected) * 100
                    return disparity_percentage
                
                # disparity = calculate_cf_disparity(rejected_group_b, model)
                # print(f"{disparity:.2f}% would have been approved.")
                ```
                """)

        with st.container(border=True):
            st.markdown("##### Step 2: Specific Path Analysis")
            st.text_area("2.1 Decompose and Classify Paths", 
                         placeholder="Example: Path 1 (zip code proxy) classified as UNFAIR. Path 2 (mediated by income) classified as FAIR.", 
                         key="causal_q7")
            with st.popover("How-To Guide"):
                 st.markdown("""
                **Goal:** Formally separate and label the causal pathways you mapped in Step 1.2.
                
                **Method:** Based on your domain knowledge and fairness principles, explicitly classify each path from the protected attribute to the outcome.
                
                **Classification:**
                - **Unfair Path:** A path that transmits bias through a proxy or an illegitimate mediator. Ex: `Race -> Zip Code -> Decision`.
                - **Fair Path:** A path that operates through a legitimate, business-relevant mediator. Ex: `Age -> Years of Experience -> Job Performance Score`.
                
                Documenting this is crucial for justifying your intervention choices.
                """)
            
            st.text_area("2.2 Quantify Contribution and Document", 
                         placeholder="Example: The zip code path accounts for 60% of observed disparity. Reason: Reflects historical residential segregation bias.", 
                         key="causal_q8")
            with st.popover("How-To Guide"):
                st.markdown("""
                **Goal:** Measure how much of the total bias is attributable to each specific unfair path.
                
                **Method:** Use advanced techniques like **Path-Specific Mediation Analysis**. This isolates the effect flowing through one path while blocking others. This is complex and often requires specialized libraries (`causalimpact`, `dowhy`).
                
                **Conceptual Python:**
                ```python
                # This is conceptual; actual implementation is complex
                from causal_lib import PathAnalysis

                analysis = PathAnalysis(model, data, dag)

                # Quantify the effect flowing only through 'zip_code'
                zip_code_effect = analysis.get_path_effect(
                    path=['race', 'zip_code', 'decision']
                )

                total_effect = analysis.get_total_effect('race', 'decision')
                
                contribution = (zip_code_effect / total_effect) * 100
                # print(f"Zip code path contributes {contribution:.2f}% of the bias.")
                ```
                """)

        with st.container(border=True):
            st.markdown("##### Step 3: Intervention Design")
            st.selectbox("3.1 Select Intervention Approach", ["Data Level", "Model Level", "Post-processing"], key="causal_q9")
            with st.popover("How-To Guide"):
                st.markdown("""
                Choose your strategy based on your findings and technical constraints.
                
                - **Data Level (Pre-processing):** Modify the data before training.
                  - *Example:* Remove proxy variables (like zip code), or apply transformations to de-bias features. Best for tackling the root cause.
                
                - **Model Level (In-processing):** Use a fairness-aware algorithm or add constraints during training.
                  - *Example:* Use an adversarial training model that tries to predict the outcome while being unable to predict the protected attribute. Gives fine-grained control.

                - **Post-processing:** Adjust model outputs after prediction.
                  - *Example:* Apply different decision thresholds for different groups to equalize outcomes. Easiest to implement, but doesn't fix the underlying model bias.
                """)

            st.text_area("3.2 Implement and Monitor", 
                         placeholder="Example: Applied a transformation to the zip code feature. Counterfactual disparity reduced by 50%.", 
                         key="causal_q10")
            with st.popover("How-To Guide"):
                st.markdown("""
                **Goal:** Apply your chosen intervention and verify its effectiveness.
                
                **Method:** This is an iterative loop:
                1.  **Implement:** Apply the change (e.g., drop the proxy feature).
                2.  **Retrain:** Train your model on the modified data.
                3.  **Re-evaluate:** Rerun your counterfactual analysis (Step 1.3) and other fairness metrics.
                4.  **Document:** Record the change in disparity. Did it improve? Did it harm model accuracy?
                5.  **Repeat:** Continue refining until you meet your fairness goals without unacceptably compromising performance.
                """)

    # TAB 3: Causal Diagram
    with tab3:
        st.info("Sketch diagrams to visualize causal relationships and document your assumptions.")
        with st.expander("💡 Causal Diagram Simulator"):
            st.write("Build a simple causal diagram by selecting relationships between variables. This helps visualize your hypotheses about how bias operates.")
            
            # 1. Define example scenarios
            example_scenarios = {
                "Loan Approval (Default)": {
                    "nodes": ["Gender", "Race", "Zip Code", "Income", "Years Experience", "Loan Decision"],
                    "default_edges": ["Race → Zip Code", "Zip Code → Loan Decision", "Income → Loan Decision"]
                },
                "Recruitment / Hiring": {
                    "nodes": ["Gender", "University Attended", "Years Experience", "Interview Score", "Hired"],
                    "default_edges": ["Gender → Years Experience", "University Attended → Interview Score", "Years Experience → Hired", "Interview Score → Hired"]
                },
                "Healthcare Diagnosis": {
                    "nodes": ["Age", "Race", "Zip Code", "Previous Conditions", "Insurance Type", "Diagnosis Severity"],
                    "default_edges": ["Race → Zip Code", "Zip Code → Insurance Type", "Previous Conditions → Diagnosis Severity", "Insurance Type → Diagnosis Severity"]
                }
            }

            # 2. Add example selector and custom variable input
            scenario_choice = st.selectbox("Load an example scenario:", list(example_scenarios.keys()), key="scenario_select")

            # Initialize or update session state for the text area
            if 'current_scenario' not in st.session_state or st.session_state.current_scenario != scenario_choice:
                st.session_state.custom_vars = "\n".join(example_scenarios[scenario_choice]["nodes"])
                st.session_state.current_scenario = scenario_choice

            custom_vars_text = st.text_area(
                "Add or edit your system's variables (one per line):",
                value=st.session_state.custom_vars,
                key="custom_vars_input",
                height=150,
                help="These variables will be the nodes in your causal diagram."
            )
            st.session_state.custom_vars = custom_vars_text # Persist user edits
            nodes = [node.strip() for node in custom_vars_text.split('\n') if node.strip()]

            # 3. Dynamically generate relationship options
            possible_relations = []
            if len(nodes) > 1:
                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if i != j:
                            possible_relations.append(f"{nodes[i]} → {nodes[j]}")

            # Filter default relations to only include those possible with the current nodes
            valid_defaults = [rel for rel in example_scenarios[scenario_choice]["default_edges"] if rel in possible_relations]

            selected_relations = st.multiselect(
                "Select the causal relationships you assume exist in your problem:",
                options=sorted(possible_relations),
                key="causal_relations",
                default=valid_defaults
            )

            # 4. Render the graph and documentation area
            if selected_relations:
                dot_string = "digraph { rankdir=LR; node [shape=box, style=rounded]; "
                for rel in selected_relations:
                    cause, effect = rel.split(" → ")
                    dot_string += f'"{cause}" -> "{effect}"; '
                dot_string += "}"
                st.graphviz_chart(dot_string)
                st.text_area(
                    "**Document Your Diagram**: Explain the paths and why they are problematic.",
                    placeholder="Case Study Example: The diagram shows the path 'Race → Zip Code → Loan Decision'. This is an unfair path because zip code acts as a proxy for race, channeling historical bias into the model's decision.",
                    key="causal_diagram_doc"
                )

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
        st.info("Practical methods for estimating causal effects when observational data is imperfect or limited.")
        
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
        with tab6:
            st.subheader("Intervention & Mitigation Strategies")
            st.info("Select and document the strategies you will use to address the biases identified in the previous steps.")

            # --- Pre-processing ---
            st.markdown("##### 1. Pre-processing (Data-Level Interventions)")
            with st.expander("🔍 Definition: Reweighing"):
                st.markdown("""
                Adjusts the weight of samples in the training data to balance the representation of different demographic groups. The goal is to make the model pay more attention to the under-represented or disadvantaged groups during training.
                - **Pros:** Model-agnostic, directly addresses data imbalance.
                - **Cons:** Can distort the original data distribution, potentially affecting overall model accuracy.
                """)
            with st.expander("💡 Interactive Example: Reweighing Simulation"):
                st.write("This simulation shows how sample weights can influence a simple decision boundary.")
                # Create sample data
                group_a_size = st.slider("Size of Group A (Privileged)", 50, 200, 150)
                group_b_size = st.slider("Size of Group B (Disadvantaged)", 10, 100, 30)
                
                # Assign weights
                weight_a = 1.0
                weight_b = st.slider("Weight for Group B", 1.0, 10.0, 5.0, help="Higher weights make the model treat Group B samples as more important.")

                # Simple logic for outcome
                outcome = "Balanced" if (group_b_size * weight_b) / (group_a_size * weight_a) > 0.8 else "Biased towards Group A"
                st.metric("Model Outcome with Current Weights", outcome)
                st.write(f"Effective representation ratio (Group B / Group A): **{((group_b_size * weight_b) / (group_a_size * weight_a)):.2f}**")
                st.info("By increasing the weight for Group B, you force the model to treat it as more prevalent, leading to a more balanced outcome even when the group is smaller in number.")

            # --- In-processing ---
            st.markdown("##### 2. In-processing (Model-Level Interventions)")
            with st.expander("🔍 Definition: Adversarial Debiasing"):
                st.markdown("""
                Trains two models simultaneously: a main **Predictor** model that tries to predict the outcome, and an **Adversary** model that tries to predict the protected attribute from the Predictor's output. The Predictor is trained to maximize its accuracy while also "fooling" the Adversary.
                - **Pros:** Can achieve a good balance between fairness and accuracy.
                - **Cons:** Increases training complexity significantly, can be unstable to train.
                """)
                st.graphviz_chart("""
                digraph {
                    rankdir=LR;
                    "Input Data" -> "Predictor Model" -> "Prediction (Y)";
                    "Predictor Model" -> "Adversary Model" [label="tries to predict Prot. Attr."];
                    "Adversary Model" -> "Predictor Model" [label="sends back loss to penalize"];
                }
                """)

            # --- Post-processing ---
            st.markdown("##### 3. Post-processing (Prediction-Level Interventions)")
            with st.expander("🔍 Definition: Calibrated Equalized Odds"):
                st.markdown("""
                Adjusts the model's output scores (probabilities) for different groups to satisfy a fairness constraint, like equal opportunity or equalized odds, without retraining the model. This is often done by setting different classification thresholds for each group.
                - **Pros:** Easy to implement, model-agnostic (works on black-box models).
                - **Cons:** Does not fix the underlying bias in the model; it only corrects the outputs.
                """)
            with st.expander("💡 Interactive Example: Threshold Adjustment"):
                st.write("Adjust decision thresholds to see how they impact approval rates and fairness.")
                
                # Simulate some scores
                np.random.seed(10)
                scores_group_a = np.random.normal(0.6, 0.15, 100)
                scores_group_b = np.random.normal(0.45, 0.15, 100)

                thresh_a = st.slider("Threshold for Group A", 0.0, 1.0, 0.5)
                thresh_b = st.slider("Threshold for Group B", 0.0, 1.0, 0.5)

                approved_a = np.sum(scores_group_a >= thresh_a)
                approved_b = np.sum(scores_group_b >= thresh_b)

                col1, col2 = st.columns(2)
                col1.metric("Group A Approval Rate", f"{approved_a}%")
                col2.metric("Group B Approval Rate", f"{approved_b}%")

                if abs(approved_a - approved_b) < 5:
                    st.success("The approval rates are now nearly equal (Demographic Parity is high).")
                else:
                    st.warning("The approval rates are unequal.")
                st.info("Notice that to achieve equal approval rates, you may need to set a lower threshold for the group with the lower average score distribution.")

            st.text_area(
                "Document Your Chosen Mitigation Strategy",
                placeholder="Example: We will apply Reweighing during pre-processing to give more importance to applicants from underrepresented zip codes. We will monitor for a 10% drop in overall accuracy.",
                key="mitigation_doc"
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

    # ...  function code] ...
def preprocessing_fairness_toolkit():
    st.header("🧪 Pre-processing Fairness Toolkit")
    with st.expander("🔍 Friendly Definition"):
        st.write("""
        **Pre-processing** means "cleaning" the data *before* the model learns from it. 
        It’s like preparing ingredients for a recipe: if you know some ingredients are biased 
        (e.g., too salty), you adjust them before cooking to ensure the final dish is balanced.
        """)
    with st.expander("🔍 **Source & Further Reading**"):
        st.markdown(
            """
            - **Kamiran, F., & Calders, T. (2012).** *Data preprocessing techniques for classification without discrimination.* A key paper on pre-processing methods.
            - **Zemel, R., et al. (2013).** *Learning Fair Representations.* Discusses transforming data into a latent space where protected attributes are obfuscated.
            """
        )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Representation Analysis", "Correlation Detection", "Label Quality",
        "Re-weighting & Re-sampling", "Transformation", "Data Generation",
        "🌍 Intersectionality", "🔧 Bias Mitigation Techniques"
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
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
             st.text_area("1. Comparison with Reference Population",
                         placeholder="E.g.: Our dataset has 70% Group A and 30% Group B, while the real population is 50/50.",
                         key="p1")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Check if your dataset's demographics mirror a known, true population.
                **Method:** Calculate the proportion of each group in your data and compare it to a benchmark like census data.
                **Python Code:**
                ```python
                # Proportions in your data
                data_proportions = df['race'].value_counts(normalize=True) * 100
                
                # Known population proportions
                census_proportions = pd.Series({
                    'Group A': 50, 'Group B': 50
                })
                
                # Create a comparison DataFrame
                comparison_df = pd.DataFrame({
                    'Data (%)': data_proportions,
                    'Census (%)': census_proportions
                })
                print(comparison_df)
                ```
                """)
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("2. Intersectional Representation Analysis",
                         placeholder="E.g.: Women from racial minorities make up only 3% of the data, though they represent 10% of the population.",
                         key="p2")
        with col2:
             with st.popover("How-To"):
                st.markdown("""
                **Goal:** Find hidden underrepresentation in overlapping subgroups.
                **Method:** Use `groupby()` on multiple protected attributes to see the size of intersectional groups.
                **Python Code:**
                ```python
                # Group by both race and gender
                intersectional_counts = df.groupby(['race', 'gender']).size().reset_index(name='count')
                
                # Calculate percentage
                total = len(df)
                intersectional_counts['percentage'] = (intersectional_counts['count'] / total) * 100
                
                print(intersectional_counts)
                ```
                """)

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("3. Representation Across Outcome Categories",
                         placeholder="E.g.: Group A constitutes 30% of applications but only 10% of approvals.",
                         key="p3")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Check if representation is consistent for different outcomes (e.g., approved vs. rejected).
                **Method:** Filter your data by the outcome label, then calculate group proportions.
                **Python Code:**
                ```python
                # Filter for 'approved' applications
                approved_df = df[df['outcome'] == 'approved']
                
                # Calculate proportions within the approved group
                approved_proportions = approved_df['race'].value_counts(normalize=True) * 100
                
                print("Representation among approvals:")
                print(approved_proportions)
                ```
                """)

    # TAB 2: Correlation Detection
    with tab2:
        st.subheader("Correlation Pattern Detection")
        with st.expander("🔍 Friendly Definition"):
            st.write("We look for seemingly neutral variables that are strongly connected to protected attributes. For example, if a postal code is strongly correlated with race, the model could use the postal code to discriminate indirectly.")
        
        with st.expander("💡 Interactive Example: Proxy Detection"):
            st.write("Visualize how a 'proxy' variable (e.g., Postal Code) can be correlated with both a Protected Attribute (e.g., Demographic Group) and the Outcome (e.g., Credit Score).")
            np.random.seed(1)
            group = np.random.randint(0, 2, 100)
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

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("1. Direct Correlations (Protected Attribute ↔ Outcome)",
                         placeholder="E.g.: In historical data, gender has a correlation of 0.3 with the hiring decision.",
                         key="p4")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Check if the outcome is directly correlated with a protected attribute in the training data.
                **Method:** Calculate the correlation. If the outcome is binary, you can use point-biserial correlation.
                **Python Code:**
                ```python
                from scipy.stats import pointbiserialr
                # Assuming 'gender' is 0/1 and 'hired' is 0/1
                corr, p_value = pointbiserialr(df['gender'], df['hired'])
                print(f"Correlation: {corr:.3f}, P-value: {p_value:.3f}")
                ```
                """)
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("2. Proxy Variable Identification (Protected Attribute ↔ Feature)",
                         placeholder="E.g.: The 'chess club attendance' feature is highly correlated with being male.",
                         key="p5")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Find features that are strong stand-ins (proxies) for protected attributes.
                **Method:** Create a correlation matrix to see the relationships between all variables. Pay close attention to high correlations between protected attributes and other features.
                **Python Code:**
                ```python
                # First, one-hot encode categorical variables
                df_encoded = pd.get_dummies(df, columns=['gender', 'zip_code'])
                
                # Calculate the correlation matrix
                correlation_matrix = df_encoded.corr()
                
                # Find correlations with the 'gender_male' column
                gender_correlations = correlation_matrix['gender_male'].sort_values(ascending=False)
                
                print("Top correlations with gender:")
                print(gender_correlations.head(10))
                ```
                """)

    # TAB 3: Label Quality
    with tab3:
        st.subheader("Label Quality Evaluation")
        with st.expander("🔍 Friendly Definition"):
            st.write("Labels are the correct answers in your training data (e.g., 'was hired', 'did not repay the loan'). If these labels come from past human decisions that were biased, your model will learn that same bias.")
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("1. Historical Bias in Decisions",
                         placeholder="Example: 'Promoted' labels in our dataset come from a period when the company had biased promotion policies, so the labels themselves are a bias source.",
                         key="p6")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Determine if the "ground truth" labels are themselves biased.
                **Method:** This is often non-technical. Analyze the source of the labels. Talk to domain experts. If possible, compare decision rates between groups from the historical data.
                **Python Code (Example):**
                ```python
                # Compare promotion rates from historical data
                promotion_rates = df.groupby('gender')['promoted'].mean() * 100
                
                print("Historical Promotion Rates:")
                print(promotion_rates)
                # A large difference may indicate biased labels.
                ```
                """)
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("2. Annotator Bias",
                         placeholder="Example: Annotator agreement analysis shows male annotators rated the same comments as 'toxic' less often than female annotators, indicating label bias.",
                         key="p7")
        with col2:
             with st.popover("How-To"):
                st.markdown("""
                **Goal:** Check if the humans who labeled your data had systematic biases.
                **Method:** If you have data on which annotator provided which label, you can compare their labeling patterns.
                **Python Code (Example):**
                ```python
                # Assuming df has 'annotator_gender' and 'label'
                labeling_tendency = df.groupby('annotator_gender')['label'].value_counts(normalize=True)
                
                print("Labeling tendency by annotator group:")
                print(labeling_tendency.unstack())
                # Look for significant differences in how groups assign labels.
                ```
                """)

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
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("Decision Criteria: Re-weight or Re-sample?",
                         placeholder="Based on my audit and model, the best strategy is...",
                         key="p8")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Choose the right technique to balance your data.
                - **Re-sample:** Best when the class imbalance is severe. Oversampling is generally preferred over undersampling to avoid data loss.
                - **Re-weight:** Best for less severe imbalance or when you don't want to alter the training set size.
                **Python Code (Oversampling):**
                ```python
                from sklearn.utils import resample
                
                majority = df[df.outcome==0]
                minority = df[df.outcome==1]
                
                minority_oversampled = resample(minority, 
                                         replace=True,    
                                         n_samples=len(majority),   
                                         random_state=123)
                
                balanced_df = pd.concat([majority, minority_oversampled])
                ```
                """)
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("Intersectionality Consideration",
                         placeholder="Example: To address underrepresentation of minority women, we will apply stratified oversampling to ensure this subgroup reaches parity with others.",
                         key="p9")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Ensure that balancing for a main group doesn't worsen the imbalance for an intersectional subgroup.
                **Method:** Apply balancing techniques *within* demographic groups (stratified sampling).
                **Python Code (Conceptual):**
                ```python
                balanced_subgroups = []
                for group_name, group_df in df.groupby(['race', 'gender']):
                    # Apply oversampling within this specific subgroup
                    # (Code similar to the above example)
                    balanced_subgroup = oversample(group_df)
                    balanced_subgroups.append(balanced_subgroup)
                
                fully_balanced_df = pd.concat(balanced_subgroups)
                ```
                """)

    with tab5:
        st.subheader("Distribution Transformation Approaches")
        with st.expander("🔍 Friendly Definition"):
            st.write("This technique directly modifies feature values to break problematic correlations with protected attributes. It’s like 'recalibrating' a variable so it means the same for all groups.")
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("1. Disparate Impact Removal",
                         placeholder="E.g.: 'Repair' the 'postal code' feature so its distribution is the same across racial groups, eliminating its use as a proxy.",
                         key="p10")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Transform a feature to remove its correlation with a protected attribute while preserving its ranking information as much as possible.
                **Method:** Use algorithms like `DisparateImpactRemover` from the AIF360 library. It adjusts feature values quantile by quantile to match distributions across groups.
                **Python Code (Conceptual with AIF360):**
                ```python
                from aif360.algorithms.preprocessing import DisparateImpactRemover
                
                # Assume 'dataset' is an AIF360 data structure
                remover = DisparateImpactRemover(repair_level=1.0)
                repaired_dataset = remover.fit_transform(dataset)
                ```
                """)

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("2. Fair Representations (LFR, LAFTR)",
                         placeholder="E.g.: Use an adversarial autoencoder to learn an applicant profile representation without gender information.",
                         key="p11")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Learn a new, compressed representation of the data that is explicitly designed to be fair.
                **Method:** Train a model (like an autoencoder with an adversary) to create data embeddings that are useful for predicting the outcome but do not contain information about the protected attribute.
                **Conceptual Explanation:**
                Instead of cleaning individual features, you are creating entirely new, "fair" features. This is an advanced technique that fundamentally changes your data before it reaches the main model.
                """)

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("3. Intersectionality Considerations",
                         placeholder="My transformation strategy will focus on intersections of gender and ethnicity...",
                         key="p12")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Ensure that making a feature fair for main groups doesn't make it unfair for subgroups.
                **Method:** When applying transformations, do it with respect to intersectional groups, not just single attributes.
                **Example:** Instead of making 'zip_code' fair for 'race', make it fair for 'race' and 'gender' combinations simultaneously. This prevents a situation where the feature becomes fair for white women and Black men, but remains biased against Black women.
                """)

    with tab6:
        st.subheader("Fairness-Aware Data Generation")
        with st.expander("🔍 Friendly Definition"):
            st.write("When data is very scarce or biased, we can generate synthetic (artificial) data to fill the gaps. This is especially useful for creating examples of very small intersectional groups or for generating counterfactual scenarios.")
        st.markdown("**When to Generate Data:** When there is severe underrepresentation or counterfactual examples are needed.")
        st.markdown("**Strategies:** Conditional Generation, Counterfactual Augmentation.")
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("Intersectionality Considerations",
                         placeholder="Example: We will use a generative model conditioned on the intersection of age and gender to create synthetic profiles of 'older women in tech', a group absent in our data.",
                         key="p13")
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Create realistic synthetic data for severely underrepresented intersectional groups.
                **Method:** Use a conditional generative model (like a CTGAN) that can generate data based on specified attributes.
                **Python Code (Conceptual with CTGAN):**
                ```python
                from ctgan import CTGANSynthesizer
                
                # Train the synthesizer on your real data
                synthesizer = CTGANSynthesizer(epochs=100)
                synthesizer.fit(df)
                
                # Generate synthetic data for a specific subgroup
                # (Create a dataframe with the conditions you want)
                conditions = pd.DataFrame({
                    'gender': ['Woman'],
                    'race': ['Minority'],
                    'age_group': ['50+']
                })
                
                synthetic_data = synthesizer.sample(
                    num_rows=100,
                    condition_column='gender',
                    condition_value='Woman' 
                    # More complex conditioning is possible
                )
                ```
                """)
                
    with tab7:
        st.subheader("Interseccionalidad en el Pre-procesamiento")
        with st.expander("🔍 Definición Amigable"):
            st.write("""
            La interseccionalidad aquí significa ir más allá de equilibrar los datos para grupos principales (ej. hombres vs. mujeres). Debemos asegurarnos de que los **subgrupos específicos** (ej. mujeres negras, hombres latinos jóvenes) también estén bien representados. Las técnicas de pre-procesamiento deben aplicarse de forma estratificada para corregir desequilibrios en estas intersecciones, que a menudo son las más vulnerables al sesgo.
            """)
        
        with st.expander("💡 Ejemplo Interactivo: Re-muestreo Estratificado Interseccional"):
            st.write("Observa cómo un conjunto de datos puede parecer equilibrado en un eje (Grupo A vs. B), pero no en sus intersecciones. El re-muestreo estratificado soluciona esto.")

            # Initial data
            np.random.seed(1)
            hombres_a = pd.DataFrame({'Característica 1': np.random.normal(2, 1, 80), 'Característica 2': np.random.normal(5, 1, 80), 'Grupo': 'Hombres A'})
            mujeres_a = pd.DataFrame({'Característica 1': np.random.normal(2.5, 1, 20), 'Característica 2': np.random.normal(5.5, 1, 20), 'Grupo': 'Mujeres A'})
            hombres_b = pd.DataFrame({'Característica 1': np.random.normal(6, 1, 50), 'Característica 2': np.random.normal(2, 1, 50), 'Grupo': 'Hombres B'})
            mujeres_b = pd.DataFrame({'Característica 1': np.random.normal(6.5, 1, 50), 'Característica 2': np.random.normal(2.5, 1, 50), 'Grupo': 'Mujeres B'})
            mujeres_b_interseccional = pd.DataFrame({'Característica 1': np.random.normal(7, 1, 10), 'Característica 2': np.random.normal(3, 1, 10), 'Grupo': 'Mujeres B (Intersección)'})

            df_original = pd.concat([hombres_a, mujeres_a, hombres_b, mujeres_b, mujeres_b_interseccional])
            
            remuestreo_factor = st.slider("Factor de sobremuestreo para 'Mujeres B (Intersección)'", 1, 10, 5, key="inter_remuestreo")
            
            if remuestreo_factor > 1:
                indices_remuestreo = mujeres_b_interseccional.sample(n=(remuestreo_factor-1)*len(mujeres_b_interseccional), replace=True).index
                df_remuestreado = pd.concat([df_original, mujeres_b_interseccional.loc[indices_remuestreo]])
            else:
                df_remuestreado = df_original

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

            for name, group in df_original.groupby('Grupo'):
                ax1.scatter(group['Característica 1'], group['Característica 2'], label=f"{name} (n={len(group)})", alpha=0.7)
            ax1.set_title("Datos Originales")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)

            for name, group in df_remuestreado.groupby('Grupo'):
                ax2.scatter(group['Característica 1'], group['Característica 2'], label=f"{name} (n={len(group)})", alpha=0.7)
            ax2.set_title("Datos con Sobremuestreo Interseccional")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)

            st.pyplot(fig)
            st.info("El grupo 'Mujeres B (Intersección)' estaba severamente subrepresentado. Al aplicar un sobremuestreo específico para este subgrupo, ayudamos al modelo a aprender sus patrones sin distorsionar el resto de los datos.")
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area("Aplica a tu caso: ¿Qué subgrupos interseccionales están subrepresentados en tus datos y qué estrategia de re-muestreo/re-ponderación estratificada podrías usar?", key="p_inter")
        with col2:
            with st.popover("Guía"):
                st.markdown("""
                **1. Identifica:** ¿Cuáles son los subgrupos clave en tu problema (ej. 'mujeres latinas', 'hombres asiáticos mayores de 60')?
                
                **2. Analiza:** Usa el código de la pestaña "Representation Analysis" para medir el tamaño de estos subgrupos. ¿Están muy por debajo de su representación poblacional?
                
                **3. Estrategia:** Describe tu plan. Ejemplo: "El subgrupo 'mujeres negras' es solo el 2% de nuestros datos, pero el 8% de la población. Aplicaremos sobremuestreo estratificado para quintuplicar su presencia en el set de entrenamiento."
                """)

    with tab8:
        st.subheader("Integrated Bias Mitigation Techniques")
        st.info("Apply specific bias mitigation techniques with interactive examples and code templates.")
        technique = st.selectbox(
            "Select bias mitigation technique:",
            ["Oversampling", "Undersampling", "Reweighting", "SMOTE", "Data Augmentation"],
            key="bias_mit_selector"
        )
        if technique == "Oversampling":
            st.markdown("### Oversampling Implementation")
            st.code("""
# Oversampling for your preprocessing pipeline
from sklearn.utils import resample
import pandas as pd

def preprocessing_oversampling(data, target_col, protected_attr):
    results = {}
    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group]
        
        if len(group_data[target_col].unique()) < 2:
            results[group] = group_data
            continue
            
        majority_class = group_data[target_col].mode()[0]
        majority = group_data[group_data[target_col] == majority_class]
        minority = group_data[group_data[target_col] != majority_class]
        
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
from imblearn.over_sampling import SMOTE
import numpy as np

def preprocessing_smote(X, y, sensitive_features):
    smote = SMOTE(random_state=42, k_neighbors=5)
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    n_original = len(X)
    sensitive_resampled = list(sensitive_features)
    
    # For synthetic samples, assign sensitive attributes based on nearest neighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    
    synthetic_samples = X_resampled[n_original:]
    distances, indices = nn.kneighbors(synthetic_samples)
    
    for index in indices.flatten():
        sensitive_resampled.append(sensitive_features[index])
        
    return X_resampled, y_resampled, np.array(sensitive_resampled)
            """, language="python")
        
        st.markdown("### Integration with Existing Preprocessing Workflow")
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_area(
                "How will you integrate this technique with your existing preprocessing steps?",
                placeholder="Example: 1. First apply correlation analysis to identify proxies. 2. Then use SMOTE to balance underrepresented intersectional groups. 3. Finally apply reweighting for fine-tuning. 4. Validate using fairness metrics.",
                key=f"integration_{technique.lower()}"
            )
        with col2:
            with st.popover("How-To"):
                st.markdown("""
                **Goal:** Create a clear, repeatable plan for applying your chosen technique.
                
                **1. Sequence:** Define the order of operations. (e.g., 'First, remove proxies, then resample').
                
                **2. Rationale:** Explain *why* you chose this sequence. (e.g., 'We remove proxies first so that SMOTE does not generate synthetic data based on biased correlations').
                
                **3. Validation:** State how you will measure success. (e.g., 'After applying SMOTE, we will re-run our demographic parity metric and expect it to be > 0.9').
                """)

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
    audit_results = run_fairness_audit(data_path, config)
    
    # Phase 2: Load and analyze data
    X, y, sensitive_attr = load_data(data_path)
    bias_analysis = analyze_dataset_bias(X, y, sensitive_attr)
    
    # Phase 3: Apply mitigation based on audit results
    if audit_results['requires_resampling']:
        if bias_analysis['imbalance_ratio'] > 5:
            # High imbalance - use SMOTE
            X, y, sensitive_attr = preprocessing_smote(X, y, sensitive_attr)
        else:
            # Moderate imbalance - use oversampling
            X, y = apply_oversampling(X, y)
    
    if audit_results['requires_reweighting']:
        sample_weights = apply_reweighting(X, y)
    else:
        sample_weights = None
    
    # Phase 4: Train with fairness constraints
    model = train_fair_model(X, y, sensitive_attr,
                           fairness_def=audit_results['selected_fairness_def'],
                           sample_weight=sample_weights)
    
    # Phase 5: Post-process if needed
    if audit_results['requires_post_processing']:
        model = apply_threshold_optimization(model, X, y, sensitive_attr)
    
    # Phase 6: Validate
    fairness_metrics = validate_complete_fairness(model, X, y, sensitive_attr)
    
    return model, fairness_metrics

def run_fairness_audit(data_path, config):
    \"\"\"Run the audit playbook programmatically\"\"\"
    # Historical context assessment
    hca_results = assess_historical_context(config['domain'])
    
    # Fairness definition selection
    fairness_def = select_fairness_definition(hca_results, config['use_case'])
    
    # Bias source identification
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
# ... [Paste all your other playbook/toolkit page functions here] ...
# e.g., preprocessing_fairness_toolkit(), inprocessing_fairness_toolkit(), etc.
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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Define a specific, measurable fairness rule that your model must follow.
            **Step 1: Choose a Fairness Metric**
            Select a metric that aligns with your fairness goals (e.g., Demographic Parity, Equalized Odds).
            **Step 2: Formulate the Constraint**
            Express the metric as a mathematical constraint. For Demographic Parity, the constraint is that the difference in the rate of positive outcomes between groups should be below a small threshold, ε.
            `|P(Ŷ=1|A=0) - P(Ŷ=1|A=1)| ≤ ε`
            **Step 3: Implement as a Penalty**
            In your code, create a function that calculates this disparity. This function will be your `C(θ)` in the Lagrangian formula.
            **Python Example (Conceptual):**
            ```python
            def demographic_parity_constraint(y_pred, sensitive_features):
                # P(Y_hat=1 | A=1)
                rate_privileged = y_pred[sensitive_features==1].mean()
                # P(Y_hat=1 | A=0)
                rate_unprivileged = y_pred[sensitive_features==0].mean()
                
                # The penalty is the difference
                return abs(rate_privileged - rate_unprivileged)

            # During training loop:
            # loss = original_loss + lambda * demographic_parity_constraint(...)
            ```
            """)

        st.markdown("**Feasibility and Trade-offs:**")
        with st.expander("🔍 Definition and Example"):
            st.write("It is not always possible to be perfectly fair and perfectly accurate at the same time. Often there is a 'trade-off'. Improving fairness can slightly reduce overall accuracy, and vice versa. It’s crucial to understand this balance.")
            st.write("**Intersectionality Example:** Forcing equal outcomes for all subgroups (e.g., Latina women, Asian men) may be mathematically impossible or require such a large sacrifice in accuracy that the model becomes unusable.")
        st.text_area("Apply to your case: What trade-off between accuracy and fairness are you willing to accept?", key="in_q2")
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Quantify and document the impact of fairness interventions on model performance.
            **Step 1: Establish Baselines**
            Train a model *without* fairness constraints. Record its accuracy, precision, recall, and fairness metrics. This is your baseline.
            **Step 2: Train Constrained Models**
            Train several versions of your model, each with a different fairness constraint strength (i.e., varying the `λ` parameter).
            **Step 3: Plot the Trade-off Curve**
            Create a scatter plot with the fairness metric on one axis and the accuracy metric on the other. Each point represents one of your trained models. This visualizes the Pareto frontier.
            **Step 4: Make a Decision**
            Based on the plot, select the model that offers the best compromise for your specific use case. For example, you might accept a 2% drop in accuracy to achieve a 50% reduction in fairness disparity. Document this decision and its rationale.
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Build a neural network architecture for adversarial training.
            **Method:** You need two distinct models that share some layers.
            1.  **Predictor Model:** Takes the input data and produces the main prediction.
            2.  **Adversary Model:** Takes the *output* (or an intermediate representation) from the predictor and tries to predict the sensitive attribute.
            **Python Example (PyTorch):**
            ```python
            import torch.nn as nn

            class Predictor(nn.Module):
                def __init__(self):
                    super(Predictor, self).__init__()
                    self.layer1 = nn.Linear(10, 32)
                    self.layer2 = nn.Linear(32, 1) # Predicts the main outcome
                def forward(self, x):
                    x = torch.relu(self.layer1(x))
                    return self.layer2(x)

            class Adversary(nn.Module):
                def __init__(self):
                    super(Adversary, self).__init__()
                    self.layer1 = nn.Linear(1, 16) # Takes predictor's output
                    self.layer2 = nn.Linear(16, 1) # Predicts sensitive attr
                def forward(self, x):
                    x = torch.relu(self.layer1(x))
                    return torch.sigmoid(self.layer2(x))
            
            predictor = Predictor()
            adversary = Adversary()
            ```
            """)

        st.markdown("**Optimization:**")
        with st.expander("🔍 Definition and Example"):
            st.write("Training can be unstable because the Predictor and Adversary have opposing objectives. Special techniques, like 'gradient reversal', are needed so the Predictor actively 'unlearns' bias.")
        st.text_area("Apply to your case: What optimization challenges do you foresee and how would you address them?", 
                     placeholder="E.g.: The adversary could become too strong at the start. We will use a gradual increase in its weight in the loss function.", 
                     key="in_q4")
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Train the predictor and adversary in a stable way.
            **Method:** The key is the **Gradient Reversal Layer (GRL)**. During the backward pass of training, this "layer" reverses the gradient flowing from the adversary to the predictor. This means that while the adversary is learning to get *better* at predicting the sensitive attribute, the predictor receives a signal to get *worse* at providing that information.
            **Python Example (Conceptual Training Loop):**
            ```python
            # alpha is the adversarial strength parameter
            for data, labels, sensitive in dataloader:
                # 1. Train the predictor
                predictor_optimizer.zero_grad()
                predictions = predictor(data)
                predictor_loss = predictor_criterion(predictions, labels)
                
                # 2. Train the adversary
                # Detach to avoid gradients flowing back to predictor here
                adversary_preds = adversary(predictions.detach())
                adversary_loss = adversary_criterion(adversary_preds, sensitive)
                adversary_loss.backward()
                adversary_optimizer.step()

                # 3. Train predictor to fool adversary (with gradient reversal)
                adversary_preds_for_predictor = adversary(predictions)
                adversary_loss_for_predictor = adversary_criterion(adversary_preds_for_predictor, sensitive)
                
                # The "trick": multiply by -alpha to reverse the gradient
                total_predictor_loss = predictor_loss - alpha * adversary_loss_for_predictor
                total_predictor_loss.backward()
                predictor_optimizer.step()
            ```
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Define and track multiple, sometimes competing, goals for your model.
            **Step 1: Define Your Objective Functions**
            Create separate Python functions for each objective you want to track.
            - `objective_1 = calculate_accuracy(y_true, y_pred)`
            - `objective_2 = calculate_demographic_parity(y_pred, sensitive_features)`
            - `objective_3 = calculate_equalized_odds(y_true, y_pred, sensitive_features)`
            **Step 2: Choose a Multi-Objective Algorithm**
            Specialized algorithms are needed to find the Pareto front. These are often found in libraries dedicated to multi-objective optimization.
            - **Scalarization:** The simplest method. Combine all objectives into a single formula with weights: `Loss = w1*Acc + w2*Fair1 + w3*Fair2`. By trying many different weights, you can trace out the Pareto front.
            - **Evolutionary Algorithms (e.g., NSGA-II):** More advanced methods that evolve a population of models to find the optimal set directly. Libraries like `DEAP` or `Pymoo` in Python can implement this.
            **Step 3: Document the Chosen Solution**
            Once you have the Pareto front, select a final model and document *why* you chose that specific trade-off point.
            """)

    # TAB 4: Code Patterns
    with tab4:
        st.subheader("Implementation Pattern Catalog")
        with st.expander("🔍 Friendly Definition"):
            st.write("These are code or pseudocode snippets showing how in-processing techniques look in practice. They serve as reusable templates for implementing fairness in your own code.")
        
        st.markdown("**Pattern 1: Regularized Loss Function**")
        st.code("""
# Example of a fairness-regularized loss function in a training loop
# (Works with any framework: TF, PyTorch, Scikit-learn)

lambda_fairness = 0.8 # Hyperparameter to tune

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(data)
    
    # Calculate standard performance loss
    performance_loss = standard_loss_function(predictions, labels)
    
    # Calculate fairness penalty
    fairness_penalty = demographic_parity_disparity(predictions, sensitive_attrs)
    
    # Combine the losses
    total_loss = performance_loss + lambda_fairness * fairness_penalty
    
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()
        """, language="python")

        st.markdown("**Pattern 2: Custom Fairlearn Reductions Model**")
        st.code("""
# Using the Fairlearn library for exponentiated gradient reduction
# This method finds a stochastic classifier that satisfies the constraint.
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# 1. Define the base estimator (your model)
estimator = LogisticRegression(solver='liblinear')

# 2. Define the fairness constraint
constraint = DemographicParity()

# 3. Create the fairness-aware model using the reduction
mitigator = ExponentiatedGradient(estimator, constraint)

# 4. Fit the model as usual
mitigator.fit(X_train, y_train, sensitive_features=A_train)

# 5. Make predictions
predictions = mitigator.predict(X_test)
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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Extend your fairness constraints to cover multiple intersectional subgroups.
            **Step 1: Define Intersectional Subgroups**
            Create a new feature in your data that represents the intersection of two or more sensitive attributes.
            ```python
            # Example: Creating an intersectional feature
            df['race_gender'] = df['race'] + '_' + df['gender']
            # Now you have groups like 'Black_Woman', 'White_Man', etc.
            ```
            **Step 2: Modify Your Constraint Function**
            Your fairness constraint function needs to calculate the disparity across *all* pairs of these new subgroups.
            ```python
            def max_intersectional_disparity(y_pred, intersectional_features):
                subgroup_rates = {}
                for group in intersectional_features.unique():
                    mask = (intersectional_features == group)
                    subgroup_rates[group] = y_pred[mask].mean()
                
                # The penalty is the difference between the max and min rates
                if not subgroup_rates:
                    return 0
                max_rate = max(subgroup_rates.values())
                min_rate = min(subgroup_rates.values())
                return max_rate - min_rate

            # Use this new function in your regularized loss
            # loss = perf_loss + lambda * max_intersectional_disparity(...)
            ```
            **Step 3: Monitor Trade-offs**
            Be aware that enforcing fairness across many small subgroups can be very difficult and may significantly impact overall accuracy. Track performance for each subgroup individually.
            """)

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
        st.info("Adjust classification thresholds after training to meet specific fairness definitions.")
        with st.expander("💡 Interactive Example"):
            run_threshold_simulation()
        
        st.text_area("Apply to your case: What fairness criterion will you use and how do you plan to analyze trade-offs?", 
                     placeholder="1. Criterion: Equal Opportunity.\n2. Calculation: Find thresholds that equalize TPR in a validation set.\n3. Deployment: Use a proxy for demographic group since we cannot use the protected attribute in production.", 
                     key="po_q1")
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Find the optimal decision thresholds for each group to satisfy a fairness constraint.
            **Step 1: Choose a Fairness Constraint**
            Select your target (e.g., `EqualizedOdds`, `DemographicParity`).
            **Step 2: Use an Optimizer**
            The `fairlearn` library provides `ThresholdOptimizer`, which finds the best thresholds for you based on your unmitigated model's scores.
            **Step 3: Fit the Optimizer**
            Train the optimizer on a validation set. It learns the group-specific thresholds needed to meet the fairness constraint.
            **Python Example (using Fairlearn):**
            ```python
            from fairlearn.postprocessing import ThresholdOptimizer
            from fairlearn.reductions import EqualizedOdds
            
            # 1. Train your original, unmitigated model
            # model.fit(X_train, y_train)
            
            # 2. Get the unmitigated model's scores on validation data
            # scores = model.predict_proba(X_val)[:, 1]
            
            # 3. Set up and fit the ThresholdOptimizer
            optimizer = ThresholdOptimizer(
                estimator=model,
                constraints="equalized_odds",
                objective="accuracy_score"
            )
            optimizer.fit(X_val, y_val, sensitive_features=A_val)
            
            # 4. Use the optimizer to make fair predictions
            # fair_predictions = optimizer.predict(X_test, sensitive_features=A_test)
            ```
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Ensure predicted probabilities are reliable across all groups.
            **Step 1: Evaluate Calibration per Group**
            - **Visualize:** Plot reliability diagrams (like in the simulation) for each demographic group separately.
            - **Quantify:** Calculate the Expected Calibration Error (ECE) for each group. A lower ECE is better.
            **Step 2: Apply a Calibration Method per Group**
            Fit a separate calibrator for each group on a validation set. This allows you to correct for group-specific miscalibration.
            **Python Example (using Scikit-learn):**
            ```python
            from sklearn.calibration import CalibratedClassifierCV

            # Assume model is your already-trained classifier
            # Assume X_cal, y_cal, A_cal are your calibration dataset
            
            calibrators = {}
            for group_id in A_cal.unique():
                # Filter data for the specific group
                mask = (A_cal == group_id)
                
                # Create and fit a calibrator for this group
                # 'isotonic' or 'sigmoid' (Platt)
                group_calibrator = CalibratedClassifierCV(
                    estimator=model, 
                    method='sigmoid', 
                    cv='prefit' # Use the already-trained model
                )
                group_calibrator.fit(X_cal[mask], y_cal[mask])
                calibrators[group_id] = group_calibrator
            
            # To predict, select the appropriate calibrator based on group
            # group = A_test.iloc[0]
            # calibrated_prob = calibrators[group].predict_proba(X_test.iloc[[0]])
            ```
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Align score distributions between groups to achieve demographic parity.
            **Method:** Quantile mapping is a common technique for distribution alignment. It adjusts the scores of one group so that its distribution matches the distribution of a reference group.
            **Step 1: Separate Scores by Group**
            From a validation set, get the predicted scores for your privileged and unprivileged groups.
            **Step 2: Learn the Quantile Mapping**
            For each score in the unprivileged group, find its quantile (percentile). Then, find the score in the privileged group that corresponds to the *same quantile*. This is the new, transformed score.
            **Python Example (Conceptual):**
            ```python
            import numpy as np

            # scores_priv and scores_unpriv are from a validation set
            def learn_quantile_map(scores_priv, scores_unpriv):
                # Sort the scores of the reference group
                sorted_priv = np.sort(scores_priv)
                
                def transform(score):
                    # Find the percentile of the new score in its own group
                    percentile = np.mean(scores_unpriv <= score)
                    # Find the corresponding score in the reference group
                    index = int(percentile * (len(sorted_priv) - 1))
                    return sorted_priv[index]
                
                return transform

            # Learn the transformation
            # score_transformer = learn_quantile_map(scores_priv_val, scores_unpriv_val)
            
            # Apply it to new data
            # transformed_scores_unpiv_test = [score_transformer(s) for s in scores_unpiv_test]
            ```
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Implement a system to defer low-confidence predictions to a human.
            **Step 1: Define the Rejection Region**
            Decide on the score range that will trigger a rejection. This can be a simple range (e.g., [0.4, 0.6]) or it can be group-specific if one group has more uncertainty.
            **Step 2: Analyze the Impact**
            - **Coverage:** What percentage of cases will be automated? (`1 - rejection_rate`)
            - **Accuracy/Fairness of Automated Decisions:** How good are the decisions on the non-rejected cases? Often, both accuracy and fairness are higher for this subset.
            - **Cost:** What is the cost of manual review for the rejected cases?
            **Step 3: Implement the Logic**
            Create a function that returns the model's decision *or* a "reject" flag.
            **Python Example:**
            ```python
            def reject_classifier(y_scores, lower_bound, upper_bound):
                predictions = []
                for score in y_scores:
                    if lower_bound <= score <= upper_bound:
                        # -1 can represent a rejected case
                        predictions.append(-1) 
                    elif score > upper_bound:
                        predictions.append(1) # Positive prediction
                    else:
                        predictions.append(0) # Negative prediction
                return np.array(predictions)
            
            # final_decisions = reject_classifier(model_scores, 0.4, 0.6)
            # human_review_cases = data[final_decisions == -1]
            ```
            """)

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
        with st.popover("How-To Guide"):
            st.markdown("""
            **Goal:** Apply post-processing techniques granularly to intersectional subgroups.
            **Method:** The core idea is to treat each intersectional subgroup as its own distinct group and apply the chosen post-processing method to it.
            **Step 1: Identify Subgroups**
            Create a combined feature for your intersectional groups (e.g., `race_gender`).
            **Step 2: Iterate and Apply**
            Loop through each unique subgroup. Within the loop, apply your chosen method (e.g., `ThresholdOptimizer`, `CalibratedClassifierCV`) using only the data for that subgroup.
            **Python Example (using `ThresholdOptimizer`):**
            ```python
            from fairlearn.postprocessing import ThresholdOptimizer

            # Assume 'race_gender' is your intersectional feature column
            
            # Store one optimizer for each subgroup
            optimizers = {}
            
            # Loop through all unique subgroups
            for group_id in X_val['race_gender'].unique():
                # Filter validation data for the current subgroup
                mask = (X_val['race_gender'] == group_id)
                X_val_group = X_val[mask]
                y_val_group = y_val[mask]
                
                # Fit an optimizer for this group
                optimizer = ThresholdOptimizer(estimator=model, ...)
                optimizer.fit(X_val_group, y_val_group)
                optimizers[group_id] = optimizer

            # When predicting, use the corresponding optimizer for the individual's subgroup
            # group = test_person['race_gender']
            # prediction = optimizers[group].predict(test_person)
            ```
            """)

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
#======================================================================
# --- MAIN APP LOGIC (The "Router") ---
# This is the central control point for the application.
#======================================================================

def main():
    """
    Main function to run the Streamlit app. Implements a router pattern.
    """
    st.sidebar.title("Playbook Selection")
    
    # This is the single, main navigation widget for the entire app.
    playbook_choice = st.sidebar.selectbox(
        "Choose the toolkit you want to use:",
        [
            "Fairness Audit Playbook",
            "Fairness Intervention Playbook",
            "Bias Mitigation Techniques",
            "Fairness Implementation Playbook",
            "Complete Integration Example"
        ],
        key="main_playbook_selector" # Added a unique key for the main navigation
    )



    # This 'if/elif' block is the router. It ensures ONLY ONE page function
    # is called per run, which prevents duplicate key errors.
    if playbook_choice == "Fairness Audit Playbook":
        audit_playbook()
    elif playbook_choice == "Fairness Intervention Playbook":
        intervention_playbook()
    elif playbook_choice == "Bias Mitigation Techniques":
        bias_mitigation_techniques_toolkit()
    elif playbook_choice == "Fairness Implementation Playbook":
        fairness_implementation_playbook()
    elif playbook_choice == "Complete Integration Example":
        complete_integration_example()

    # Add assistant widgets to the sidebar after the main navigation
    ai_playbook_assistant()
    research_assistant()

if __name__ == "__main__":
    main()

