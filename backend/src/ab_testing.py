"""
A/B Testing Framework for Model Experimentation

This module provides comprehensive A/B testing capabilities for comparing
different model versions and analyzing their performance.

Features:
- Traffic splitting with configurable ratios
- User assignment to variants (sticky sessions)
- Performance metric tracking per variant
- Statistical significance testing
- Experiment management (start, stop, promote)
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
from scipy import stats
from loguru import logger


class ExperimentStatus(str, Enum):
    """Experiment status enum"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(str, Enum):
    """Variant type enum"""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class Variant:
    """A/B test variant configuration"""
    id: str
    name: str
    model_name: str  # Which model to use for this variant
    traffic_percentage: float  # 0-100
    variant_type: VariantType = VariantType.TREATMENT
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VariantMetrics:
    """Metrics for a single variant"""
    variant_id: str

    # Sample size
    total_users: int = 0
    total_predictions: int = 0

    # Performance metrics
    avg_prediction_time_ms: float = 0.0
    avg_confidence: float = 0.0

    # Prediction distribution
    positive_predictions: int = 0
    negative_predictions: int = 0

    # Risk level distribution
    low_risk_count: int = 0
    medium_risk_count: int = 0
    high_risk_count: int = 0

    # User engagement
    avg_interactions: float = 0.0
    conversion_rate: float = 0.0  # % of users who completed desired action

    # User satisfaction
    avg_rating: float = 0.0
    rating_count: int = 0

    # Errors
    error_count: int = 0
    error_rate: float = 0.0

    # Raw data for statistical testing
    prediction_times: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    conversions: List[int] = field(default_factory=list)  # 1 = converted, 0 = not
    ratings: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        # Don't include raw data in serialization (too large)
        data.pop('prediction_times', None)
        data.pop('confidence_scores', None)
        data.pop('conversions', None)
        data.pop('ratings', None)
        return data


@dataclass
class Experiment:
    """A/B test experiment configuration"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Configuration
    min_sample_size: int = 100  # Minimum samples before statistical testing
    confidence_level: float = 0.95  # 95% confidence
    target_metric: str = "conversion_rate"  # Primary metric to optimize

    # Metrics
    metrics: Dict[str, VariantMetrics] = field(default_factory=dict)

    # User assignments
    user_assignments: Dict[str, str] = field(default_factory=dict)  # user_id -> variant_id

    def __post_init__(self):
        """Initialize metrics for each variant"""
        if not self.metrics:
            self.metrics = {
                variant.id: VariantMetrics(variant_id=variant.id)
                for variant in self.variants
            }

    def to_dict(self) -> dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['ended_at'] = self.ended_at.isoformat() if self.ended_at else None
        data['variants'] = [v.to_dict() for v in self.variants]
        data['metrics'] = {k: v.to_dict() for k, v in self.metrics.items()}
        data.pop('user_assignments', None)  # Don't serialize user assignments
        return data


class ABTestManager:
    """
    A/B Testing Manager

    Handles experiment creation, user assignment, metric tracking,
    and statistical analysis.
    """

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
        logger.info("A/B Testing Manager initialized")

    # ==================== Experiment Management ====================

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        min_sample_size: int = 100,
        confidence_level: float = 0.95,
        target_metric: str = "conversion_rate"
    ) -> Experiment:
        """
        Create a new A/B test experiment

        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variant configurations
            min_sample_size: Minimum sample size before testing
            confidence_level: Statistical confidence level (0-1)
            target_metric: Primary metric to optimize

        Returns:
            Created experiment
        """
        # Validate traffic percentages
        total_traffic = sum(v['traffic_percentage'] for v in variants)
        if not (99.9 <= total_traffic <= 100.1):  # Allow small floating point errors
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")

        # Create experiment
        experiment_id = str(uuid.uuid4())
        variant_objects = [
            Variant(
                id=str(uuid.uuid4()),
                name=v['name'],
                model_name=v['model_name'],
                traffic_percentage=v['traffic_percentage'],
                variant_type=VariantType(v.get('variant_type', 'treatment')),
                description=v.get('description', '')
            )
            for v in variants
        ]

        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            target_metric=target_metric
        )

        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")

        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Can only start DRAFT experiments, current status: {experiment.status}")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        self.active_experiments.append(experiment_id)

        logger.info(f"Started experiment: {experiment.name} (ID: {experiment_id})")
        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Can only pause RUNNING experiments")

        experiment.status = ExperimentStatus.PAUSED
        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)

        logger.info(f"Paused experiment: {experiment.name}")
        return experiment

    def resume_experiment(self, experiment_id: str) -> Experiment:
        """Resume a paused experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Can only resume PAUSED experiments")

        experiment.status = ExperimentStatus.RUNNING
        if experiment_id not in self.active_experiments:
            self.active_experiments.append(experiment_id)

        logger.info(f"Resumed experiment: {experiment.name}")
        return experiment

    def stop_experiment(self, experiment_id: str, cancelled: bool = False) -> Experiment:
        """Stop an experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.CANCELLED if cancelled else ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.utcnow()

        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)

        logger.info(f"Stopped experiment: {experiment.name} (Status: {experiment.status})")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """List all experiments, optionally filtered by status"""
        experiments = list(self.experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        # Sort by created_at descending
        experiments.sort(key=lambda e: e.created_at, reverse=True)

        return experiments

    # ==================== User Assignment ====================

    def assign_user_to_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Tuple[str, str]:
        """
        Assign a user to a variant using consistent hashing

        Args:
            experiment_id: Experiment ID
            user_id: User identifier (IP, session ID, user ID, etc.)

        Returns:
            Tuple of (variant_id, model_name)
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment.name} is not running")

        # Check if user already assigned (sticky sessions)
        if user_id in experiment.user_assignments:
            variant_id = experiment.user_assignments[user_id]
            variant = next(v for v in experiment.variants if v.id == variant_id)
            return variant_id, variant.model_name

        # Assign user to variant using consistent hashing
        variant_id, model_name = self._hash_user_to_variant(
            user_id,
            experiment.variants
        )

        # Store assignment
        experiment.user_assignments[user_id] = variant_id

        logger.debug(f"Assigned user {user_id} to variant {variant_id} in experiment {experiment_id}")

        return variant_id, model_name

    def _hash_user_to_variant(
        self,
        user_id: str,
        variants: List[Variant]
    ) -> Tuple[str, str]:
        """
        Consistently hash user to variant based on traffic percentages

        Uses SHA-256 hashing for deterministic assignment
        """
        # Create hash of user_id
        hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)

        # Convert to percentage (0-100)
        percentage = (hash_value % 10000) / 100.0

        # Assign to variant based on traffic percentage
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_percentage
            if percentage < cumulative:
                return variant.id, variant.model_name

        # Fallback (should never reach here if percentages sum to 100)
        return variants[0].id, variants[0].model_name

    def get_user_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[Tuple[str, str]]:
        """Get user's assigned variant"""
        experiment = self.experiments.get(experiment_id)
        if not experiment or user_id not in experiment.user_assignments:
            return None

        variant_id = experiment.user_assignments[user_id]
        variant = next(v for v in experiment.variants if v.id == variant_id)
        return variant_id, variant.model_name

    # ==================== Metric Tracking ====================

    def track_prediction(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        prediction_time_ms: float,
        confidence: float,
        prediction: int,
        risk_level: str,
        error: bool = False
    ):
        """Track a prediction event"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return

        metrics = experiment.metrics.get(variant_id)
        if not metrics:
            return

        # Update counts
        metrics.total_predictions += 1

        # Track unique users
        if user_id not in experiment.user_assignments or \
           experiment.user_assignments[user_id] == variant_id:
            metrics.total_users = len([
                uid for uid, vid in experiment.user_assignments.items()
                if vid == variant_id
            ])

        if error:
            metrics.error_count += 1
        else:
            # Update performance metrics
            metrics.prediction_times.append(prediction_time_ms)
            metrics.avg_prediction_time_ms = np.mean(metrics.prediction_times)

            metrics.confidence_scores.append(confidence)
            metrics.avg_confidence = np.mean(metrics.confidence_scores)

            # Update prediction distribution
            if prediction == 1:
                metrics.positive_predictions += 1
            else:
                metrics.negative_predictions += 1

            # Update risk level distribution
            if risk_level.lower() == "low":
                metrics.low_risk_count += 1
            elif risk_level.lower() == "medium":
                metrics.medium_risk_count += 1
            elif risk_level.lower() == "high":
                metrics.high_risk_count += 1

        # Update error rate
        metrics.error_rate = metrics.error_count / metrics.total_predictions if metrics.total_predictions > 0 else 0.0

    def track_conversion(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        converted: bool
    ):
        """Track a conversion event"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return

        metrics = experiment.metrics.get(variant_id)
        if not metrics:
            return

        # Track conversion
        metrics.conversions.append(1 if converted else 0)

        # Update conversion rate
        if metrics.conversions:
            metrics.conversion_rate = np.mean(metrics.conversions) * 100

    def track_rating(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        rating: float
    ):
        """Track a user rating"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return

        metrics = experiment.metrics.get(variant_id)
        if not metrics:
            return

        # Track rating
        metrics.ratings.append(rating)
        metrics.rating_count = len(metrics.ratings)
        metrics.avg_rating = np.mean(metrics.ratings)

    def track_interaction(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str
    ):
        """Track a user interaction"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return

        metrics = experiment.metrics.get(variant_id)
        if not metrics:
            return

        # For simplicity, we're just incrementing total interactions
        # In a real implementation, you'd track per-user interactions
        metrics.avg_interactions += 1

    # ==================== Statistical Analysis ====================

    def calculate_statistical_significance(
        self,
        experiment_id: str,
        control_variant_id: str,
        treatment_variant_id: str,
        metric: str = "conversion_rate"
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between two variants

        Uses two-sample t-test or z-test depending on sample size

        Args:
            experiment_id: Experiment ID
            control_variant_id: Control variant ID
            treatment_variant_id: Treatment variant ID
            metric: Metric to compare

        Returns:
            Dictionary with statistical test results
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        control_metrics = experiment.metrics.get(control_variant_id)
        treatment_metrics = experiment.metrics.get(treatment_variant_id)

        if not control_metrics or not treatment_metrics:
            raise ValueError("Variant metrics not found")

        # Get data for the specified metric
        if metric == "conversion_rate":
            control_data = control_metrics.conversions
            treatment_data = treatment_metrics.conversions
        elif metric == "confidence":
            control_data = control_metrics.confidence_scores
            treatment_data = treatment_metrics.confidence_scores
        elif metric == "prediction_time":
            control_data = control_metrics.prediction_times
            treatment_data = treatment_metrics.prediction_times
        elif metric == "rating":
            control_data = control_metrics.ratings
            treatment_data = treatment_metrics.ratings
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Check if we have enough data
        if len(control_data) < experiment.min_sample_size or \
           len(treatment_data) < experiment.min_sample_size:
            return {
                "significant": False,
                "reason": "insufficient_data",
                "control_sample_size": len(control_data),
                "treatment_sample_size": len(treatment_data),
                "min_sample_size": experiment.min_sample_size
            }

        # Calculate means
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)

        # Calculate relative lift
        if control_mean > 0:
            relative_lift = ((treatment_mean - control_mean) / control_mean) * 100
        else:
            relative_lift = 0

        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

        # Check significance
        alpha = 1 - experiment.confidence_level
        significant = p_value < alpha

        # Calculate confidence interval for the difference
        pooled_std = np.sqrt(
            (np.var(control_data) / len(control_data)) +
            (np.var(treatment_data) / len(treatment_data))
        )

        critical_value = stats.t.ppf(1 - alpha/2, len(control_data) + len(treatment_data) - 2)
        margin_of_error = critical_value * pooled_std

        mean_diff = treatment_mean - control_mean
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error

        return {
            "significant": significant,
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "confidence_level": experiment.confidence_level,
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "mean_difference": float(mean_diff),
            "relative_lift_percent": float(relative_lift),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper)
            },
            "control_sample_size": len(control_data),
            "treatment_sample_size": len(treatment_data),
            "recommendation": self._get_recommendation(significant, relative_lift, p_value)
        }

    def _get_recommendation(
        self,
        significant: bool,
        relative_lift: float,
        p_value: float
    ) -> str:
        """Get recommendation based on test results"""
        if not significant:
            if p_value < 0.1:
                return "Trending towards significance. Consider collecting more data."
            return "No significant difference detected. Continue testing or abandon experiment."

        if relative_lift > 0:
            if relative_lift > 10:
                return "Strong positive lift detected! Consider promoting treatment variant."
            return "Positive lift detected. Monitor for consistency before promoting."
        else:
            if relative_lift < -10:
                return "Strong negative impact detected! Consider stopping experiment."
            return "Slight negative impact detected. Monitor closely."

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results with statistical analysis"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Find control variant
        control_variant = next(
            (v for v in experiment.variants if v.variant_type == VariantType.CONTROL),
            experiment.variants[0]  # Fallback to first variant
        )

        # Calculate statistical significance for each treatment variant
        results = {
            "experiment": experiment.to_dict(),
            "duration_hours": self._calculate_duration(experiment),
            "comparisons": []
        }

        for variant in experiment.variants:
            if variant.id == control_variant.id:
                continue

            try:
                comparison = self.calculate_statistical_significance(
                    experiment_id,
                    control_variant.id,
                    variant.id,
                    experiment.target_metric
                )
                comparison["variant_name"] = variant.name
                comparison["variant_id"] = variant.id
                results["comparisons"].append(comparison)
            except Exception as e:
                logger.error(f"Error calculating significance: {e}")

        # Determine winner
        results["winner"] = self._determine_winner(experiment, results["comparisons"])

        return results

    def _calculate_duration(self, experiment: Experiment) -> float:
        """Calculate experiment duration in hours"""
        if not experiment.started_at:
            return 0

        end_time = experiment.ended_at or datetime.utcnow()
        duration = end_time - experiment.started_at
        return duration.total_seconds() / 3600

    def _determine_winner(
        self,
        experiment: Experiment,
        comparisons: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine winning variant"""
        if not comparisons:
            return {"winner": None, "reason": "No comparisons available"}

        # Find variant with best significant improvement
        significant_improvements = [
            c for c in comparisons
            if c["significant"] and c["relative_lift_percent"] > 0
        ]

        if not significant_improvements:
            return {"winner": None, "reason": "No significant improvements detected"}

        # Sort by relative lift
        winner = max(significant_improvements, key=lambda c: c["relative_lift_percent"])

        return {
            "winner": winner["variant_id"],
            "variant_name": winner["variant_name"],
            "lift": winner["relative_lift_percent"],
            "confidence": winner["confidence_level"]
        }

    def promote_variant(
        self,
        experiment_id: str,
        variant_id: str
    ) -> Dict[str, Any]:
        """
        Promote a variant to production

        This would typically update your model serving configuration
        to use the winning variant's model
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        variant = next((v for v in experiment.variants if v.id == variant_id), None)
        if not variant:
            raise ValueError(f"Variant {variant_id} not found")

        # Stop the experiment
        self.stop_experiment(experiment_id)

        logger.info(f"Promoted variant {variant.name} (model: {variant.model_name}) to production")

        return {
            "promoted_variant": variant.to_dict(),
            "model_name": variant.model_name,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
ab_test_manager = ABTestManager()
