"""
A/B Testing API Endpoints

FastAPI routes for A/B testing management and analytics
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Header, Query
from pydantic import BaseModel, Field

from .ab_testing import (
    ab_test_manager,
    ExperimentStatus,
    VariantType,
)
from loguru import logger


router = APIRouter(prefix="/api/ab-testing", tags=["A/B Testing"])


# ==================== Request/Response Models ====================

class VariantCreate(BaseModel):
    """Request model for creating a variant"""
    name: str = Field(..., description="Variant name")
    model_name: str = Field(..., description="Model to use for this variant")
    traffic_percentage: float = Field(..., ge=0, le=100, description="Traffic percentage (0-100)")
    variant_type: str = Field(default="treatment", description="Variant type: control or treatment")
    description: str = Field(default="", description="Variant description")


class ExperimentCreate(BaseModel):
    """Request model for creating an experiment"""
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    variants: List[VariantCreate] = Field(..., min_items=2, description="List of variants (min 2)")
    min_sample_size: int = Field(default=100, ge=10, description="Minimum sample size")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level")
    target_metric: str = Field(default="conversion_rate", description="Primary metric to optimize")


class ExperimentResponse(BaseModel):
    """Response model for experiment"""
    id: str
    name: str
    description: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    min_sample_size: int
    confidence_level: float
    target_metric: str
    variants: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class VariantAssignment(BaseModel):
    """Response model for variant assignment"""
    experiment_id: str
    variant_id: str
    model_name: str
    user_id: str


class PredictionTracking(BaseModel):
    """Request model for tracking a prediction"""
    experiment_id: str
    variant_id: str
    user_id: str
    prediction_time_ms: float
    confidence: float
    prediction: int
    risk_level: str
    error: bool = False


class ConversionTracking(BaseModel):
    """Request model for tracking a conversion"""
    experiment_id: str
    variant_id: str
    user_id: str
    converted: bool


class RatingTracking(BaseModel):
    """Request model for tracking a rating"""
    experiment_id: str
    variant_id: str
    user_id: str
    rating: float = Field(..., ge=0, le=5)


class InteractionTracking(BaseModel):
    """Request model for tracking an interaction"""
    experiment_id: str
    variant_id: str
    user_id: str


class StatisticalTest(BaseModel):
    """Request model for statistical significance test"""
    control_variant_id: str
    treatment_variant_id: str
    metric: str = Field(default="conversion_rate")


class PromoteVariant(BaseModel):
    """Request model for promoting a variant"""
    variant_id: str


# ==================== Helper Functions ====================

def get_user_identifier(request: Request, x_user_id: Optional[str] = None) -> str:
    """
    Get user identifier from request

    Priority:
    1. X-User-ID header
    2. Session cookie
    3. Client IP address
    """
    if x_user_id:
        return x_user_id

    # Check session cookie
    session_id = request.cookies.get("session_id")
    if session_id:
        return f"session:{session_id}"

    # Fallback to IP address
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


# ==================== Experiment Management Endpoints ====================

@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(experiment: ExperimentCreate):
    """
    Create a new A/B test experiment

    Creates an experiment in DRAFT status. Use /experiments/{id}/start to begin testing.
    """
    try:
        # Convert variants to dict format
        variants_data = [v.dict() for v in experiment.variants]

        # Create experiment
        exp = ab_test_manager.create_experiment(
            name=experiment.name,
            description=experiment.description,
            variants=variants_data,
            min_sample_size=experiment.min_sample_size,
            confidence_level=experiment.confidence_level,
            target_metric=experiment.target_metric
        )

        return ExperimentResponse(**exp.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create experiment")


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List all experiments

    Optionally filter by status: draft, running, paused, completed, cancelled
    """
    try:
        status_filter = ExperimentStatus(status) if status else None
        experiments = ab_test_manager.list_experiments(status=status_filter)

        return [ExperimentResponse(**exp.to_dict()) for exp in experiments]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to list experiments")


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    experiment = ab_test_manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return ExperimentResponse(**experiment.to_dict())


@router.post("/experiments/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(experiment_id: str):
    """Start an experiment"""
    try:
        experiment = ab_test_manager.start_experiment(experiment_id)
        return ExperimentResponse(**experiment.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to start experiment")


@router.post("/experiments/{experiment_id}/pause", response_model=ExperimentResponse)
async def pause_experiment(experiment_id: str):
    """Pause a running experiment"""
    try:
        experiment = ab_test_manager.pause_experiment(experiment_id)
        return ExperimentResponse(**experiment.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error pausing experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause experiment")


@router.post("/experiments/{experiment_id}/resume", response_model=ExperimentResponse)
async def resume_experiment(experiment_id: str):
    """Resume a paused experiment"""
    try:
        experiment = ab_test_manager.resume_experiment(experiment_id)
        return ExperimentResponse(**experiment.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error resuming experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume experiment")


@router.post("/experiments/{experiment_id}/stop", response_model=ExperimentResponse)
async def stop_experiment(
    experiment_id: str,
    cancelled: bool = Query(default=False, description="Mark as cancelled instead of completed")
):
    """Stop an experiment"""
    try:
        experiment = ab_test_manager.stop_experiment(experiment_id, cancelled=cancelled)
        return ExperimentResponse(**experiment.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop experiment")


@router.delete("/experiments/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    try:
        if experiment_id not in ab_test_manager.experiments:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Only allow deletion of non-running experiments
        experiment = ab_test_manager.experiments[experiment_id]
        if experiment.status == ExperimentStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running experiment. Stop it first."
            )

        del ab_test_manager.experiments[experiment_id]

        if experiment_id in ab_test_manager.active_experiments:
            ab_test_manager.active_experiments.remove(experiment_id)

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete experiment")


# ==================== User Assignment Endpoints ====================

@router.get("/experiments/{experiment_id}/assign", response_model=VariantAssignment)
async def assign_user(
    experiment_id: str,
    request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """
    Assign user to a variant

    Returns the assigned variant and model to use.
    Uses X-User-ID header, session cookie, or IP address for identification.
    """
    try:
        # Get user identifier
        user_id = get_user_identifier(request, x_user_id)

        # Check if already assigned
        existing = ab_test_manager.get_user_variant(experiment_id, user_id)
        if existing:
            variant_id, model_name = existing
        else:
            # Assign to variant
            variant_id, model_name = ab_test_manager.assign_user_to_variant(
                experiment_id,
                user_id
            )

        return VariantAssignment(
            experiment_id=experiment_id,
            variant_id=variant_id,
            model_name=model_name,
            user_id=user_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error assigning user: {e}")
        raise HTTPException(status_code=500, detail="Failed to assign user")


@router.get("/experiments/{experiment_id}/variant", response_model=Optional[VariantAssignment])
async def get_user_variant(
    experiment_id: str,
    request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """
    Get user's assigned variant

    Returns None if user not yet assigned.
    """
    try:
        user_id = get_user_identifier(request, x_user_id)
        assignment = ab_test_manager.get_user_variant(experiment_id, user_id)

        if not assignment:
            return None

        variant_id, model_name = assignment

        return VariantAssignment(
            experiment_id=experiment_id,
            variant_id=variant_id,
            model_name=model_name,
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"Error getting user variant: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user variant")


# ==================== Tracking Endpoints ====================

@router.post("/track/prediction", status_code=204)
async def track_prediction(tracking: PredictionTracking):
    """Track a prediction event"""
    try:
        ab_test_manager.track_prediction(
            experiment_id=tracking.experiment_id,
            variant_id=tracking.variant_id,
            user_id=tracking.user_id,
            prediction_time_ms=tracking.prediction_time_ms,
            confidence=tracking.confidence,
            prediction=tracking.prediction,
            risk_level=tracking.risk_level,
            error=tracking.error
        )
        return None

    except Exception as e:
        logger.error(f"Error tracking prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to track prediction")


@router.post("/track/conversion", status_code=204)
async def track_conversion(tracking: ConversionTracking):
    """Track a conversion event"""
    try:
        ab_test_manager.track_conversion(
            experiment_id=tracking.experiment_id,
            variant_id=tracking.variant_id,
            user_id=tracking.user_id,
            converted=tracking.converted
        )
        return None

    except Exception as e:
        logger.error(f"Error tracking conversion: {e}")
        raise HTTPException(status_code=500, detail="Failed to track conversion")


@router.post("/track/rating", status_code=204)
async def track_rating(tracking: RatingTracking):
    """Track a user rating"""
    try:
        ab_test_manager.track_rating(
            experiment_id=tracking.experiment_id,
            variant_id=tracking.variant_id,
            user_id=tracking.user_id,
            rating=tracking.rating
        )
        return None

    except Exception as e:
        logger.error(f"Error tracking rating: {e}")
        raise HTTPException(status_code=500, detail="Failed to track rating")


@router.post("/track/interaction", status_code=204)
async def track_interaction(tracking: InteractionTracking):
    """Track a user interaction"""
    try:
        ab_test_manager.track_interaction(
            experiment_id=tracking.experiment_id,
            variant_id=tracking.variant_id,
            user_id=tracking.user_id
        )
        return None

    except Exception as e:
        logger.error(f"Error tracking interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to track interaction")


# ==================== Analytics Endpoints ====================

@router.post("/experiments/{experiment_id}/analyze", response_model=Dict[str, Any])
async def analyze_experiment(
    experiment_id: str,
    test: StatisticalTest
):
    """
    Calculate statistical significance between variants

    Performs hypothesis testing to determine if treatment variant
    is significantly different from control variant.
    """
    try:
        result = ab_test_manager.calculate_statistical_significance(
            experiment_id=experiment_id,
            control_variant_id=test.control_variant_id,
            treatment_variant_id=test.treatment_variant_id,
            metric=test.metric
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze experiment")


@router.get("/experiments/{experiment_id}/results", response_model=Dict[str, Any])
async def get_experiment_results(experiment_id: str):
    """
    Get comprehensive experiment results

    Returns detailed results including:
    - Experiment configuration and metrics
    - Statistical comparisons between variants
    - Winner determination
    - Recommendations
    """
    try:
        results = ab_test_manager.get_experiment_results(experiment_id)
        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting experiment results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get experiment results")


@router.get("/experiments/{experiment_id}/metrics", response_model=Dict[str, Any])
async def get_experiment_metrics(experiment_id: str):
    """Get current metrics for all variants"""
    try:
        experiment = ab_test_manager.get_experiment(experiment_id)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return {
            "experiment_id": experiment_id,
            "metrics": {vid: metrics.to_dict() for vid, metrics in experiment.metrics.items()},
            "total_users": len(experiment.user_assignments),
            "total_predictions": sum(m.total_predictions for m in experiment.metrics.values())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get experiment metrics")


# ==================== Promotion Endpoints ====================

@router.post("/experiments/{experiment_id}/promote", response_model=Dict[str, Any])
async def promote_variant(
    experiment_id: str,
    promote: PromoteVariant
):
    """
    Promote a variant to production

    Stops the experiment and marks the variant as the winner.
    In production, this would update model serving configuration.
    """
    try:
        result = ab_test_manager.promote_variant(experiment_id, promote.variant_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error promoting variant: {e}")
        raise HTTPException(status_code=500, detail="Failed to promote variant")


# ==================== Admin Endpoints ====================

@router.get("/admin/stats", response_model=Dict[str, Any])
async def get_admin_stats():
    """Get overall A/B testing statistics"""
    try:
        total_experiments = len(ab_test_manager.experiments)
        running_experiments = len(ab_test_manager.active_experiments)

        experiments_by_status = {}
        for status in ExperimentStatus:
            count = len([
                e for e in ab_test_manager.experiments.values()
                if e.status == status
            ])
            experiments_by_status[status.value] = count

        total_users = sum(
            len(e.user_assignments)
            for e in ab_test_manager.experiments.values()
        )

        total_predictions = sum(
            sum(m.total_predictions for m in e.metrics.values())
            for e in ab_test_manager.experiments.values()
        )

        return {
            "total_experiments": total_experiments,
            "running_experiments": running_experiments,
            "experiments_by_status": experiments_by_status,
            "total_users_assigned": total_users,
            "total_predictions_tracked": total_predictions,
        }

    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get admin stats")


@router.get("/admin/experiments/active", response_model=List[ExperimentResponse])
async def get_active_experiments():
    """Get all currently running experiments"""
    try:
        experiments = [
            ab_test_manager.experiments[exp_id]
            for exp_id in ab_test_manager.active_experiments
            if exp_id in ab_test_manager.experiments
        ]

        return [ExperimentResponse(**exp.to_dict()) for exp in experiments]

    except Exception as e:
        logger.error(f"Error getting active experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active experiments")
