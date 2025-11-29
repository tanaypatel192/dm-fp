"""
Database Models and Setup

SQLAlchemy models for storing predictions, batch jobs, and metrics
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from loguru import logger

Base = declarative_base()


class Prediction(Base):
    """Prediction records for auditing and analytics"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    # Input data
    patient_data = Column(JSON, nullable=False)
    # Model used
    model_name = Column(String(50), nullable=False, index=True)
    # Results
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # Low, Medium, High
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    # Performance
    processing_time_ms = Column(Float, nullable=True)
    # Additional data
    shap_values = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_model_created', 'model_name', 'created_at'),
        Index('idx_risk_level', 'risk_level'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "patient_data": self.patient_data,
            "model_name": self.model_name,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time_ms": self.processing_time_ms,
        }


class BatchJob(Base):
    """Batch processing jobs"""
    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True, index=True)
    # Job info
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    status = Column(String(20), nullable=False, index=True)  # pending, processing, completed, failed
    # Input
    total_patients = Column(Integer, nullable=False)
    model_name = Column(String(50), nullable=False)
    # Progress
    processed_patients = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)
    failed_predictions = Column(Integer, default=0)
    # Results summary
    low_risk_count = Column(Integer, default=0)
    medium_risk_count = Column(Integer, default=0)
    high_risk_count = Column(Integer, default=0)
    avg_probability = Column(Float, nullable=True)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    # Performance
    total_processing_time_ms = Column(Float, nullable=True)

    # Relationship to predictions
    predictions = relationship("BatchPrediction", back_populates="batch_job", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "status": self.status,
            "total_patients": self.total_patients,
            "processed_patients": self.processed_patients,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "low_risk_count": self.low_risk_count,
            "medium_risk_count": self.medium_risk_count,
            "high_risk_count": self.high_risk_count,
            "avg_probability": self.avg_probability,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_processing_time_ms": self.total_processing_time_ms,
        }


class BatchPrediction(Base):
    """Individual predictions within a batch job"""
    __tablename__ = "batch_predictions"

    id = Column(Integer, primary_key=True, index=True)
    batch_job_id = Column(Integer, ForeignKey("batch_jobs.id"), nullable=False, index=True)
    # Input
    patient_index = Column(Integer, nullable=False)
    patient_data = Column(JSON, nullable=False)
    # Results
    prediction = Column(Integer, nullable=True)
    probability = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)
    # Error handling
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    batch_job = relationship("BatchJob", back_populates="predictions")

    __table_args__ = (
        Index('idx_batch_patient', 'batch_job_id', 'patient_index'),
    )


class ModelMetric(Base):
    """Model performance metrics history"""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False, index=True)
    # Metrics
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    # Additional metrics
    confusion_matrix = Column(JSON, nullable=True)
    classification_report = Column(JSON, nullable=True)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_current = Column(Boolean, default=True, index=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_model_current', 'model_name', 'is_current'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_current": self.is_current,
        }


class APILog(Base):
    """API request logs for monitoring"""
    __tablename__ = "api_logs"

    id = Column(Integer, primary_key=True, index=True)
    # Request info
    method = Column(String(10), nullable=False)
    path = Column(String(255), nullable=False, index=True)
    query_params = Column(JSON, nullable=True)
    # Response info
    status_code = Column(Integer, nullable=False, index=True)
    response_time_ms = Column(Float, nullable=False)
    # Client info
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(String(255), nullable=True)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_path_status', 'path', 'status_code'),
        Index('idx_ip_created', 'ip_address', 'created_at'),
    )


# Database connection management
class DatabaseManager:
    """Async database manager"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_maker = None

    async def connect(self):
        """Initialize database connection"""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
            )

            self.session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("✓ Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_maker:
            raise RuntimeError("Database not connected")
        return self.session_maker()


# Global database instance
db_manager = DatabaseManager("sqlite+aiosqlite:///./diabetes_predictions.db")


# Dependency for FastAPI
async def get_db():
    """Dependency for getting database session"""
    async with db_manager.get_session() as session:
        try:
            yield session
        finally:
            await session.close()
