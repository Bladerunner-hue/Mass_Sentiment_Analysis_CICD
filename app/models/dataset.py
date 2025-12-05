"""Dataset storage models for PostgreSQL.

Stores downloaded datasets from Kaggle, HuggingFace, and Twitter in PostgreSQL
instead of large CSV/Parquet files to avoid GitHub file size limits.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB

from app.extensions import db


class DatasetMetadata(db.Model):
    """Metadata about stored datasets."""
    
    __tablename__ = "dataset_metadata"
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)
    source = db.Column(db.String(50), nullable=False)  # kaggle, huggingface, twitter
    source_id = db.Column(db.String(255))  # Original dataset ID
    subset = db.Column(db.String(100))  # e.g., 'sentiment' for tweet_eval
    split = db.Column(db.String(50))  # train, test, validation
    num_rows = db.Column(db.Integer, default=0)
    columns = db.Column(JSONB)  # Column names and types
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    samples = db.relationship("DatasetSample", back_populates="dataset", 
                              cascade="all, delete-orphan", lazy="dynamic")
    
    __table_args__ = (
        Index("ix_dataset_metadata_source", "source"),
        Index("ix_dataset_metadata_source_id", "source_id"),
    )
    
    def __repr__(self) -> str:
        return f"<DatasetMetadata {self.name} ({self.source})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id,
            "subset": self.subset,
            "split": self.split,
            "num_rows": self.num_rows,
            "columns": self.columns,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatasetSample(db.Model):
    """Individual samples/rows from datasets.
    
    Stores text and label data for sentiment analysis training.
    """
    
    __tablename__ = "dataset_samples"
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset_metadata.id", ondelete="CASCADE"), 
                           nullable=False)
    
    # Core fields for sentiment analysis
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.Integer)  # Numeric label (0, 1, 2, etc.)
    label_text = db.Column(db.String(50))  # Text label (positive, negative, neutral)
    
    # Additional metadata (flexible JSONB for source-specific fields)
    extra_data = db.Column(JSONB)  # tweet_id, user_id, timestamps, etc.
    
    # Relationships
    dataset = db.relationship("DatasetMetadata", back_populates="samples")
    
    __table_args__ = (
        Index("ix_dataset_samples_dataset_id", "dataset_id"),
        Index("ix_dataset_samples_label", "label"),
    )
    
    def __repr__(self) -> str:
        return f"<DatasetSample {self.id} label={self.label}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "text": self.text,
            "label": self.label,
            "label_text": self.label_text,
            "extra_data": self.extra_data,
        }


class DatasetRepository:
    """Repository for dataset operations."""
    
    @staticmethod
    def create_dataset(
        name: str,
        source: str,
        source_id: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        columns: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> DatasetMetadata:
        """Create or update dataset metadata."""
        dataset = DatasetMetadata.query.filter_by(name=name).first()
        
        if dataset:
            # Update existing
            dataset.source = source
            dataset.source_id = source_id
            dataset.subset = subset
            dataset.split = split
            dataset.columns = columns
            dataset.description = description
            dataset.updated_at = datetime.utcnow()
        else:
            # Create new
            dataset = DatasetMetadata(
                name=name,
                source=source,
                source_id=source_id,
                subset=subset,
                split=split,
                columns=columns,
                description=description,
            )
            db.session.add(dataset)
        
        db.session.commit()
        return dataset
    
    @staticmethod
    def add_samples(
        dataset: DatasetMetadata,
        samples: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """Add samples to a dataset in batches.
        
        Args:
            dataset: Dataset metadata
            samples: List of dicts with 'text', 'label', 'label_text', 'metadata'
            batch_size: Commit every N samples
            
        Returns:
            Number of samples added
        """
        count = 0
        for i, sample_data in enumerate(samples):
            sample = DatasetSample(
                dataset_id=dataset.id,
                text=sample_data.get("text", ""),
                label=sample_data.get("label"),
                label_text=sample_data.get("label_text"),
                extra_data=sample_data.get("extra_data"),
            )
            db.session.add(sample)
            count += 1
            
            if (i + 1) % batch_size == 0:
                db.session.commit()
        
        # Commit remaining
        db.session.commit()
        
        # Update row count
        dataset.num_rows = DatasetSample.query.filter_by(dataset_id=dataset.id).count()
        db.session.commit()
        
        return count
    
    @staticmethod
    def get_dataset_by_name(name: str) -> Optional[DatasetMetadata]:
        """Get dataset by name."""
        return DatasetMetadata.query.filter_by(name=name).first()
    
    @staticmethod
    def get_samples(
        dataset_id: int,
        limit: Optional[int] = None,
        offset: int = 0,
        label: Optional[int] = None,
    ) -> List[DatasetSample]:
        """Get samples from a dataset."""
        query = DatasetSample.query.filter_by(dataset_id=dataset_id)
        
        if label is not None:
            query = query.filter_by(label=label)
        
        query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @staticmethod
    def get_all_samples_as_dataframe(dataset_id: int):
        """Get all samples as a pandas DataFrame for training."""
        import pandas as pd
        
        samples = DatasetSample.query.filter_by(dataset_id=dataset_id).all()
        
        return pd.DataFrame([
            {"text": s.text, "label": s.label, "label_text": s.label_text}
            for s in samples
        ])
    
    @staticmethod
    def list_datasets(source: Optional[str] = None) -> List[DatasetMetadata]:
        """List all datasets, optionally filtered by source."""
        query = DatasetMetadata.query
        
        if source:
            query = query.filter_by(source=source)
        
        return query.order_by(DatasetMetadata.created_at.desc()).all()
    
    @staticmethod
    def delete_dataset(dataset_id: int) -> bool:
        """Delete a dataset and all its samples."""
        dataset = DatasetMetadata.query.get(dataset_id)
        if dataset:
            db.session.delete(dataset)
            db.session.commit()
            return True
        return False
    
    @staticmethod
    def clear_samples(dataset_id: int) -> int:
        """Delete all samples from a dataset (for re-import)."""
        count = DatasetSample.query.filter_by(dataset_id=dataset_id).delete()
        
        dataset = DatasetMetadata.query.get(dataset_id)
        if dataset:
            dataset.num_rows = 0
            
        db.session.commit()
        return count
