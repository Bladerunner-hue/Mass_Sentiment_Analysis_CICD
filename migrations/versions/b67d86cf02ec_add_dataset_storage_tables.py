"""Add dataset storage tables

Revision ID: b67d86cf02ec
Revises: 
Create Date: 2025-12-05 12:44:50.298658

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'b67d86cf02ec'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create dataset_metadata table
    op.create_table('dataset_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('source_id', sa.String(length=255), nullable=True),
        sa.Column('subset', sa.String(length=100), nullable=True),
        sa.Column('split', sa.String(length=50), nullable=True),
        sa.Column('num_rows', sa.Integer(), nullable=True),
        sa.Column('columns', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_dataset_metadata_source', 'dataset_metadata', ['source'], unique=False)
    op.create_index('ix_dataset_metadata_source_id', 'dataset_metadata', ['source_id'], unique=False)

    # Create dataset_samples table
    op.create_table('dataset_samples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('label', sa.Integer(), nullable=True),
        sa.Column('label_text', sa.String(length=50), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['dataset_metadata.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_dataset_samples_dataset_id', 'dataset_samples', ['dataset_id'], unique=False)
    op.create_index('ix_dataset_samples_label', 'dataset_samples', ['label'], unique=False)

    # Add stream_metadata column to sentiment_analyses if it doesn't exist
    try:
        op.add_column('sentiment_analyses', sa.Column('stream_metadata', sa.JSON(), nullable=True))
    except Exception:
        pass  # Column may already exist


def downgrade():
    op.drop_index('ix_dataset_samples_label', table_name='dataset_samples')
    op.drop_index('ix_dataset_samples_dataset_id', table_name='dataset_samples')
    op.drop_table('dataset_samples')
    
    op.drop_index('ix_dataset_metadata_source_id', table_name='dataset_metadata')
    op.drop_index('ix_dataset_metadata_source', table_name='dataset_metadata')
    op.drop_table('dataset_metadata')
    
    try:
        op.drop_column('sentiment_analyses', 'stream_metadata')
    except Exception:
        pass
