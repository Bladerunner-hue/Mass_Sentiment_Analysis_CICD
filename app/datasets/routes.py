"""Routes for dataset management UI."""

import os
from pathlib import Path
from flask import (
    render_template, request, jsonify, flash, redirect, url_for, current_app
)
from flask_login import login_required

from app.datasets import bp
from app.services.dataset_service import DatasetService
from app.services.twitter_service import XTwitterService


def get_dataset_service():
    """Get or create the dataset service singleton."""
    if not hasattr(current_app, '_dataset_service'):
        current_app._dataset_service = DatasetService()
    return current_app._dataset_service


def get_twitter_service():
    """Get or create the X/Twitter service singleton."""
    if not hasattr(current_app, '_twitter_service'):
        current_app._twitter_service = XTwitterService()
    return current_app._twitter_service


# =============================================================================
# Main Pages
# =============================================================================

@bp.route('/')
@login_required
def index():
    """Dataset management dashboard."""
    dataset_service = get_dataset_service()
    local_datasets = dataset_service.list_local_datasets()
    
    # Check API configurations
    apis_configured = {
        "kaggle": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "huggingface": bool(os.environ.get("HUGGINGFACE_API_TOKEN")),
        "twitter": bool(os.environ.get("X_BEARER_TOKEN"))
    }
    
    return render_template(
        'datasets/index.html',
        local_datasets=local_datasets,
        apis_configured=apis_configured
    )


@bp.route('/browse')
@login_required
def browse():
    """Browse datasets from Kaggle and HuggingFace."""
    source = request.args.get('source', 'huggingface')
    query = request.args.get('q', '')
    
    return render_template(
        'datasets/browse.html',
        source=source,
        query=query
    )


@bp.route('/twitter')
@login_required
def twitter():
    """X/Twitter data collection interface."""
    twitter_service = get_twitter_service()
    
    return render_template(
        'datasets/twitter.html',
        is_configured=twitter_service.is_configured
    )


@bp.route('/local/<path:dataset_path>')
@login_required
def view_local(dataset_path):
    """View a local dataset."""
    dataset_service = get_dataset_service()
    
    # Find matching files
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    full_path = data_dir / dataset_path
    
    if not full_path.exists():
        flash(f"Dataset not found: {dataset_path}", "error")
        return redirect(url_for('datasets.index'))
    
    # Get file list
    files = []
    if full_path.is_dir():
        for f in full_path.glob("*"):
            if f.is_file() and f.suffix in ['.parquet', '.csv', '.json']:
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(data_dir)),
                    "size_mb": f.stat().st_size / 1024 / 1024
                })
    
    return render_template(
        'datasets/view_local.html',
        dataset_path=dataset_path,
        files=files
    )


# =============================================================================
# API Endpoints - HuggingFace
# =============================================================================

@bp.route('/api/hf/search')
@login_required
def api_hf_search():
    """Search HuggingFace datasets."""
    query = request.args.get('q', '')
    task = request.args.get('task')
    limit = request.args.get('limit', 20, type=int)
    sort = request.args.get('sort', 'downloads')
    
    dataset_service = get_dataset_service()
    results = dataset_service.search_hf_datasets(
        query=query,
        task=task,
        limit=limit,
        sort=sort
    )
    
    return jsonify({"datasets": results})


@bp.route('/api/hf/info/<path:dataset_id>')
@login_required
def api_hf_info(dataset_id):
    """Get HuggingFace dataset information."""
    dataset_service = get_dataset_service()
    info = dataset_service.get_hf_dataset_info(dataset_id)
    return jsonify(info)


@bp.route('/api/hf/download', methods=['POST'])
@login_required
def api_hf_download():
    """Download a HuggingFace dataset."""
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    subset = data.get('subset')
    split = data.get('split', 'train')
    
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    
    dataset_service = get_dataset_service()
    result = dataset_service.download_hf_dataset(
        dataset_id=dataset_id,
        subset=subset,
        split=split
    )
    
    return jsonify(result)


# =============================================================================
# API Endpoints - Kaggle
# =============================================================================

@bp.route('/api/kaggle/search')
@login_required
def api_kaggle_search():
    """Search Kaggle datasets."""
    query = request.args.get('q', '')
    sort_by = request.args.get('sort', 'hottest')
    limit = request.args.get('limit', 20, type=int)
    
    dataset_service = get_dataset_service()
    results = dataset_service.search_kaggle_datasets(
        query=query,
        sort_by=sort_by,
        limit=limit
    )
    
    return jsonify({"datasets": results})


@bp.route('/api/kaggle/info/<path:dataset_id>')
@login_required
def api_kaggle_info(dataset_id):
    """Get Kaggle dataset information."""
    dataset_service = get_dataset_service()
    info = dataset_service.get_kaggle_dataset_info(dataset_id)
    return jsonify(info)


@bp.route('/api/kaggle/download', methods=['POST'])
@login_required
def api_kaggle_download():
    """Download a Kaggle dataset."""
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    
    dataset_service = get_dataset_service()
    result = dataset_service.download_kaggle_dataset(dataset_id=dataset_id)
    
    return jsonify(result)


# =============================================================================
# API Endpoints - X/Twitter
# =============================================================================

@bp.route('/api/twitter/search', methods=['POST'])
@login_required
def api_twitter_search():
    """Search recent tweets."""
    data = request.get_json()
    query = data.get('query')
    max_results = data.get('max_results', 100)
    
    if not query:
        return jsonify({"error": "query is required"}), 400
    
    twitter_service = get_twitter_service()
    result = twitter_service.search_recent_tweets(
        query=query,
        max_results=max_results
    )
    
    return jsonify(result)


@bp.route('/api/twitter/user/<username>')
@login_required
def api_twitter_user(username):
    """Get user tweets."""
    max_results = request.args.get('max_results', 100, type=int)
    
    twitter_service = get_twitter_service()
    result = twitter_service.get_user_tweets(
        username=username,
        max_results=max_results
    )
    
    return jsonify(result)


@bp.route('/api/twitter/collect', methods=['POST'])
@login_required
def api_twitter_collect():
    """Collect tweets for sentiment analysis."""
    data = request.get_json()
    query = data.get('query')
    target_count = data.get('target_count', 1000)
    languages = data.get('languages', ['en'])
    
    if not query:
        return jsonify({"error": "query is required"}), 400
    
    twitter_service = get_twitter_service()
    result = twitter_service.collect_for_sentiment_analysis(
        query=query,
        target_count=target_count,
        languages=languages
    )
    
    return jsonify(result)


# =============================================================================
# API Endpoints - Local Dataset Management
# =============================================================================

@bp.route('/api/local/list')
@login_required
def api_local_list():
    """List local datasets."""
    dataset_service = get_dataset_service()
    datasets = dataset_service.list_local_datasets()
    return jsonify({"datasets": datasets})


@bp.route('/api/local/preview', methods=['POST'])
@login_required
def api_local_preview():
    """Preview a local dataset file."""
    data = request.get_json()
    path = data.get('path')
    num_rows = data.get('num_rows', 10)
    
    if not path:
        return jsonify({"error": "path is required"}), 400
    
    dataset_service = get_dataset_service()
    result = dataset_service.get_dataset_preview(path, num_rows)
    
    return jsonify(result)


@bp.route('/api/local/delete', methods=['POST'])
@login_required
def api_local_delete():
    """Delete a local dataset."""
    data = request.get_json()
    path = data.get('path')
    
    if not path:
        return jsonify({"error": "path is required"}), 400
    
    dataset_service = get_dataset_service()
    result = dataset_service.delete_dataset(path)
    
    return jsonify(result)


# =============================================================================
# API Endpoints - HuggingFace Upload
# =============================================================================

@bp.route('/api/hf/upload', methods=['POST'])
@login_required
def api_hf_upload():
    """Upload a local dataset to HuggingFace Hub."""
    data = request.get_json()
    local_path = data.get('local_path')
    repo_id = data.get('repo_id')
    repo_type = data.get('repo_type', 'dataset')
    private = data.get('private', False)
    
    if not local_path or not repo_id:
        return jsonify({"error": "local_path and repo_id are required"}), 400
    
    dataset_service = get_dataset_service()
    result = dataset_service.upload_to_huggingface(
        local_path=local_path,
        repo_id=repo_id,
        repo_type=repo_type,
        private=private
    )
    
    return jsonify(result)
