"""Main blueprint routes for web UI."""

from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user

from app.main import bp
from app.main.forms import SingleAnalysisForm, QuickAnalysisForm
from app.extensions import db
from app.models.analysis import SentimentAnalysis
from app.services.sentiment_service import SentimentService


# Sentiment service singleton
_sentiment_service = None


def get_sentiment_service():
    """Get or create sentiment service singleton."""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service


@bp.route('/')
def index():
    """Landing page."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('main/index.html')


@bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with statistics and recent analyses."""
    # Get user statistics
    stats = SentimentAnalysis.get_user_stats(current_user.id)

    # Get recent analyses
    recent = SentimentAnalysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        SentimentAnalysis.created_at.desc()
    ).limit(10).all()

    return render_template(
        'main/dashboard.html',
        stats=stats,
        recent_analyses=recent
    )


@bp.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """Single text analysis page."""
    form = SingleAnalysisForm()
    result = None

    if form.validate_on_submit():
        text = form.text.data
        include_emotions = form.include_emotions.data

        service = get_sentiment_service()

        if include_emotions:
            result = service.analyze_full(text)
        else:
            result = service.analyze_quick(text)

        # Save to database
        analysis = SentimentAnalysis.create_from_result(
            user_id=current_user.id,
            text=text,
            result=result,
            source='web'
        )
        db.session.add(analysis)
        db.session.commit()

        result['id'] = analysis.id

    return render_template(
        'main/analyze.html',
        form=form,
        result=result
    )


@bp.route('/analyze/quick', methods=['POST'])
@login_required
def quick_analyze():
    """AJAX endpoint for quick analysis."""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400

    service = get_sentiment_service()
    result = service.analyze_quick(text)

    return jsonify(result)


@bp.route('/history')
@login_required
def history():
    """Analysis history page with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = 20

    sentiment_filter = request.args.get('sentiment')
    source_filter = request.args.get('source')

    query = SentimentAnalysis.query.filter_by(user_id=current_user.id)

    if sentiment_filter:
        query = query.filter_by(sentiment=sentiment_filter)
    if source_filter:
        query = query.filter_by(source=source_filter)

    pagination = query.order_by(
        SentimentAnalysis.created_at.desc()
    ).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )

    return render_template(
        'main/history.html',
        analyses=pagination.items,
        pagination=pagination,
        sentiment_filter=sentiment_filter,
        source_filter=source_filter
    )


@bp.route('/history/<int:analysis_id>')
@login_required
def analysis_detail(analysis_id):
    """View single analysis details."""
    analysis = SentimentAnalysis.query.filter_by(
        id=analysis_id,
        user_id=current_user.id
    ).first_or_404()

    return render_template(
        'main/analysis_detail.html',
        analysis=analysis
    )


@bp.route('/history/<int:analysis_id>/delete', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """Delete an analysis."""
    analysis = SentimentAnalysis.query.filter_by(
        id=analysis_id,
        user_id=current_user.id
    ).first_or_404()

    db.session.delete(analysis)
    db.session.commit()

    flash('Analysis deleted successfully.', 'success')
    return redirect(url_for('main.history'))


@bp.route('/about')
def about():
    """About page."""
    return render_template('main/about.html')


@bp.route('/api-docs')
def api_docs():
    """Redirect to API documentation."""
    return redirect('/api/v1/docs')


@bp.route('/health')
def health():
    """Health check endpoint for load balancers and monitoring."""
    import datetime
    from app.extensions import db, cache

    checks = {
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'service': 'sentiment-analyzer',
        'version': '1.0.0',
        'checks': {}
    }

    # Database check
    try:
        db.session.execute(db.text('SELECT 1'))
        checks['checks']['database'] = {'status': 'ok', 'response_time_ms': 0}
    except Exception as e:
        checks['checks']['database'] = {'status': 'error', 'error': str(e)}
        checks['status'] = 'unhealthy'

    # Redis/Cache check
    try:
        cache.set('health_check', 'ok', timeout=1)
        result = cache.get('health_check')
        if result == 'ok':
            checks['checks']['cache'] = {'status': 'ok'}
        else:
            checks['checks']['cache'] = {
                'status': 'error', 'error': 'Cache not working'
            }
            checks['status'] = 'unhealthy'
    except Exception as e:
        checks['checks']['cache'] = {'status': 'error', 'error': str(e)}
        checks['status'] = 'unhealthy'

    # ML Model check
    try:
        service = get_sentiment_service()
        # Quick test with minimal text
        result = service.analyze_quick("test")
        if result and 'sentiment' in result:
            checks['checks']['ml_model'] = {'status': 'ok'}
        else:
            checks['checks']['ml_model'] = {
                'status': 'error', 'error': 'Invalid response'
            }
            checks['status'] = 'unhealthy'
    except Exception as e:
        checks['checks']['ml_model'] = {'status': 'error', 'error': str(e)}
        checks['status'] = 'unhealthy'

    # Celery check (if available)
    try:
        from app.extensions import celery
        # Simple task to check if Celery is responsive
        result = celery.control.inspect().active()
        if result:
            checks['checks']['celery'] = {'status': 'ok'}
        else:
            checks['checks']['celery'] = {
                'status': 'warning', 'message': 'No active workers'
            }
    except Exception as e:
        checks['checks']['celery'] = {'status': 'error', 'error': str(e)}

    status_code = 200 if checks['status'] == 'healthy' else 503
    return jsonify(checks), status_code
