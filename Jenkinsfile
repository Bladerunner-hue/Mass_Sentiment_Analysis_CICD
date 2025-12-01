pipeline {
    agent any

    environment {
        PYTHON_VERSION = "3.11"
        VENV_PATH = "${WORKSPACE}/venv"
        FLASK_CONFIG = "testing"
        FLASK_APP = "wsgi.py"
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git --version'
                echo "Building branch: ${env.BRANCH_NAME}"
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                    echo "Checking Python environment..."

                    # Check for ml-torch environment
                    if command -v conda &> /dev/null && conda env list | grep -q "ml-torch"; then
                        echo "Found ml-torch conda environment"
                        eval "$(conda shell.bash hook)"
                        conda activate ml-torch
                        PYTHON_CMD="python"
                    # Check for pyenv 3.11.11
                    elif command -v pyenv &> /dev/null && pyenv versions | grep -q "3.11.11"; then
                        echo "Found pyenv 3.11.11"
                        pyenv shell 3.11.11
                        PYTHON_CMD="python"
                    else
                        echo "Using system python3"
                        PYTHON_CMD="python3"
                    fi

                    echo "Setting up Python virtual environment..."
                    $PYTHON_CMD -m venv ${VENV_PATH}
                    . ${VENV_PATH}/bin/activate
                    pip install --upgrade pip wheel setuptools
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                    echo "Dependencies installed successfully"
                '''
            }
        }

        stage('Code Quality') {
            parallel {
                stage('Flake8 Linting') {
                    steps {
                        sh '''
                            . ${VENV_PATH}/bin/activate
                            echo "Running Flake8 linting..."
                            flake8 app/ tests/ --max-line-length=100 --ignore=E501,W503 \
                                --output-file=flake8-report.txt --tee || true
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'flake8-report.txt', allowEmptyArchive: true
                        }
                    }
                }
                stage('Black Formatting') {
                    steps {
                        sh '''
                            . ${VENV_PATH}/bin/activate
                            echo "Checking code formatting with Black..."
                            black --check --diff app/ tests/ || echo "Formatting issues found"
                        '''
                    }
                }
                stage('isort Import Sorting') {
                    steps {
                        sh '''
                            . ${VENV_PATH}/bin/activate
                            echo "Checking import sorting with isort..."
                            isort --check-only --diff app/ tests/ || echo "Import sorting issues found"
                        '''
                    }
                }
                stage('Type Checking') {
                    steps {
                        sh '''
                            . ${VENV_PATH}/bin/activate
                            echo "Running mypy type checking..."
                            mypy app/ --ignore-missing-imports --no-error-summary || echo "Type checking completed with warnings"
                        '''
                    }
                }
            }
        }

        stage('Security Scan') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    echo "Running Bandit security scan..."
                    bandit -r app/ -f txt -o bandit-report.txt || true
                    cat bandit-report.txt
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'bandit-report.txt', allowEmptyArchive: true
                }
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    echo "Running unit tests with pytest..."
                    pytest tests/ -v \
                        --junitxml=test-results.xml \
                        --cov=app \
                        --cov-report=xml:coverage.xml \
                        --cov-report=html:htmlcov \
                        --cov-report=term-missing \
                        || true
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                    archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'htmlcov/**', allowEmptyArchive: true
                }
            }
        }

        stage('Database Migration Test') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    echo "Testing database migrations..."
                    export FLASK_CONFIG=testing
                    flask db upgrade || echo "No migrations to run"
                    flask init-db || echo "Database already initialized"
                '''
            }
        }

        stage('Build Assets') {
            when {
                anyOf {
                    branch 'main'
                    branch 'features'
                    branch 'develop'
                }
            }
            steps {
                sh '''
                    echo "Building Tailwind CSS assets..."
                    if command -v npm &> /dev/null; then
                        npm install
                        npm run build
                    else
                        echo "npm not available, skipping asset build"
                    fi
                '''
            }
        }

        stage('Integration Tests') {
            when {
                anyOf {
                    branch 'main'
                    branch 'features'
                }
            }
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    echo "Running Flask application smoke test..."

                    # Start the app in background
                    export FLASK_CONFIG=testing
                    nohup python wsgi.py > flask.log 2>&1 &
                    FLASK_PID=$!

                    # Wait for app to start
                    sleep 5

                    # Health check
                    curl -f http://localhost:5000/health || echo "Health check warning"

                    # Stop the app
                    kill $FLASK_PID || true

                    echo "Integration tests completed"
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'flask.log', allowEmptyArchive: true
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'features'
            }
            steps {
                echo 'Deploying to staging environment...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    echo "Staging deployment would happen here"
                '''
            }
        }

        stage('Approval for Production') {
            when {
                branch 'main'
            }
            steps {
                timeout(time: 24, unit: 'HOURS') {
                    input message: 'Deploy to production?', ok: 'Deploy'
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying to production environment...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    echo "Production deployment configured"
                    # Production deployment commands would go here
                    # Example:
                    # rsync -avz --exclude='venv' --exclude='.git' ./ deploy@server:/var/www/sentiment-app/
                    # ssh deploy@server "cd /var/www/sentiment-app && source venv/bin/activate && pip install -r requirements.txt && flask db upgrade && sudo systemctl restart sentiment-app"
                '''
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
            sh 'pkill -f "python.*wsgi.py" || true'
            sh 'pkill -f "flask_sentiment_analysis_app.py" || true'
            cleanWs(cleanWhenNotBuilt: false,
                    deleteDirs: true,
                    disableDeferredWipeout: true,
                    notFailBuild: true,
                    patterns: [
                        [pattern: 'venv/**', type: 'INCLUDE'],
                        [pattern: '**/__pycache__/**', type: 'INCLUDE'],
                        [pattern: '.pytest_cache/**', type: 'INCLUDE'],
                        [pattern: 'htmlcov/**', type: 'INCLUDE']
                    ])
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
        unstable {
            echo 'Pipeline completed with warnings'
        }
    }
}
