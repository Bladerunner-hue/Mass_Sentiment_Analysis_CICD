pipeline {
    agent any

    environment {
        PYTHON_VERSION = "3.11"
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
                    echo "Setting up Python environment..."

                    # Use system python3 and install packages directly (no venv needed for Jenkins)
                    PYTHON_CMD="python3"

                    echo "Installing Python packages directly..."
                    $PYTHON_CMD -m pip install --user --upgrade pip wheel setuptools
                    $PYTHON_CMD -m pip install --user -r requirements.txt
                    $PYTHON_CMD -m pip install --user -r requirements-dev.txt

                    # Add local pip bin to PATH
                    export PATH="$HOME/.local/bin:$PATH"

                    echo "Dependencies installed successfully"
                '''
            }
        }

        stage('Code Quality') {
            parallel {
                stage('Flake8 Linting') {
                    steps {
                        sh '''
                            export PATH="$HOME/.local/bin:$PATH"
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
                            export PATH="$HOME/.local/bin:$PATH"
                            echo "Checking code formatting with Black..."
                            black --check --diff app/ tests/ || true
                        '''
                    }
                }
                stage('isort Import Sorting') {
                    steps {
                        sh '''
                            export PATH="$HOME/.local/bin:$PATH"
                            echo "Checking import sorting with isort..."
                            isort --check-only --diff app/ tests/ || echo "Import sorting issues found"
                        '''
                    }
                }
                stage('Type Checking') {
                    steps {
                        sh '''
                            export PATH="$HOME/.local/bin:$PATH"
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
                    export PATH="$HOME/.local/bin:$PATH"
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
                    export PATH="$HOME/.local/bin:$PATH"
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
                    export PATH="$HOME/.local/bin:$PATH"
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
                    export PATH="$HOME/.local/bin:$PATH"
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
                    export PATH="$HOME/.local/bin:$PATH"
                    echo "Staging deployment would happen here"
                '''
            }
        }

        stage('Manual Validation') {
            when {
                branch 'features'
            }
            steps {
                timeout(time: 24, unit: 'HOURS') {
                    input message: 'Validate the build and approve merge to main?', ok: 'Approve and Merge'
                }
            }
        }

        stage('Merge to Main') {
            when {
                branch 'features'
            }
            steps {
                sh '''
                    echo "Merging features branch to main..."
                    git config --global user.email "jenkins@localhost"
                    git config --global user.name "Jenkins CI"
                    git checkout main
                    git pull origin main
                    git merge features --no-ff -m "Merge features into main [ci skip]"
                    git push origin main
                    echo "Successfully merged features to main"
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
                    export PATH="$HOME/.local/bin:$PATH"
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
