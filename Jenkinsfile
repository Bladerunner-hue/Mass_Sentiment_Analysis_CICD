pipeline {
    agent any

    environment {
        VENV_PATH = "${WORKSPACE}/venv"
    }

    stages {
        stage('Setup Environment') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features') {
                        sh '''
                            python3 -m venv ${VENV_PATH}
                            . ${VENV_PATH}/bin/activate
                            pip install --upgrade pip
                            pip install -r requirements.txt
                        '''
                    }
                }
            }
        }

        stage('Run Flask App') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features') {
                        sh '''
                            . ${VENV_PATH}/bin/activate
                            nohup python flask_sentiment_analysis_app.py > flask.log 2>&1 &
                            sleep 5
                            echo "Flask app started on port 5000"
                        '''
                    }
                }
            }
        }

        stage('Testing') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features') {
                        sh '''
                            . ${VENV_PATH}/bin/activate
                            pytest conftest.py -v
                        '''
                    }
                }
            }
        }

        stage('Release') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features') {
                        echo 'Release available'
                    }
                }
            }
        }

        stage('Accepting next step') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features') {
                        input 'Proceed to live development?'
                    }
                }
            }
        }

        stage('Main Merging') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features' || env.BRANCH_NAME == 'main') {
                        sh 'git checkout features'
                        sh 'git pull'
                        sh 'git remote update'
                        sh 'git fetch'
                        sh 'git checkout origin/main'
                        sh 'git merge features'
                        sh 'git config user.email "alexandre.nouar@gmail.com"'
                        sh 'git config user.name "CenturyGhost"'
                        withCredentials([gitUsernamePassword(credentialsId: 'GitHubb')]) {
                            sh 'git push https://github.com/CenturyGhost/rattrapage.git'
                        }
                    }
                }
            }
        }

        stage('Cleanup') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'features') {
                        sh 'pkill -f flask_sentiment_analysis_app.py || true'
                        echo 'Flask app stopped'
                    }
                }
            }
        }
    }

    post {
        always {
            sh 'pkill -f flask_sentiment_analysis_app.py || true'
            cleanWs()
        }
    }
}
