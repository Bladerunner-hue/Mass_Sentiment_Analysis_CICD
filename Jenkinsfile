pipeline {
agent any

stages {

stage('build') {

     steps {
        sh 'docker build -t tender_matsumoto .' 
        sh 'docker run -p 5000:5000 tender_matsumoto'
            }
}
 }
    }   
