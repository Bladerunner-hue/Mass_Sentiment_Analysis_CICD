pipeline {
agent any

stages {

stage('build') {

     steps {
        sh 'docker build -t name .' 
        sh 'docker run -p 5000:5000 name'
            }
}
 }
}
