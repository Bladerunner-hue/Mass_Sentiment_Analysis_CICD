pipeline {
agent any

stages {

   stage('build') {

     steps {
        sh 'docker build -t tender_matsumoto .'         
            }

   }

   stage('build flask app'){
      steps{
         sh 'docker run -p 5000:5000 tender_matsumoto'
   }
   }

   stage('container shutdown'){
      steps{
         sh 'docker rm -f tender_matsumoto'
      }
   }
 }
}   
