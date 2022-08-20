pipeline {
agent any

stages {

   stage('build') {

      steps {
         script
         if(env.BRANCH_NAME == 'features'){
         sh 'docker build -t tender_matsumoto .'
         }         
            }

   }

   stage('build flask app'){
      
      steps{
         script
         if(env.BRANCH_NAME == 'features'){
         sh 'docker run -p 5000:5000 tender_matsumoto'
         }
   }
   }

   stage('container shutdown'){
      steps{
         script
         if(env.BRANCH_NAME == 'features'){
         sh 'docker rmi -f tender_matsumoto'
      }}
   }
 }
}   
