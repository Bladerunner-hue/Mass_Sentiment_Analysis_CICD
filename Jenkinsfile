pipeline {
agent any

stages {

   stage('build') {

      steps {
         script{
         if(env.BRANCH_NAME == 'features'||env.BRANCH_NAME == 'main'){
         sh 'docker build -t tender_matsumoto .'
         }         
            }}

   }

   stage('build flask app'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'||env.BRANCH_NAME == 'main'){
         sh 'docker run -p 5000:5000 tender_matsumoto'
         }}
   }
   }

   stage('release'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         echo 'TALEX IS ON FIRE'
         }}
   }
   }

   stage('Acceptance tests'){
      
      steps{
         script{
            // Variables for input
                    def inputConfig
                    def inputTest
         if(env.BRANCH_NAME == 'features'){
         input 'Proceed to live development ?'
         }}
   }
   }

   stage('container shutdown'){
      steps{
         script{
         if(env.BRANCH_NAME == 'features'||env.BRANCH_NAME == 'main'){
         sh 'docker rmi -f tender_matsumoto'
      }}
   }
 }
}   
}
