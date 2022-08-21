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

   stage('Deliver for development') {
            when {
                branch 'features'
            }
            steps {
                sh './jenkins/scripts/deliver-for-development.sh'
                input message: 'Finished using the web site? (Click "Proceed" to continue)'
                sh './jenkins/scripts/kill.sh'
            }
        }

   stage('Master merging'){
      
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
