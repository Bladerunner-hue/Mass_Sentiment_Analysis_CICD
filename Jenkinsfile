pipeline {
agent any

stages {

   stage('build') {

      steps {
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'docker build -t tender_matsumoto .'
         } 
         else if(env.BRANCH_NAME == 'main'){
            'not proper place'
         }

         }
      }
   }

      stage('build flask app'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'docker run -p 5000:5000 tender_matsumoto'
         }
         
         }
      }
   }

   stage('testing'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'python3 conftest.py'
         }
         
         }
      }
   }

   stage('release'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){

         echo 'release available'
         }
      
         }
      }
   }
   

   stage('Accepting next step'){
      
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

   stage('Master merging'){
      
      steps{
         script{
         def passwordVariable = 'gearsofwarhalo33'
         def usernameVariable = 'CenturyGhost'
           
         if(env.BRANCH_NAME == 'features'||env.BRANCH_NAME == 'main'){
         sh 'git checkout origin/features'
         sh 'git pull'
         sh 'git remote update'
         sh 'git fetch'
         sh 'git checkout origin/main'
         sh 'git merge origin/features'
         withCredentials([usernamePassword(credentialsId : 'GitHub', passwordVariable:'GIT_PASSWORD', usernameVariable:'GIT_USERNAME')]){
            sh 'git push http://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/CenturyGhost/rattrapage.git'
         }
         }}
   }
   }

   stage('container shutdown'){
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'docker rmi -f tender_matsumoto'
      }

      }
   }
 }

   }
}
