pipeline {
agent any

stages {

   stage('build') {

      steps {
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'docker build -t VaderCICD .'
         } 
         else if(env.BRANCH_NAME == 'main'){
            'not proper place'
         }

         }
      }
   }
//change change change oui

      stage('build flask app'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'docker run -p 5000:5000 VaderCICD'
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

   stage('main merging'){
      
      steps{
         script{

            // Variables for input
         if(env.BRANCH_NAME == 'features'||env.BRANCH_NAME == 'main'){
         sh 'git checkout features'
         sh 'git pull'
         sh 'git remote update'
         sh 'git fetch'
         sh 'git checkout origin/main'
         sh 'git merge features'
         sh "git config user.email \"alexandre.nouar@gmail.com\""
         sh "git config user.name \"CenturyGhost\""
withCredentials([gitUsernamePassword(credentialsId:'GitHubb')]) {

sh 'git push https://github.com/CenturyGhost/rattrapage.git'

               }
            }
         }
      }
   }
   

   stage('container shutdown'){
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         sh 'docker rmi -f VaderCICD'
      }

      }
   }
 }

   }
}
