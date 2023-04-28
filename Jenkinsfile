pipeline {
agent any

stages {

   stage('build') {

      steps {
         script{
         if(env.BRANCH_NAME == 'features'){
         bat 'docker build -t VaderCICD'
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
         bat 'docker run -p 5000:5000 VaderCICD'
         }
         
         }
      }
   }

   stage('testing'){
      
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         bat 'python3 conftest.py'
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
         bat 'git checkout features'
         bat 'git pull'
         bat 'git remote update'
         bat 'git fetch'
         bat 'git checkout origin/main'
         bat 'git merge features'
         bat "git config user.email \"alexandre.nouar@gmail.com\""
         bat "git config user.name \"CenturyGhost\""
withCredentials([gitUsernamePassword(credentialsId:'GitHubb')]) {

bat 'git push https://github.com/CenturyGhost/rattrapage.git'

               }
            }
         }
      }
   }
   

   stage('container shutdown'){
      steps{
         script{
         if(env.BRANCH_NAME == 'features'){
         bat 'docker rmi -f VaderCICD'
      }

      }
   }
 }

   }
}
