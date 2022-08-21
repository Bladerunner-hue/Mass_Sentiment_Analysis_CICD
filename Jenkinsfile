def groovyfile
pipeline {
agent any

stages {

   stage('build') {

      steps {
         script{
         def filename = 'jenkins' + env.BRANCH_NAME + '.groovy'
         groovyfile = load filename
         groovyfile.build_app()         
      }
   }
}

   stage('build flask app'){
      
      steps{
         script{
         groovyfile.run_app()
      }
   }
}

   stage('release'){
      
      steps{
         script{
         groovyfile.release_app()
      }
   }
}

   stage('Accepting next step'){
      
      steps{
         script{
         groovyfile.accept_app()   
      }
   }
}
   stage('Master merging'){
      
      steps{
         script{
         groovyfile.merge_app()
      }
   }
}

   stage('container shutdown'){
      steps{
         script{
         groovyfile.stop_app()
     }
   }
 }
}   
}
