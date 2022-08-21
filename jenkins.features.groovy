def build_app(){
  sh 'docker-compose up -d'
}

def run_app(){
  sh 'docker run -p 5000:5000 tender_matsumoto'
}

def release_app(){
  echo 'TALEX IS ON FIRE'
}

def accept_app(){
  input 'Proceed to live development ?'
}

def merge_app(){
  echo 'merging to master'
}

def stop_app(){
  sh 'docker rmi -f tender_matsumoto'
}

return this
