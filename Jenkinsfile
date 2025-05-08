pipeline {
    agent {
        docker {
            image 'python:3.9-slim'
            args '-u root -v /var/run/docker.sock:/var/run/docker.sock'
        }
    }
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('maleksoula')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0'
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Test') {
            steps {
                // Run your tests here (you should add tests first)
                sh 'python -m pytest tests/ '
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("maleksoula/ImageSearchEngine:${env.BUILD_ID}")
                }
            }
        }
        
        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'maleksoula') {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    //  Run locally with Docker
                    sh 'docker run -d -p 5423:5423 -v ./images:/app/images $ImageSearchEngine:latest'
                }
            }
        }
    }
    
    post {
        always {
            // Clean up
            echo 'Cleaning up workspace...'
            cleanWs()
        }
    }
}