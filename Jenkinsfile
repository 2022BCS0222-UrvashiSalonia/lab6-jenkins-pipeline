pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = "2022bcs0222urvashi/ml-model"
        CURRENT_ACCURACY = ""
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Cloning repository...'
                checkout scm
            }
        }
        
        stage('Setup Python Virtual Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo 'Training model...'
                sh '''
                    . venv/bin/activate
                    python train.py
                '''
            }
        }
        
        stage('Read Accuracy') {
            steps {
                echo 'Reading accuracy from metrics.json...'
                script {
                    def metricsFile = readFile('app/artifacts/metrics.json')
                    def metrics = readJSON text: metricsFile
                    CURRENT_ACCURACY = metrics.accuracy
                    echo "Current Model Accuracy: ${CURRENT_ACCURACY}"
                }
            }
        }
        
        stage('Compare Accuracy') {
            steps {
                echo 'Comparing accuracy with baseline...'
                script {
                    def bestAccuracy = 0.0
                    try {
                        bestAccuracy = credentials('best-accuracy').toDouble()
                    } catch (Exception e) {
                        echo "No baseline accuracy found. Setting to 0.0"
                        bestAccuracy = 0.0
                    }
                    
                    echo "Best Accuracy: ${bestAccuracy}"
                    echo "Current Accuracy: ${CURRENT_ACCURACY}"
                    
                    if (CURRENT_ACCURACY.toDouble() > bestAccuracy) {
                        env.DEPLOY = "true"
                        echo "✓ New model is better! Will build and push Docker image."
                    } else {
                        env.DEPLOY = "false"
                        echo "✗ New model is not better. Skipping Docker build."
                    }
                }
            }
        }
        
        stage('Build Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                echo 'Building Docker image...'
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub-creds') {
                        def customImage = docker.build("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                    }
                }
            }
        }
        
        stage('Push Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                echo 'Pushing Docker image to Docker Hub...'
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub-creds') {
                        def customImage = docker.image("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                        customImage.push()
                        customImage.push('latest')
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo 'Archiving artifacts...'
            archiveArtifacts artifacts: 'app/artifacts/**', allowEmptyArchive: false
        }
        success {
            echo '✓ Pipeline completed successfully!'
        }
        failure {
            echo '✗ Pipeline failed!'
        }
    }
}
