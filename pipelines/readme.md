

# Install kubenetes
## Context env
kubectl config get-contexts

kubectl config use-context docker-desktop

## Install using local kubetenes

choco install k3d -y

k3d version

k3d cluster create dev \
  --servers 1 \
  --agents 2 \
  --volume "/mnt/c/Myfiles/ETL/data:/mnt/etl-data@all"






kubectl create ns argo

kubectl create serviceaccount argo-workflow -n argo

cd /mnt/c/myfiles/myprojects/crypto/ai-platform/pipelines/workflows

kubectl apply -f argo-workflow-clusterrole.yaml

kubectl create clusterrolebinding argo-workflow-binding \
  --clusterrole=argo-workflow \
  --serviceaccount=argo:argo-workflow




k3d cluster delete dev

 k3d cluster list

## get image using ecr 
Windows
 └─ WSL
     ├─ docker login ECR
     ├─ kubectl create secret ecr-secret
     └─ k3d cluster
          └─ Argo Workflow Pod
               └─ pulls image from ECR

aws sts get-caller-identity


aws ecr get-login-password \
  --region ap-northeast-1 \
| docker login \
  --username AWS \
  --password-stdin your-id.dkr.ecr.ap-northeast-1.amazonaws.com


 kubectl create secret docker-registry ecr-secret \
  --docker-server=your-id.dkr.ecr.ap-northeast-1.amazonaws.com \
  --docker-username=AWS \
  --docker-password="$(aws ecr get-login-password --region ap-northeast-1)" \
  --namespace argo

## install Argo
kubectl get secret ecr-secret -n argo

spec:
  serviceAccountName: argo-workflow

kubectl patch serviceaccount argo-workflow \
  -n argo \
  -p '{"imagePullSecrets":[{"name":"ecr-secret"}]}'

kubectl get sa argo-workflow -n argo -o yaml

