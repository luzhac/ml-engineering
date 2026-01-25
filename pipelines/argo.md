# Install
kubectl create namespace argo

kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

kubectl get pods -n argo

kubectl -n argo edit deployment argo-server
args:
  - server
  - --auth-mode=server
  - --insecure

kubectl -n argo port-forward svc/argo-server 2746:2746


kubectl create -f hello-world.yaml -n argo

kubectl create -f etl-public-data.yaml -n argo

kubectl create -f iris-pipline.yaml -n argo


kubectl describe wf etl-public-data-trtmc  -n argo

kubectl create -f etl-public-data-cron.yaml

# Debug 

kubectl get wf -n argo




