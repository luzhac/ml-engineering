docker build --no-cache -t ai-inference .

docker run -p 8000:8000 ai-inference

kubectl get service -n ai-inference
kube

kubectl port-forward svc/quant-inference 8000:80 -n ai-inference
