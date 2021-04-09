docker rm -f mockserver
docker run --name mockserver -d -p 8040:80 mock-mast
