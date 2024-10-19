#!/usr/bin/env bash
# Dockerfile is in the root
cd ..

# start docker
# sudo service docker start

# list current docker packages
# docker container ls -a

# delete existing deepface packages
docker rm -f $(docker ps -a -q --filter "ancestor=deepface-elibra")

# build deepface image
docker build -f Dockerfile-elibra -t deepface-elibra .

# copy weights from your local
# docker cp ~/.deepface/weights/. <CONTAINER_ID>:/root/.deepface/weights/

# run the built image
# docker run --net="host" deepface
docker run -p 5005:5000 deepface-elibra

# or pull the pre-built image from docker hub and run it
# docker pull serengil/deepface
# docker run -p 5005:5000 serengil/deepface

# to access the inside of docker image when it is in running status
# docker exec -it <CONTAINER_ID> /bin/sh

# healthcheck
# sleep 3s
# curl localhost:5000
