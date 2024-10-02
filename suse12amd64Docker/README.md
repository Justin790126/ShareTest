# AMD64 container

build load amd64 image

````
docker buildx build --platform linux/amd64  -t m2amd64suse12 --load .
````

run container first time

````
docker run -it --platform linux/amd64 --name suse12x86 m2amd64suse12 bash
````


start container next time

````
docker start suse12x86
````

execute bash in container

````
docker exec -it suse12x86 bash
````
