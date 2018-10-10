# [DEPRECATED] pretrained.ml

[DEPRECATED] Sortable and searchable compilation of pre-trained deep learning models. With demos and code.

**DEPRECATED**: You can find an alternative on [modeldepot.io (with live demos)](https://modeldepot.io) or on [modelzoo.co](http://modelzoo.co)

## A word of warning
This is running on a server without GPU, hence it seems slow.

Also, the code may look a bit like monkey-patching for the following reasons:
 - Models are cloned as submodules: therefore we have to mess around with the python path :-(
 - There is a queuing systems for the jobs (allows user to see their job's position in the queue)

## About
Having spent too much time installing deep learning models just to evaluate their performance, I created this repo for several reasons:
 - Access a free demo of deep learning models
 - Gather available deep learning models
 - Get a docker container running the model for a quick install

## Installation
Requirements: Docker, docker-compose and enough space free for the model weights.

    git clone https://github.com/EliotAndres/pretrained.ml --recursive
    cd containers
    docker-compose build
    docker-compose up -d

## Useful commands
    docker ps #list images
    docker attach [container_id] #attach a shell to specific image

## Contributing
Many models are missing. Any help is welcome ! You have two options to contribute.

Easy way: Add a model to the list without a demo:
 - Fork the repo
 - Edit the **docs/models.yaml** file (you can even edit it with Github's editor)
 - Make a pull request

Other way: add a model with a demo:
 - Fork the repo
 - Add the model inside one of the docker containers
 - Create a route in the serve.py file
 - Add a demo calling the route
 - Make a pull request


## Todos
- [ ] Use nvidia-docker ?
- [ ] Add flag to compile Tensorflow
- [ ] Consider splitting each model in a different container ?
- [ ] Linter
- [ ] Add analytics
