# pretrained.ml
Sortable and searchable compilation of pre-trained deep learning models. With demos and code.

## About
Having spent too much time installing deep learning models just to evaluate their performance, I created this repo for several reasons:
 - Access a free demo of deep learning models
 - Get a docker container running the model for a quick install
 - Gather all the deep learning models available

## Installation

    git clone https://github.com/EliotAndres/pretrained.ml
    cd containers/tensorflow
    sudo docker build . -t keras
    docker run -dit -p 0.0.0.0:8081:8091 -t keras

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
- [ ] Add queuing system
- [ ] Linter
- [ ] Resize images for segmentation
