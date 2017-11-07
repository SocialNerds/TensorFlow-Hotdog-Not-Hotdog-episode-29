# Hot dog - Not Hot Dog application with TensorFlow

Did you ever want to have your own Hot Dog - Not Hot Dog application?
You can now! See the live video [here](https://youtu.be/7ki0mY0eQGU) to
learn how to do it in TensorFlow.

# Installation
Requirements
- You need to have [Docker](https://docs.docker.com/engine/installation/) installed

Steps

- Go to data/scripts/images/car and add 100-200 photos of cars 
- Go to data/scripts/images/hd and add 100-200 photos of Hot Dogs
- Go to data/scripts/images/test and add a photo of a car or a Hot Dog, named test.jpg

Run in root folder,
~~~~
docker-compose build && docker-compose up -d
~~~~

Login to the container,
~~~~
docker exec -it tensorflow /bin/bash -c "TERM=$TERM exec bash"
~~~~

Go to /scripts folder and run
~~~~
python tf.py
~~~~

# By SocialNerds
* [SocialNerds.gr](https://www.socialnerds.gr/)
* [YouTube](https://www.youtube.com/SocialNerdsGR)
* [Facebook](https://www.facebook.com/SocialNerdsGR)
* [Twitter](https://twitter.com/socialnerdsgr)