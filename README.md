# cppn-gan-vae tensorflow on CIFAR-10

Train [Compositional Pattern Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) as a Generative Model, using Generative Adversarial Networks and Variational Autoencoder techniques to produce high resolution images.

![CIFAR Truck](https://cdn.rawgit.com/hardmaru/cppn-gan-vae-cifar-tensorflow/master/examples/truck_8_reconstr_full.jpeg)
*CPPN Output after training on the `Truck` class of CIFAR-10*

`sampler.py` can be used inside IPython to interactively see results from the models being trained.

See my blog post at [blog.otoro.net](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/) for more details on training on the MNIST set.

This version is an experimental hacked version of the MNIST model to train on CIFAR-10.  Results are not really good yet, but I decided to just put the code up in case anyone wants to play with it and make it work.

I wrote another [blog post](http://blog.otoro.net/2016/04/06/the-frog-of-cifar-10/) about some of the current generative results on CIFAR-10, and what I think can improve this model going forward.

Tested this implementation on TensorFlow 0.60.

Used images2gif.py written by Almar Klein, Ant1, Marius van Voorden.

# License

BSD - images2gif.py

MIT - everything else
