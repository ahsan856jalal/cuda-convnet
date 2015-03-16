**Note July 18, 2014**:
  * I've released an update to `cuda-convnet`, called [cuda-convnet2](https://code.google.com/p/cuda-convnet2/). The two main new features are faster training on Kepler-generation GPUs and support for multi-GPU training.

This is a fast C++/CUDA implementation of convolutional (or more generally, feed-forward) neural networks. It can model arbitrary layer connectivity and network depth. Any directed acyclic graph of layers will do. Training is done using the back-propagation algorithm.

Fermi-generation GPU (GTX 4xx, GTX 5xx, or Tesla equivalent) required.

### Documentation ###
  * [Compiling](Compiling.md) -- how to check out and compile this code.
  * [Data](Data.md) -- what kind of data this net can train on.
  * [LayerParams](LayerParams.md) -- how to specify an architecture for the net.
  * [NeuronTypes](NeuronTypes.md) -- types of hidden unit nonlinearities.
  * [TrainingNet](TrainingNet.md) -- how to train the net.
  * [Options](Options.md) -- the command-line arguments that the net takes.
  * [ViewingNet](ViewingNet.md) -- how to look inside the checkpoints saved by the net.
  * [CheckingGradients](CheckingGradients.md) -- how to numerically test the gradients for correctness.

### Fast results ###
  * **11%** error on [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) in **75 minutes**, with image translations and horizontal reflections ([def](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-conv-local-11pct.cfg), [params](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-conv-local-11pct.cfg)).
  * **13%** error on CIFAR-10 in **25 minutes**, with image translations and horizontal reflections ([def](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-conv-local-13pct.cfg), [params](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-conv-local-13pct.cfg)).
    * See [Methodology](Methodology.md) for details of training.
> Filters learned by this net:
> > ![http://cuda-convnet.googlecode.com/svn/wiki/images/13pct-filters.png](http://cuda-convnet.googlecode.com/svn/wiki/images/13pct-filters.png)
  * **18%** error on CIFAR-10 in **20 minutes**, without any image translations/transformations/preprocessing ([def](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg), [params](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-18pct.cfg)).
  * **26%** error on CIFAR-10 in **80 seconds**, without any image translations/transformations/preprocessing ([def](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg), [params](http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-80sec.cfg)).


---


## Recent changes ##
  * **Jul 17, 2012**
    * Fixed bug in contrast normalization backpropagation code which caused wrong gradients to be computed near image borders. (Thanks Hannes Schulz).
  * **Mar 13, 2012**
    * Added [response-normalization across maps](LayerParams#Local_response_normalization_layer_(across_maps).md) layer.
    * Started modifying the code to support rectangular (i.e. non-square) images. The convolution code now supports rectangular images, but the remaining code does not yet. So the whole package still requires square images.
  * **Feb 8, 2012**
    * Most layer types now should work well with minibatch size 64 or 32.
    * Fixed --conserve-mem option so it can be combined with -f (i.e. its value can be changed after a net has been saved).
  * See [ChangeLog](ChangeLog.md) for older changes.

## Features ##
### Supported [layer types](LayerParams.md): ###
  * Layers with weights:
    * Fully-connected
    * Convolutional, including sparsely-connected convolutional
    * Locally-connected, unshared
  * Others:
    * Local pooling (avg, max), including overlapping pooling regions
    * Local response normalization
    * Local contrast normalization
    * Softmax
    * Elementwise sum, elementwise max
    * Gaussian blur + "bed of nails" subsampling
    * Resize with bilinear filtering

### Supported [neuron activation functions](NeuronTypes.md): ###
  * Logistic
  * Hyperbolic tangent
  * Rectified linear
  * [Others](NeuronTypes.md)

### Supported [objectives](LayerParams#Logistic_regression_cost_layer.md): ###
  * Logistic regression
  * Sum-of-squares

### Other features: ###
  * Efficient implementation of convolution in CUDA.
    * Supports arbitrary stride size at zero loss of efficiency (except that which comes from reducing the problem size).
    * Implicitly pads your images with an arbitrary-sized border of zeros without using any extra memory.
    * Supports block-random sparse connectivity at no performance cost (see [LayerParams](LayerParams#Random_sparse_convolution_layer.md)).
  * Modular design makes it easy to add new layer types, neuron activation functions, or objectives if you should need them.
  * Mostly avoids use of temporary memory where it isn't strictly needed.
  * Optimizes multiple objectives simultaneously.
  * Saves checkpoints to disk as python [pickled objects](http://docs.python.org/library/pickle.html), so you can write [python scripts](ViewingNet.md) to probe the mind of your neural net.
  * Capable of training on one batch of data while simultaneously loading the next from disk (or pre-processing it in some way, if necessary).
  * Numerically tests gradient computations for correctness.

---

## Contact ##
  * My university [web page](http://www.cs.toronto.edu/~kriz/)
  * My [email](mailto:akrizhevsky@gmail.com)

## Acknowledgements ##
  * I am grateful to Ilya Sutskever for suggestions that led to many of the features in this package.