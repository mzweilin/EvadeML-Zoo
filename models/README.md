# Pre-trained Models

*Note*: Make sure the last layer in Keras model definition is the Softmax activation layer, *i.e.* model.layers[-2].output is the logits and model.layers[-1].output is the softmax activation, because some attack algorithms require the logits output.

## Dataset: MNIST

| Model Name | Architecture  | Testing Accuracy |  Mean Confidence |
|------------|---------------|------------------|------------------|
| Cleverhans |               |     0.9919       |  0.8897          |
| Carlini    |               |     0.9943       |  0.9939          |

## Dataset: CIFAR-10

|      Model Name     | Architecture  |  Parameters      | Testing Accuracy |  Mean Confidence |
|---------------------|---------------|------------------|------------------|------------------|
| Carlini             |               |                  |     0.7796       |  0.7728          |
| DenseNet            |               | (L=40,k=12)      |     0.9484         |  0.9215        |


## Dataset: ImageNet (ILSVRC)

| Model Name | Architecture  | Top-1 Accuracy   |  Top-5 Accuracy |
|------------|---------------|------------------|-----------------|
| MobileNet  |               |     0.68360      |  0.88250        |
|Inception v3|               |  0.76276         |     0.93032     |


