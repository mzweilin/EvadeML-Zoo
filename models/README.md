# Pre-trained Models

*Note*: Make sure the last layer in Keras model definition is the Softmax activation layer, *i.e.* model.layers[-2].output is the logits and model.layers[-1].output is the softmax activation, because some attack algorithms require the logits output.

## Dataset: MNIST

| Model Name | # Trainable Parameters  | Testing Accuracy |  Mean Confidence |
|------------|-------------------------|------------------|------------------|
| Cleverhans |  710,218                |     0.9919       |  0.8897          |
| Carlini    |  312,202                |     0.9943       |  0.9939          |
| PGDbase    |  3,274,634                |     0.9917       |  0.9915          |
| PGDtrained    |  3,274,634                |     0.9853       |  0.9777          |

## Dataset: CIFAR-10

|      Model Name     |  # Trainable Parameters  | Testing Accuracy |  Mean Confidence |
|---------------------|--------------------------|------------------|------------------|
| Carlini             | 1,147,978                |     0.7796       |  0.7728          |
| DenseNet(L=40,k=12) | 1,019,722                |     0.9484       |  0.9215        |


## Dataset: ImageNet (ILSVRC)

| Model Name | # Trainable Parameters  | Top-1 Accuracy   |  Top-5 Accuracy |
|------------|-------------------------|------------------|-----------------|
| MobileNet  |  4,231,976              |     0.68360      |  0.88250        |
|Inception v3| 23,817,352              |  0.76276         |     0.93032     |


