# Attack Algorithms

## Default Parameters

### Cleverhans Attacks
 
* eps: (required float) maximum distortion of adversarial example compared to original input
* eps_iter: (required float) step size for each attack iteration
* nb_iter: (required int) Number of attack iterations.
* theta: (optional float) Perturbation introduced to modified components (can be positive or negative)
* gamma: (optional float) Maximum percentage of perturbed features

|  Parameter |  FGSM   |  BIM   | JSMA |
|------------|---------|--------|------|
|    eps     |  0.1    |   0.1  |- |
|  eps_iter  |   -   |   0.05 |  - |
|  nb_iter   |   -   |  10    |  - |
|   theta    |   -   |   -  | 1    |
|   gamma    |   -   |   -  | 0.1  |


### C/W attacks
|       Parameter       | C/W L2   |  C/W Li   | C/W L0 |
|-----------------------|----------|-----------|--------|
| batch_size            |   1    |    -    |   -  |
| confidence            |  0    |    -    |   -  |
| learning_rate         |  0.01    |    0.005    |   0.01  |
| binary_search_steps   |   9    |    -    |   -  |
| max_iterations        |   10000    |    1000    |   1000  |
| abort_early           |   true    |    true    |   true|
| initial_const         |   0.001    |    0.00001    |   0.001  |
| largest_const         |   -    |    2e+1    |   2e+6  |
| reduce_const          |   -    |    false    |   false  |
| decrease_factor       |   -    |    0.9    |   -  |
| const_factor          |   -    |    2.0    |   2.0  |
| independent_channels  |   -    |    -    |   false  |


### DeepFool

* num_classes: limits the number of classes to test against.
* overshoot: used as a termination criterion to prevent vanishing updates 
* max_iter: maximum number of iterations for deepfool

|       Parameter       |  DeepFool   |  Universal Adversarial Perturbations |
|-----------------------|-------------|--------------------------------------|
|    num_classes        |  10         |             ?                        |
|    overshoot          | 0.02        |             ?                        |
|    max_iter           |  50         |             ?                        |



|     Method        |      Source       |          Default Parameters             |
|-------------------|-------------------|-----------------------------------------|
|      FGSM         | Cleverhans        |     eps=0.1                                    |
|      BIM          | Cleverhans        |     eps=0.1&eps_iter=0.05&nb_iter=10    |
|      JSMA         | Cleverhans        |     theta=1&gamma=0.1     |
| CarliniL2 | nn_robust_attack  |    batch_size=1&confidence=0&learning_rate=0.01& binary_search_steps=9&max_iterations=10000& abort_early=true&initial_const=0.001 |
| CarliniLi | nn_robust_attack  |    learning_rate=5e-3&max_iterations=1000&abort_early=true&initial_const=1e-5&largest_const=2e+1&reduce_const=false&decrease_factor=0.9&const_factor=2.0 |
| CarliniL0 | nn_robust_attack  |    learning_rate=1e-2&max_iterations=1000&abort_early=true&initial_const=1e-3&largest_const=2e6&reduce_const=false&const_factor=2.0&independent_channels=false |
| DeepFool          |  Universal          |? |

