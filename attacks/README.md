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
| largest_const         |   -    |    2e1    |   2e6  |
| reduce_const          |   -    |    false    |   false  |
| decrease_factor       |   -    |    0.9    |   -  |
| const_factor          |   -    |    2.0    |   2.0  |
| independent_channels  |   -    |    -    |   false  |

*Note*: C/W Li has an additional variable Tau to control the perturbations, thus the largest_const could be smaller to get successful adversarial examples while it significantly saves computation resource.

### DeepFool

* num_classes: limits the number of classes to test against.
* overshoot: used as a termination criterion to prevent vanishing updates 
* max_iter: maximum number of iterations for deepfool

Example: `python --attacks "deepfool?overshoot=9&max_iter=50"`

### Universal Adversarial Perturbations

* delta: controls the desired fooling rate (default = 80% fooling rate when delta == 0.2)
* max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
* xi: controls the l_p magnitude of the perturbation (default = 10)
* p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
* num_classes: num_classes (limits the number of classes to test against, by default = 10)
* overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
* max_iter_df: maximum number of iterations for deepfool (default = 10)

Example: `python --attacks "unipert?overshoot=9&max_iter_df=50"`

|       Parameter       |  DeepFool   |  Universal Adversarial Perturbations |
|-----------------------|-------------|--------------------------------------|
|    num_classes        |  10         |             10                       |
|    overshoot          |  0.02       |             0.02                     |
|    max_iter           |  50         |             -                        |
|    max_iter_df        |  -          |             10                       |
|    max_iter_uni       |  -          |             np.inf                   |
|    delta              |  -          |             0.2                      |
|    xi                 |  -          |             10                       |
|    p                  |  -          |             np.inf                   |


