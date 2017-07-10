# Attack Algorithms

## Default Parameters of Attacks from Cleverhans
|  Parameter |  FGSM   |  BIM   | JSMA |
|------------|---------|--------|------|
|    eps     |  0.1    |   0.1  |- |
|  eps_iter  |   -   |   0.05 |  - |
|  nb_iter   |   -   |  10    |  - |
|   theta    |   -   |   -  | 1    |
|   gamma    |   -   |   -  | 0.1  |


## Default Parameters of Attacks from C/W's nn_robust_attacks
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

|     Method        |      Source       |          Default Parameters             |
|-------------------|-------------------|-----------------------------------------|
|      FGSM         | Cleverhans        |     eps=0.1                                    |
|      BIM          | Cleverhans        |     eps=0.1&eps_iter=0.05&nb_iter=10    |
|      JSMA         | Cleverhans        |     theta=1&gamma=0.1     |
| CarliniL2 | nn_robust_attack  |    batch_size=1&confidence=0&learning_rate=0.01& binary_search_steps=9&max_iterations=10000& abort_early=true&initial_const=0.001 |
| CarliniLi | nn_robust_attack  |    learning_rate=5e-3&max_iterations=1000&abort_early=true&initial_const=1e-5&largest_const=2e+1&reduce_const=false&decrease_factor=0.9&const_factor=2.0 |
| CarliniL0 | nn_robust_attack  |    learning_rate=1e-2&max_iterations=1000&abort_early=true&initial_const=1e-3&largest_const=2e6&reduce_const=false&const_factor=2.0&independent_channels=false |
| DeepFool          |  Universal          |? |

