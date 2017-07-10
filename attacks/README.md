# Attack Algorithms



|     Method        |      Source       |          Default Parameters             |
|-------------------|-------------------|-----------------------------------------|
|      FGSM         | Cleverhans        |     eps=0.1                                    |
|      BIM          | Cleverhans        |     eps=0.1&eps_iter=0.05&nb_iter=10    |
|      JSMA         | Cleverhans        |     theta=1&gamma=0.1     |
| CarliniL2 | nn_robust_attack  |    batch_size=1&confidence=0&learning_rate=0.01& binary_search_steps=9&max_iterations=10000& abort_early=true&initial_const=0.001 |
| CarliniLi | nn_robust_attack  |    learning_rate=5e-3&max_iterations=1000&abort_early=true& initial_const=1e-5&largest_const=2e+1 &reduce_const=false&decrease_factor=0.9&const_factor=2.0 |
| CarliniL0 | nn_robust_attack  |  ? |
| DeepFool          |  Universal          |? |

