# Attack Algorithms



|     Method        |      Source       |          Default Parameters             |
|-------------------|-------------------|-----------------------------------------|
|      FGSM         | Cleverhans        |     eps=0.1                                    |
|      BIM          | Cleverhans        |     eps=0.1&eps_iter=0.05&nb_iter=10    |
|      JSMA         | Cleverhans        |     theta=1&gamma=0.1     |
| Carlini/Wagner L2 | nn_robust_attack  |    batch_size=1&confidence=0&learning_rate=0.01&binary_search_steps=9&max_iterations=10000&abort_early=true&initial_const=0.001 |
| Carlini/Wagner Li | nn_robust_attack  |  ? |
| Carlini/Wagner L0 | nn_robust_attack  |  ? |
| DeepFool          |  ?                  |? |

