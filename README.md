# AdvZoo

The goal of this project:
* Several datasets ready to use: MNIST, CIFAR-10, ImageNet-ILSVRC and more.
* Pre-trained really state-of-the-art models to attack.
* Existing attacking methods: FGSM, BIM, JSMA, Carlini/Wagner-L2/Li/L0 and more.
* Visualization of adversarial examples.
* Existing defense methods.

## 1. Install dependencies.

```bash
pip install tensorflow-gpu==1.1.0 keras==2.0.4 matplotlib h5py pillow scikit-learn click future
```

## 2. Usage of `python main.py`
```bash
usage: main.py [-h] [--dataset_name DATASET_NAME] [--nb_examples NB_EXAMPLES]
               [--test_mode [TEST_MODE]] [--notest_mode]
               [--model_name MODEL_NAME] [--attacks ATTACKS]
               [--visualize [VISUALIZE]] [--novisualize] [--defense DEFENSE]
               [--result_folder RESULT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Supported: MNIST, CIFAR-10, ImageNet.
  --nb_examples NB_EXAMPLES
                        The number of examples selected for attacks.
  --test_mode [TEST_MODE]
                        Only select one sample for each class.
  --notest_mode
  --model_name MODEL_NAME
                        Supported: carlini for MNIST and CIFAR-10; cleverhans
                        and cleverhans_adv_trained for MNIST; resnet50 for
                        ImageNet.
  --attacks ATTACKS     Attack name and parameters in URL style, separated by
                        semicolon.
  --visualize [VISUALIZE]
                        Output the image examples as image or not.
  --novisualize
  --defense DEFENSE     Supported: feature_squeezing (robustness&detection).
  --result_folder RESULT_FOLDER
                        The output folder for results.
```

### Example.
```bash
python main.py --dataset_name MNIST --model_name carlini --nb_examples 100 --attacks "FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=100&max_iterations=1000;CarliniL2?targeted=next&batch_size=100&max_iterations=1000&confidence=2;CarliniLi?targeted=next;CarliniL0?targeted=next;" --defense feature_squeezing
```