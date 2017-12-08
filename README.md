# EvadeML-Zoo

The goal of this project:
* Several datasets ready to use: MNIST, CIFAR-10, ImageNet-ILSVRC and more.
* Pre-trained state-of-the-art models to attack. [[See details]](models/README.md).
* Existing attacking methods: FGSM, BIM, JSMA, Deepfool, Universal Perturbations, Carlini/Wagner-L2/Li/L0 and more. [[See details]](attacks/README.md).
* Visualization of adversarial examples.
* Existing defense methods as baseline.

The code was developed on Python 2, but should be runnable on Python 3 with tiny modifications.

Please follow [the instructions](Reproduce_FeatureSqueezing.md) if you would like to reproduce the results on our Feature Squeezing paper.


## 1. Install dependencies.

```bash
pip install -r requirements_cpu.txt
```

If you are going to run the code on GPU, install this list instead:
```bash
pip install -r requirements_gpu.txt
```

## 2. Fetch submodules.
```bash
git submodule update --init --recursive
```

## 3. Download pre-trained models.
```bash
mkdir downloads; curl -sL https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/downloads.tar.gz | tar xzv -C downloads
```

## 4. Usage of `python main.py`
```
usage: python main.py [-h] [--dataset_name DATASET_NAME] [--model_name MODEL_NAME]
               [--select [SELECT]] [--noselect] [--nb_examples NB_EXAMPLES]
               [--balance_sampling [BALANCE_SAMPLING]] [--nobalance_sampling]
               [--test_mode [TEST_MODE]] [--notest_mode] [--attacks ATTACKS]
               [--clip CLIP] [--visualize [VISUALIZE]] [--novisualize]
               [--robustness ROBUSTNESS] [--detection DETECTION]
               [--detection_train_test_mode [DETECTION_TRAIN_TEST_MODE]]
               [--nodetection_train_test_mode] [--result_folder RESULT_FOLDER]
               [--verbose [VERBOSE]] [--noverbose]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Supported: MNIST, CIFAR-10, ImageNet.
  --model_name MODEL_NAME
                        Supported: cleverhans, cleverhans_adv_trained and
                        carlini for MNIST; carlini and DenseNet for CIFAR-10;
                        ResNet50, VGG19, Inceptionv3 and MobileNet for
                        ImageNet.
  --select [SELECT]     Select correctly classified examples for the
                        experiement.
  --noselect
  --nb_examples NB_EXAMPLES
                        The number of examples selected for attacks.
  --balance_sampling [BALANCE_SAMPLING]
                        Select the same number of examples for each class.
  --nobalance_sampling
  --test_mode [TEST_MODE]
                        Only select one sample for each class.
  --notest_mode
  --attacks ATTACKS     Attack name and parameters in URL style, separated by
                        semicolon.
  --clip CLIP           L-infinity clip on the adversarial perturbations.
  --visualize [VISUALIZE]
                        Output the image examples for each attack, enabled by
                        default.
  --novisualize
  --robustness ROBUSTNESS
                        Supported: FeatureSqueezing.
  --detection DETECTION
                        Supported: feature_squeezing.
  --detection_train_test_mode [DETECTION_TRAIN_TEST_MODE]
                        Split into train/test datasets.
  --nodetection_train_test_mode
  --result_folder RESULT_FOLDER
                        The output folder for results.
  --verbose [VERBOSE]   Stdout level. The hidden content will be saved to log
                        files anyway.
  --noverbose
```

### 5. Example.
```bash
python main.py --dataset_name MNIST --model_name carlini \
--nb_examples 2000 --balance_sampling \
--attacks "FGSM?eps=0.1;" \
--robustness "none;FeatureSqueezing?squeezer=bit_depth_1;" \
--detection "FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr=0.05;"
```
