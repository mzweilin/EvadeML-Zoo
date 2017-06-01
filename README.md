# AdvZoo

### 1. Install dependencies.

```bash
pip install tensorflow==1.0.0 keras==2.0.4 matplotlib h5py pillow scikit-learn click
```
### 2. Run the experiment.
```bash
python main.py "FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next;"
```
