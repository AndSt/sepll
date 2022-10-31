#  SepLL: Separating Latent Class Labels from Weak Supervision Noise

Authors: Andreas Stephan, Vasiliki Kougia, Benjamin Roth

This repo contains the code related to the paper [SepLL: Separating Latent Class Labels from Weak Supervision Noise](https://arxiv.org/abs/2204.13409).
<br />It was accepted as a Findings of EMNLP 2022.

If there's questions, please contact us [here](mailto:andreas.stephan@univie.ac.at)

---------

## Abstract

In the weakly supervised learning paradigm, \emph{labeling functions} automatically assign heuristic, often noisy, labels to data samples.
In this work, we provide a method for learning from weak labels by separating two types of complementary information associated with the labeling functions: 
information related to the target label and information specific to one labeling function only.
Both types of information are reflected to different degrees by all labeled instances.
In contrast to previous works that aimed at correcting or removing wrongly labeled instances, we learn a branched deep model that uses all data as is, but splits the labeling function information in the latent space.
Specifically, we propose the end-to-end model \emph{SepLL} which extends a transformer classifier by introducing a latent space for labeling function specific and task-specific information.
The learning signal is only given by the labeling functions matches, no pre-processing or label model is required for our method. 
Notably, the task prediction is made from the latent layer without any direct task signal.
Experiments on Wrench text classification tasks show that our model is competitive with the state-of-the-art, and yields a new best average performance. 

-----

## Installation

The code is written in JAX, so the first step is to install [JAX](https://github.com/google/jax). 
Note that there is different versions for CPU / GPU, depending on your environment.
Secondly, run ```pip install requirements_full.txt``` to mirror our setup.
In case you want to work with newer version of the used libraries, run ```pip install requirements.txt```. 
We do not continuously test whether this works.

Finally you should also run ```pip install -e .``` to make sure imports work.

-----

## Data 

All the datasets of our experiments are integrated in the [WRENCH](https://github.com/JieyuZ2/wrench) benchmark. 
We make use of the [Knodle](github.com/knodle/knodle) format to represent labeling function matches as this suits our training better.
In sepll/data/wrench.py you can find code to transform the WRENCH format to the Knodle format.
Alternatively you can download the transformed data here: [https://knodle.cc/minio/knodle/datasets/wrench_knodle_format/](https://knodle.cc/minio/knodle/datasets/wrench_knodle_format/)

-----

## Trained Models

If you want to have access to the best performing models, feel free to contact us under <andreas.stephan@univie.ac.at>.

-----

### Usage

In ```scripts/config.cfg``` you find a configuration for the dataset YouTube. 
You need to set three data paths, namely ```data_dir``` (directory holding all datasets from the link above),
```work_dir``` where results etc. are being written and the ".

You can then run a training (including evaluation) with ```python sepll/train.py --flagfile=scripts/config.cfg```

Advertisement: We develop a wrapper to work locally (on Jupyter) and on SLURM more or less automatically.
 Get in touch with us if you are interested to join efforts.
Look [here](https://github.com/AndSt/slurm_utils_test_repo) if you are interested.

-----

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2204.13409,
  doi = {10.48550/ARXIV.2204.13409},
  url = {https://arxiv.org/abs/2204.13409},
  author = {Stephan, Andreas and Roth, Benjamin},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {WeaNF: Weak Supervision with Normalizing Flows},  
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Acknowledgements

This research was funded by the WWTF through the project ”Knowledge-infused Deep Learning for Natural Language Processing” (WWTF Vienna Re- search Group VRG19-008).
