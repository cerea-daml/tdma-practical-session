# Introduction to surrogate modelling in the geosciences

#### Marc Bocquet¹ [marc.bocquet@enpc.fr](mailto:marc.bocquet@enpc.fr) and Alban Farchi¹ [alban.farchi@enpc.fr](mailto:alban.farchi@enpc.fr)
##### (1) CEREA, École des Ponts and EdF R&D, IPSL, Île-de-France, France

[![DOI](https://zenodo.org/badge/639316033.svg)](https://zenodo.org/doi/10.5281/zenodo.10479131)

During this session, we will apply standard machine learning methods to learn the dynamics of the Lorenz 1996 model. 
The objective here is to get a preview of how machine learning can be applied to geoscientific models in a low-order models where testing is quick.

These practical sessions are part of the 
[TDMA summer school](https://tdma2023.sciencesconf.org) 
held in 2023 in Grenoble, France.

## Installation

Install conda, for example through [miniconda](https://docs.conda.io/en/latest/miniconda.html) or through [mamba](https://mamba.readthedocs.io/en/latest/installation.html).

Clone the repertory:

    $ git clone git@github.com:cerea-daml/tdma-practical-session.git

Go to the repertory. Once there, create a dedicated anaconda environment for the sessions:

    $ conda env create -f environment.yaml

Activate the newly created environment:

    $ conda activate tdma

Open the notebook (e.g. with Jupyter) and follow the instructions:

    $ jupyter-notebook questions.ipynb
