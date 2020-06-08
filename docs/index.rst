.. ADA documentation master file.

ADA: (Yet) Another Domain Adaptation library
============================================

.. toctree::
   :hidden:
   
   getting_started
   benchmarks
   known_issues

Context
-------

The aim of ADA is to help researchers build new methods for unsupervised
and semi-supervised domain adaptation. The library is built on top of
`PyTorch-Lightning <https://pytorch-lightning.readthedocs.io/en/latest/new-project.html>`__,
enabling fast development of new models.

We built ADA with the idea of:

-  minimizing the boilerplate when developing a new method (loading data
   from several domains, logging errors, switching from CPU to GPU).
-  allowing fair comparison between methods by running all of them
   within the exact same environment.

Quick description
-----------------

Methods from the main 3 groups of methods are available for unsupervised
domain adaptation:

-  Adversarial methods: Domain-adversarial neural networks
   (`DANN <https://arxiv.org/abs/1505.07818>`__) and Conditional
   Adversarial Domain Adaptation networks
   (`CDAN <https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf>`__),
-  Optimal-Transport-based methods: Wasserstein distance guided
   representation learning
   (`WDGRL <https://arxiv.org/pdf/1707.01217.pdf>`__), for which we
   propose two implementations, the second one being a variant better
   adapted to the PyTorch-Lightning, allowing for multi-GPU training.
-  MMD-based methods: Deep Adaptation Networks
   (`DAN <http://proceedings.mlr.press/v37/long15.pdf>`__) and Joint
   Adaptation Networks (`JAN <https://arxiv.org/pdf/1605.06636.pdf>`__)

All these methods are implemented in 
:mod:`ada.models.architectures`.

Adversarial and OT-based methods both rely on 3 networks:

-  a feature extractor network mapping inputs :math:`x\in\mathcal{X}` to
   a latent space :math:`\mathcal{Z}`,
-  a task classifier network that learns to predict labels
   :math:`y \in \mathcal{Y}` from latent vectors,
-  a domain classifier network that tries to predict whether samples in :math:`\mathcal{Z}`
   come from the source or target domain.

MMD-based methods don't make use of the critic network.

Quick start
-----------

First you need to install the library. It has been tested with python
3.6+, with the latest versions of pytorch-lightning.

If you want to create a new conda environment, run:

::

        conda env create -n adaenv python=3.7
        conda activate adaenv

Install the library (with developer mode if you want to develop your own
models later on, otherwise you can skip the ``-e``):

::

        pip install -e adalib

*Note*: on **Windows**, it could be necessary to first install pytorch
and torchvision with conda:

::

        conda install -c pytorch pytorch
        conda install -c pytorch torchvision
        pip install -e adalib

Run on of the scripts:

::

        cd scripts
        python run_simple.py

By default, this script launches experiments with all kinds of methods
on a blobs dataset -- it doesn't take any parameter, you can change it
easily from the script. It may take a few minutes to finish.

Most parameters are available and can be changed through configuration
files, which are all grouped in the ``configs`` folder: - datasets -
network layers and training parameters - methods (Source, DANN, CDAN…),
and their specific parameters

Checkout the :doc:`getting_started` page to get a more in-depth description of how you can use configuration files to run
most of your experiments.

Advanced options
----------------

The script ``run_full_options.py`` runs the same kind of experiments
allowing for more variants (semi-supervised, unbalanced, with gpus and
MLFlow logging). You can run it without parameters or with ``-h`` to get
help.

MLFlow
~~~~~~

You can log results to MLFlow. Start a MLFlow server in another
terminal:

::

        conda activate adaenv
        mlflow ui --port=31014

Streamlit application
~~~~~~~~~~~~~~~~~~~~~

Optionally, you can use the ``streamlit`` app. First install
``streamlit`` with ``pip install streamlit``, then launch the app like
this:

::

        streamlit run run_toys_app.py

This will start a web app with a default port = 8501.

Benchmarks results
------------------

MNIST -> MNIST-M (5 runs)
~~~~~~~~~~~~~~~~~~~~~~~~~

+----------+-----------------+-----------------+
| Method   | source acc      | target acc      |
+==========+=================+=================+
| Source   | 89.0% +- 2.52   | 34.0% +- 1.71   |
+----------+-----------------+-----------------+
| DANN     | 94.2% +- 1.57   | 37.5% +- 2.85   |
+----------+-----------------+-----------------+
| CDAN     | 98.7% +- 0.19   | 68.4% +- 1.80   |
+----------+-----------------+-----------------+
| CDAN-E   | 98.7% +- 0.12   | 69.6% +- 1.51   |
+----------+-----------------+-----------------+
| DAN      | 98.0% +- 0.68   | 47.0% +- 1.85   |
+----------+-----------------+-----------------+
| JAN      | 96.4% +- 4.57   | 52.9% +- 2.16   |
+----------+-----------------+-----------------+
| WDGRL    | 93.9% +- 2.70   | 52.0% +- 4.82   |
+----------+-----------------+-----------------+

MNIST -> USPS (5 runs)
~~~~~~~~~~~~~~~~~~~~~~

+----------+-----------------+-----------------+
| Method   | source acc      | target acc      |
+==========+=================+=================+
| Source   | 99.2% +- 0.08   | 94.2% +- 1.07   |
+----------+-----------------+-----------------+
| DANN     | 99.1% +- 0.15   | 93.8% +- 1.06   |
+----------+-----------------+-----------------+
| CDAN     | 98.8% +- 0.17   | 90.7% +- 1.17   |
+----------+-----------------+-----------------+
| CDAN-E   | 98.9% +- 0.11   | 90.3% +- 0.98   |
+----------+-----------------+-----------------+
| DAN      | 99.0% +- 0.14   | 95.0% +- 0.83   |
+----------+-----------------+-----------------+
| JAN      | 98.6% +- 0.30   | 89.5% +- 2.00   |
+----------+-----------------+-----------------+
| WDGRL    | 98.7% +- 0.13   | 85.7% +- 6.57   |
+----------+-----------------+-----------------+

See :doc:`/benchmarks` for more complete benchmarks.

Contributing
------------

Code
~~~~

You can find the latest version on github. Before submitting code,
please run ``black`` to have clean code formatting:

::

        pip install black
        black .

Documentation
~~~~~~~~~~~~~

First ``pip`` install ``sphinx``, ``sphinx-paramlinks``,
``recommonmark``. Generate the documentation:

::

        cd docs
        sphinx-apidoc -o source/ ../adalib/ada ../scripts/
        make html

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
