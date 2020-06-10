.. ADA documentation master file.

ADA: (Yet) Another Domain Adaptation library
============================================

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

You can find an introduction to ADA on `medium <https://medium.com/criteo-labs/introducing-ada-another-domain-adaptation-library-5df8b79378ee>`__.

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

The full list of implemented algorithm with references can be found in the :doc:`/algorithms` section.

.. include:: quickstart.rst

Checkout the :doc:`getting_started` page to get a more in-depth description of how you can use configuration files to run
most of your experiments.

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

Citing
------

If this library is useful for your research please cite:

::

        @misc{adalib2020,
        title={(Yet) Another Domain Adaptation library},
        author={Tousch, Anne-Marie and Renaudin, Christophe},
        url={https://github.com/criteo-research/pytorch-ada},
        year={2020}
        }



Browse the documentation
========================

.. toctree::
   :maxdepth: 2
   
   getting_started
   algorithms
   benchmarks
   known_issues
   source/modules


Browse the API
==============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
