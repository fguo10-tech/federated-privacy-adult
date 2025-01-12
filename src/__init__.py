"""federated-privacy-adult — draft reference implementation.

Sub-packages
------------
``src.common``     seeding, configuration, logging, metrics, plotting, Before/After tables
``src.federated``  Flower-based weighted FedAvg, Opacus differential privacy, UCI Adult clients

The import package is called ``src``, following the folder layout. Use it from a checkout
installed in editable mode (``pip install -e ".[all]"``), one virtual environment per project.
"""

__version__ = "0.1.0.dev0"
