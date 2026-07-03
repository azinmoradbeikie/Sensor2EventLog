Installation
============

Requirements
------------

Sensor2EventLog targets Python 3.10+ and depends on:

.. code-block:: text

   numpy>=1.21.0
   pandas>=1.3.0
   scipy>=1.7.0
   scikit-learn>=0.24.0
   hmmlearn>=0.2.8
   matplotlib>=3.4.0
   seaborn>=0.11.0

From source
-----------

.. code-block:: bash

   git clone https://github.com/azinmoradbeikie/Sensor2EventLog.git
   cd Sensor2EventLog
   pip install -r requirements.txt

From PyPI
---------

.. code-block:: bash

   pip install sensor2eventlog

Verifying the install
----------------------

Run the toy walkthrough to confirm everything is wired up correctly:

.. code-block:: bash

   python3 tutorial/toy_walkthrough.py

If you plan to run the accompanying notebook, also install ``ipykernel``:

.. code-block:: bash

   python -m pip install ipykernel
