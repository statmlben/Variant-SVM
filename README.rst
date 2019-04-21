Variant-SVMs
============

VarSVM is a Python module for solving variants Support Vector Machines (SVM).

This project was started by `Ben Dai <http://users.stat.umn.edu/~bdai/>`_. If there is any problem and suggestion please contact me via <bdai@umn.edu>.

Installation
------------

Dependencies
~~~~~~~~~~~~

Variant-SVMs requires:

- Python (>= 3.5)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)


User installation
~~~~~~~~~~~~~~~~~

Install Variant-SVMs using ``pip`` ::

	pip install git+https://github.com/statmlben/VarSVM.git

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/statmlben/VarSVM.git


Documentation
------------

The mathematical formulation for each model can be found in `VariantSVMs <./Variant-SVMs.pdf>`_.

Weighted SVM
~~~~~~~~~~~~

- class VarSVM.weightsvm(alpha=[], beta=[], C=1., max_iter = 1000, eps = 1e-4, print_step = 1)
	- **alpha**: Dual variable.
	- **beta**: Primal variable, or coefficients of the support vector in the decision function.
	- **C**: Penalty parameter C of the error term.
	- **max_iter**: Hard limit on iterations for coordinate descent.
	- **eps**: Tolerance for stopping criterion based on the relative l1 norm for difference of beta and beta_old.
	- **print_step**: If print the interations for coordinate descent, 1 indicates YES, 0 indicates NO.







