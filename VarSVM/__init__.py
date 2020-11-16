import pyximport
pyximport.install()
from .fastloop import CD, CD_drift, noneg_CD_drift
from VarSVM.weightsvm import weightsvm
from VarSVM.driftsvm import driftsvm
from VarSVM.noneg_driftsvm import noneg_driftsvm

# __all__ = ['weightsvm', 'driftsvm', 'noneg_driftsvm']
