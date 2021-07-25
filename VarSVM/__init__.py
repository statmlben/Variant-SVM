import pyximport
pyximport.install()
from .fastloop import CD, CD_drift, noneg_CD_drift
from varsvm.weightsvm import weightsvm
from varsvm.driftsvm import driftsvm
from varsvm.noneg_driftsvm import noneg_driftsvm

# __all__ = ['weightsvm', 'driftsvm', 'noneg_driftsvm']
