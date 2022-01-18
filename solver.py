from settings import *
from tools import *


# layout of the minimization parameter vector:
# <primarySPDs->wavelengths(nm)

def extractSPDS(a, numwaves):
    sds = np.asarray(a)[:3 * numwaves].reshape((3, numwaves))
    return sds
    
def extractCMFS(a, numwaves):
    cmfs = np.array([[]])
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    for idx, wave in enumerate(waves):
        if idx == 0:
            cmfs = np.append(cmfs, [CMFS[wave]], axis=1)
        else:
            cmfs = np.append(cmfs, [CMFS[wave]], axis=0)
    return cmfs

def extractIlluminantSPD(a, numwaves):
    illuminantSPD = np.array([])
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    for wave in waves:
        illuminantSPD = np.append(illuminantSPD, illuminant_SPD[wave])
    return illuminantSPD

def extractIlluminantModifier(a, numwaves):
    illuminantModifier = np.asarray(a)[4 * numwaves:5 * numwaves]
    return illuminantModifier

