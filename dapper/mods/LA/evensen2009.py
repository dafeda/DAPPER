"""A mix of `bib.evensen2009ensemble` and `bib.sakov2008implications`.

.. note::
    Since there is no noise, and the system is stable,
    the rmse's from this HMM go to zero as `T` goes to infinity.
    Thus, benchmarks largely depend on the initial error,
    and so these absolute rmse values are not so useful
    for quantatative evaluation of DA methods.
    For that purpose, see `dapper.mods.LA.raanes2015` instead.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.LA import Fmat, sinusoidal_sample
from dapper.mods.Lorenz96 import LPs

Nx = 1000
Ny = 4

# Measurements are made at 4 spatial points defined in `jj`.
# A new measurement is made every 5 seconds, set using `dto`.
# Since T = 300, the matrix of measurements has 300 / 5 = 60 rows and 4 columns.
jj = np.linspace(0, Nx, Ny, endpoint=False, dtype=int)

T = 300
dt = 1
tseq = modelling.Chronology2(dt=dt, T=T, dto=round(5 * dt))

# WITHOUT explicit matrix (assumes dt == dx/c):
# step = lambda x,t,dt: np.roll(x,1,axis=x.ndim-1)
# WITH:
Fm = Fmat(Nx, c=-1, dx=1, dt=tseq.dt)


def step(x, t, dt):
    assert dt == tseq.dt
    return x @ Fm.T


Dyn = modelling.Operator(M=Nx, model=step, linear=lambda x, t, dt: Fm, noise=0)

# In the animation, it can sometimes/somewhat occur
# that the truth is outside 3*sigma !!!
# Yet this is not so implausible because sinusoidal_sample()
# yields (multivariate) uniform (random numbers) -- not Gaussian.
wnum = 25
a = np.sqrt(5) / 10
X0 = modelling.RV(M=Nx, func=lambda N: a * sinusoidal_sample(Nx, wnum, N))

Obs = modelling.partial_Id_Obs(Nx, jj)
Obs["noise"] = 0.01
Obs = modelling.Operator(
    M=Obs.get("M"), model=Obs.get("model"), linear=Obs.get("linear"), noise=0.01
)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, liveplotters=LPs(jj))

####################
# Suggested tuning
####################
# xp = EnKF('PertObs',N=100,infl=1.02)
