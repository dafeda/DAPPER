"""Reproduce results from `bib.sakov2008deterministic`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.QG import LP_setup, model_config, sample_filename, shape
from dapper.tools.localization import nd_Id_localization

############################
# Time series, model, initial condition
############################

model = model_config("sakov2008", {})
Dyn = modelling.Operator(M=np.prod(shape), model=model.step, noise=0)

# Considering that I have 8GB mem on the Mac, and the estimate:
# ≈ (8 bytes/float)*(129² float/stat)*(7 stat/k) * K,
# it should be possible to run experiments of length (K) < 8000.
t = modelling.Chronology(dt=model.prms["dtout"], dko=1, T=1500, BurnIn=250)
# In my opinion the burn in should be 400.
# Sakov also used 10 repetitions.

X0 = modelling.RV(M=Dyn.M, file=sample_filename)


############################
# Observation settings
############################

# This will look like satellite tracks when plotted in 2D
Ny = 300
jj = np.linspace(0, Dyn.M, Ny, endpoint=False, dtype=int)

# Want: random_offset(t1)==random_offset(t2) if t1==t2.
# Solutions: (1) use caching (ensure maxsize=inf) or (2) stream seeding.
# Either way, use a local random stream to avoid interfering with global stream
# (and e.g. ensure equal outcomes for 1st and 2nd run of the python session).
rstream = np.random.RandomState()
max_offset = jj[1] - jj[0]


def random_offset(t):
    rstream.seed(int(t / model.prms["dtout"] * 100))
    u = rstream.rand()
    return int(np.floor(max_offset * u))


def obs_inds(t):
    return jj + random_offset(t)


@modelling.ens_compatible
def hmod(E, t):
    return E[obs_inds(t)]


# Localization.
batch_shape = [2, 2]  # width (in grid points) of each state batch.
# Increasing the width
#  => quicker analysis (but less rel. speed-up by parallelzt., depending on NPROC)
#  => worse (increased) rmse (but width 4 is only slightly worse than 1);
#     if inflation is applied locally, then rmse might actually improve.
localizer = nd_Id_localization(shape[::-1], batch_shape[::-1], obs_inds, periodic=False)

Obs = modelling.Operator(
    M=Ny,
    model=hmod,
    noise=modelling.GaussRV(C=4 * np.eye(Ny)),
    localizer=localizer,
    loc_shift=lambda ii, dt: ii,
)

# Jacobian left unspecified coz it's (usually) employed by methods that
# compute full cov, which in this case is too big.


############################
# Other
############################
HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0, liveplotters=LP_setup(obs_inds))


####################
# Suggested tuning
####################
# Reproducing Fig 7 from Sakov and Oke "DEnKF" paper from 2008.

# Notes:
# - If N<=25, then typically need to increase the dissipation
#      to be almost sure to avoid divergence. See counillon2009.py for example.
#    - We have not had the need to increase the dissipation parameter for the EnKF.
# - Our experiments differ from Sakov's in the following minor details:
#    - We use a batch width (unsure what Sakov uses).
#    - The "EnKF-Matlab" code has a bug: it forgets to take sqrt() of the taper coeffs.
#      This is equivalent to: R_actually_used = R_reported / sqrt(2).
# - The boundary cells are all fixed at 0 by BCs,
#   but are included in the state vector (amounting to 3% of the its length),
#   and thus in RMSE calculations (which is not quite fair/optimal).

# from dapper.mods.QG.sakov2008 import HMM                       # rmse.a:
# xps += LETKF(mp=True, N=25,infl=1.04       ,loc_rad=10)        # 0.64
# xps += LETKF(mp=True, N=25,infl='-N',xN=2.0,loc_rad=10)        # 0.66
# xps += SL_EAKF(       N=25,infl=1.04       ,loc_rad=10)        # 0.62
# xps += SL_EAKF(       N=25,infl=1.03       ,loc_rad=10)        # 0.58
#
# Iterative:
# Yet to try: '-N' inflation, larger N, different loc_rad, and
# smaller Lag (testing lag>3 was worse [with this loc_shift])
# xps += iLEnKS('Sqrt',N=25,infl=1.03,loc_rad=12,nIter=3,Lag=2) # 0.59
#
# N = 45
# xps += LETKF(mp=True, N=N,infl=1.02       ,loc_rad=10)        # 0.52
# xps += LETKF(mp=True, N=N,infl='-N',xN=1.5,loc_rad=10)        # 0.51
