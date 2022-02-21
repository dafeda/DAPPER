from matplotlib import ticker
import numpy as np

from dapper.tools.chronos import Ticker
import dapper.mods as modelling


def test_chronology_replacement():
    T = 300
    dt = 1
    K = round(T / dt)
    dko = 5
    tseq = modelling.Chronology(dt=dt, dko=dko, T=T, BurnIn=-1, Tplot=100)
    # obs_mask is zero for time epochs where we don't have a observation,
    # and one when we do have a observation.
    obs_mask = np.zeros(K, dtype=int)
    obs_mask[np.arange(dko, K + dko, dko) - 1] = 1
    tseq2 = modelling.Chronology2(dt=1, T=300, obs_mask=obs_mask)

    print(tseq2.kko)

    assert (tseq.kko == tseq2.kko).all()
    assert tseq.K == tseq2.K
    assert tseq.Ko == tseq2.Ko


def test_ticker():
    dt = 1
    K = 10
    dko = 2
    obs_mask = np.zeros(K, dtype=int)
    obs_mask[np.arange(dko, K + dko, dko) - 1] = 1
    kko = np.where(obs_mask == 1)[0] + 1
    kk = np.arange(K + 1)
    tt = kk * dt

    tckr = Ticker(tt, kko)

    tseq = modelling.Chronology2(dt=dt, T=K, obs_mask=obs_mask)

    next(tckr)
    for ticker in tseq.ticker:
        kk_current, kko_current, tt_current, dt_current = next(tckr)
        assert kk_current == ticker.kk
        assert kko_current == ticker.kko
        assert tt_current == ticker.tt
        assert dt_current == ticker.dt
