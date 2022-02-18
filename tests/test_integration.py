import numpy as np

import dapper.mods as modelling
from dapper.mods.LA.raanes2015 import X0, step


def test_with_recursion():
    simulator = modelling.with_recursion(step, prog="Simulating")

    x0 = X0.sample(1).squeeze()

    dt = 1
    t = 0
    k = 100
    xx_sim = simulator(x0, k=k, t=t, dt=dt)
    xx_forward = modelling.run_forward(step, k, x0, t, dt, "Simulating")

    assert np.array_equal(xx_sim, xx_forward)
