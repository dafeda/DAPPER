"""Demonstrate the Linear Advection (LA) model."""
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.LA.raanes2015 import X0, step
from dapper.tools.viz import amplitude_animation

x0 = X0.sample(1).squeeze()
dt = 1
xx = modelling.run_forward(step, nsteps=500, x=x0, t=0, dt=dt, prog="Simulating")

anim = amplitude_animation(xx, dt)
plt.show()
