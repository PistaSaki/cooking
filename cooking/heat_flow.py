import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, cos


# %%

# The model is as follows:
# we have a homogenous rod spanning from -1 to 1. Its initial temperature is constant 1 and we keep the ends at 0.
# We are interested in the temperature evolution of the cooling rod.

# Using the Fourrier method, we express the starting teperature function (constant 1) as
# $\sum_k c_k \cos(\omega_k x)$ where $\omega_k = \pi (k + 1/2)$
# and $c_k$ can be easily calculated using L2 scalar product.
# Each of the cosine-functions is an eigen-vector of the laplacian with the eigen-value $- \omega_k^2$.
# Thus with time each coefficient deteriorates as $exp( - \alpha \omega_k^2 t)$ where $\alpha$ is

def _omega(k):
    """ Return  frequency of k-th cosine wave whose values at +1, -1 are 0

    That is, for any integer k >= 0, the function `x -> cos(omega_k * x)` has zeros at +1, -1
    """
    return pi * (k + 1 / 2)

def _c(k):
    """Return coefficient of `cos(omega_k *x)` in the expansion of constan 1"""
    return (2 * (-1) ** k) / (pi * (k + 1 / 2))


def u(t, x, max_k=10, alpha=1.):
    """Temperature at position x at time t of a homogenous rod

    The rod starts at x=-1 and ends at x=1. Its starting temperature at t=0 is 1 and its ends are kept at temperature 0.
    """
    return sum(_c(k) * exp(-_omega(k) ** 2 * alpha * t) * cos(_omega(k) * x) for k in range(max_k))


def integral_u(t, max_k=10, alpha=1.):
    return sum(_c(k) ** 2 * exp(-_omega(k) ** 2 * alpha * t) for k in range(max_k))


# %%
fig, ax = plt.subplots()
xxx = np.linspace(-1, 1, 100)
for t in np.linspace(0, 1, 6):
    uuu = u(t=t, x=xxx, max_k=100)
    ax.plot(xxx, uuu, label=f"time={t}")

ax.legend()
ax.set_xlabel("position")
ax.set_ylabel("temperature")
ax.set_title("Temperature evolution when cooling from 1 to 0")
# %%
max_k = 100
times = np.linspace(0, 5, 200)
middle_temperatures = u(t=times, x=0, max_k=max_k)
relaxed_temperatures = 1 / 2 * integral_u(t=times, max_k=max_k)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 4])
ax1.plot(times, middle_temperatures, label="temperature in the middle")
ax1.plot(times, relaxed_temperatures, label="relaxed temperature")
ax1.set_xlabel("time")
ax1.set_ylabel("temperature")
ax1.legend()

ax2.plot(middle_temperatures, relaxed_temperatures, label="relaxed vs middle temperature")
ax2.plot(middle_temperatures, 2 / pi * middle_temperatures, label="$\\frac{2}{\\pi} \\times$ middle_temperature")
ax2.set_xlabel("middle temperature")
ax2.set_ylabel("relaxed temperature")
ax2.legend()

fig.suptitle("cooling from 1 to 0")

#############
# the above would be valid if our stuff-temperature was 1 and the environment temperature was 0.
# We are interested in heating instead of cooling so we make the transformation temperature -> - temperature
middle_temperatures = - middle_temperatures
relaxed_temperatures = - relaxed_temperatures

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 4])
ax1.plot(times, middle_temperatures, label="temperature in the middle")
ax1.plot(times, relaxed_temperatures, label="relaxed temperature")
ax1.set_xlabel("time")
ax1.set_ylabel("temperature")
ax1.legend()

ax2.plot(middle_temperatures, relaxed_temperatures, label="relaxed vs middle temperature")
ax2.plot(middle_temperatures, 2 / pi * middle_temperatures, label="$\\frac{2}{\\pi} \\times$ middle_temperature")
# ax2.plot(middle_temperatures, 1 - 2 / pi * (1 - middle_temperatures),
#          label="1 - $\\frac{2}{\\pi} \\times$ (1 - middle_temperature)")
ax2.set_xlabel("middle temperature")
ax2.set_ylabel("relaxed temperature")
ax2.legend()

fig.suptitle("heating from -1 to 0")
