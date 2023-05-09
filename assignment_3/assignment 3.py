import math
from typing import Any
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def complex_setup() -> Any:
    x_val = np.linspace(-2, 2, 1000)
    y_val = np.linspace(-2, 2, 1000)
    x, y = np.meshgrid(x_val, y_val, indexing='ij')
    c_array = x + y * 1.j
    return c_array


def z_array() -> Any:
    time_arr = np.zeros([1000, 1000])
    dichromatic = np.zeros([1000, 1000])
    c_arr = complex_setup()
    for row in range(1000):
        for col in range(1000):
            time = time_update(c_arr[row][col])
            time_arr[row][col] = time
            if time < 100:
                dichromatic[row][col] = 0.5
    return time_arr, dichromatic


def time_update(arr) -> Any:
    time = 0
    z = 0
    while time <= 100:
        time += 1
        z_new = (z ** 2) + arr
        if abs(z_new) < 10e6:
            z = z_new
        else:
            break
    return time


def run_all():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    a, b = z_array()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle("A graphical representation of the mandelbrot set")
    ax = ax1
    pcm = ax.pcolormesh(x, y, abs(a))
    fig.colorbar(pcm, ax=ax)
    bx = ax2
    pcm = bx.pcolormesh(x, y, b)
    bounds = [0, 1]
    cmap = mpl.colors.ListedColormap(['r'])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(pcm, cmap=cmap, norm=norm, ax=bx,
                 boundaries=[-0.01] + bounds + [1.01],
                 ticks=bounds)
    plt.show()


def lorenz(t, w_naught, sigma, rho, beta):
    """The Lorenz equations."""
    u, v, w = w_naught
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


def lorenz_solution(max_time=60):
    w_naught = (0., 1., 0.)
    sigma, rho, beta = 10., 28, 8. / 3.
    solution = solve_ivp(lorenz, (0, max_time), w_naught,
                         args=(sigma, rho, beta), dense_output=True)
    return solution


def slightly_different_lorenz(max_time=60):
    w_naught = (0.00000000, 1.00000001, 0.00000000)
    sigma, rho, beta = 10., 28, 8. / 3.
    solution = solve_ivp(lorenz, (0, max_time), w_naught,
                         args=(sigma, rho, beta), dense_output=True)
    return solution


def distance_between_slight_changes():
    """
    """
    original = lorenz_solution()
    original_sol = original.y
    o_x = original_sol[0]
    o_y = original_sol[1]
    o_z = original_sol[2]
    different = slightly_different_lorenz().y
    d_x = different[0]
    d_y = different[1]
    d_z = different[2]
    time_array = original.t
    difference_arr = np.zeros([718])
    for n in range(len(difference_arr)):
        difference_arr[n] = math.log10(distance_helper(o_x[n], o_y[n], o_z[n],
                                                       d_x[n], d_y[n], d_z[n]))
    plt.plot(time_array, difference_arr)
    plt.xlabel("The change in time (0.01s intervals)")
    plt.ylabel("logarithmic difference")
    plt.title("Solution differences between W and W'")
    plt.show()


def distance_helper(x1, y1, z1, x2, y2, z2) -> float:
    x_dis = (x2 - x1) ** 2
    y_dis = (y2 - y1) ** 2
    z_dis = (z2 - z1) ** 2
    return math.sqrt(x_dis + y_dis + z_dis)


def show_figure_1(max_time=60):
    """
    >>> show_figure_1()
    [123]
    """
    solution = lorenz_solution(max_time)
    time_array = solution.t
    solution_array = solution.y
    for row in range(len(solution_array)):
        for col in range(0, 504):
            solution_array[row][col] = int(solution_array[row][col] * 10)
    plt.plot(time_array, solution_array[1])
    plt.xlabel("The change in time (0.01s intervals)")
    plt.ylabel("Numerical solution to the convection equation in the Y-axis")
    plt.title("A reproduction of Lorenz's figure 1")
    plt.show()


def show_figure_2(max_time=60):
    solution = lorenz_solution(max_time)
    solution = slightly_different_lorenz(max_time)
    t = np.linspace(14, 60, 10000)
    x, y, z = solution.sol(t)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(y, z)
    ax2.plot(x, y)
    ax1.set_xlabel("The Y component to the lorenz system")
    ax1.set_ylabel("The Z component to the lorenz system")
    ax2.set_xlabel("The X component to the lorenz system")
    ax2.set_ylabel("The Y component to the lorenz system")
    plt.suptitle("Numerical solution to the convection equations")
    plt.show()
