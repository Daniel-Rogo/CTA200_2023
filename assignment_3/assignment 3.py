import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


# ALL THE CODE THAT YOU NEED TO RUN IS AT THE BOTTOM

def z_array():
    """This function takes the complex value array and iterates it using the
    required formula of z_new = z^2 + c. This returns a large array of values
    that represent the growth of the mandelbrot set
    """
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


def complex_setup():
    """This function creates the array of complex values that are required
    for the z_array function. The array is taking x and iy values
    between -2 and 2.
    """
    x_val = np.linspace(-2, 2, 1000)
    y_val = np.linspace(-2, 2, 1000)
    x, y = np.meshgrid(x_val, y_val, indexing='ij')
    c_array = x + y * 1.j
    return c_array


def time_update(arr):
    """THis is a helper function for the z_array function. It takes in the
    current array that z_array is iterating through and then goes the iterative
    calculation of the mandelbrot formula. It returns the number of iterations
    needed to surpass a limit or stay bounded.
    """
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


def lorenz(t, w_naught, sigma, rho, beta):
    """This is the basic function of the lorenz equations, it takes in the
    arguments for initial conditions, and the sigma, rho, beta conditions.
    It returns the values for dx, dy and the dz with respect to time
    """
    x, y, z = w_naught
    xp = -sigma * (x - y)
    yp = rho * x - y - x * z
    zp = -beta * z + x * y
    return xp, yp, zp


def lorenz_solution(max_time=60):
    """This function uses solve_ivp to iteratively solve the lorenz equation
    given the same initial condition used in the original paper
    """
    w_naught = (0., 1., 0.)
    sigma, rho, beta = 10., 28, 8. / 3.
    interval = np.linspace(0, 30, 3000)
    solution = solve_ivp(lorenz, (0, max_time), w_naught,
                         args=(sigma, rho, beta), dense_output=True,
                         t_eval=interval)
    return solution


def slightly_different_lorenz(max_time=60):
    """This function uses very lightly modified version of the initial
    conditions to generate the different diagram, this graph is used to show
    how a small difference results in a large change over time
    """
    w_naught = (0.00000000, 1.00000001, 0.00000000)
    sigma, rho, beta = 10., 28, 8. / 3.
    interval = np.linspace(0, 30, 3000)
    solution = solve_ivp(lorenz, (0, max_time), w_naught,
                         args=(sigma, rho, beta), dense_output=True,
                         t_eval=interval)
    return solution


def distance_helper(x1, y1, z1, x2, y2, z2):
    """This is a helper function that computes the distance between a set of 3
    dimensional coordinates"""
    x_dis = (x2 - x1) ** 2
    y_dis = (y2 - y1) ** 2
    z_dis = (z2 - z1) ** 2
    return math.sqrt(x_dis + y_dis + z_dis)


# These are all the codes you need to run to give all the outputs.
# - run_all() prints the outputs of the first question
# - show_figure_1 or 2 prints the outputs the lorenz data
# - distance_between_slight_changes outputs the difference between two 
# slightly different starting points


def run_all():
    """This function calls z_array to get the data from the mandelbrot set and
    then uses matplotlib to plot the graphs. The left graph is the time based
    graph, showing the number of iteration a certain section took to reach
    infinity, and the right graph is a dichromatic graph showing the values
    that go to infinity and those who stay bounded."""
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


def show_figure_1(max_time=60):
    """This function returns the figure 1 from the original lorenz paper.
    The graph represents the integrated y value (of the lorenz equations
    with respect to time.
    """
    solution = lorenz_solution(max_time)
    time_array = solution.t
    solution_array = solution.y
    for row in range(len(solution_array)):
        for col in range(0, 504):
            solution_array[row][col] = int(solution_array[row][col] * 10)
    plt.plot(time_array[:300] * 100, solution_array[1][:300])
    plt.xlabel("The change in time (0.01s intervals)")
    plt.ylabel("Numerical solution to the convection equation in the Y-axis")
    plt.title("A reproduction of Lorenz's figure 1")
    plt.show()


def show_figure_2(max_time=60):
    """This function uses the lorenz equation data to plot the Y-Z and X-Y
    graphical relations.
    """
    solution = lorenz_solution(max_time)
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


def distance_between_slight_changes():
    """This function takes in the data from the original lorenz graph and the
    slightly modified graph, then finds the distance at every point between
    the two. The end result shows a logarithmic distance change between the
    two graphs over a long period of time. It the outputs the log-linear graph.
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
    difference_arr = np.zeros([3000])
    for n in range(len(difference_arr)):
        difference_arr[n] = math.log10(distance_helper(o_x[n], o_y[n], o_z[n],
                                                       d_x[n], d_y[n], d_z[n]))
    plt.plot(time_array, difference_arr)
    plt.xlabel("The change in time (0.01s intervals)")
    plt.ylabel("logarithmic difference")
    plt.title("Solution differences between W and W'")
    plt.show()

    
input("type anything to see the first figure")
print("close this figure to open the next")
run_all()
input("type anything to continue to the next figure!")
print("close this figure to open the next")
show_figure_1()
input("type anything to continue to the next figure!")
print("close this figure to open the next")
show_figure_2()
input("type anything to continue to the next figure!")
print("close this figure to open the next")
distance_between_slight_changes()
