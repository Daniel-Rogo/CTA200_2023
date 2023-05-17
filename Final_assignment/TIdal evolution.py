import math
from typing import Any
import numpy as np
from matplotlib import pyplot as plt, ticker
from scipy.integrate import solve_ivp


def get_values_in_si(unit: str, val: float) -> Any:
    if unit == "g":
        return val / 1000.
    elif unit == "cm":
        return val / 100.
    elif unit == "km":
        return val * 1000
    elif unit == "s":
        return val / 3.154e+7
    elif unit == "1/s":
        return val * 3.154e+7
    elif unit == "1/s^2":
        return val * 3.154e+7 * 3.154e+7
    else:
        return "conversion fail"


# Sun
M_sun = get_values_in_si("g", 1.98e+33)

# Earth
M_earth = get_values_in_si("g", 5.97e+27)
semi_major_axis_earth = get_values_in_si("km", 1.49e8)
R_earth = get_values_in_si("km", 6371)
K_earth = 0.298
Angular_Velo_earth_initial = (2 * math.pi) / 86164
I_earth = 0.3299 * M_earth * (R_earth ** 2)

# Moon
semi_major_axis_lunar_initial = get_values_in_si("km", 384000)
M_lunar = get_values_in_si("g", 7.349e+25)

# Physics constants
Tidal_quality_factor = 11.5
G_const = 6.67e-11


def initial_values() -> Any:
    """This function initialises the present day values of the earth and moon
    angular momentum, and the earth's spin
    """
    a_m_e_i = M_earth * math.sqrt((M_earth + M_sun) * G_const *
                                  semi_major_axis_earth)
    s_m_e_i = I_earth * Angular_Velo_earth_initial
    a_m_l_i = M_lunar * math.sqrt((M_earth + M_lunar) * G_const *
                                  semi_major_axis_lunar_initial)
    return a_m_e_i, s_m_e_i, a_m_l_i


# This is run to set the initial values of equations 6, 7 and 8.
Angular_moment_earth_initial, Angular_Spin_moment_earth_initial, Angular_moment_lunar_initial = initial_values()

# Converting all seconds to years
G_const_year = get_values_in_si("1/s^2", G_const)
Angular_Velo_earth_initial_year = get_values_in_si("1/s", Angular_Velo_earth_initial)
Angular_moment_earth_initial_year = get_values_in_si("1/s", Angular_moment_earth_initial)
Angular_Spin_moment_earth_initial_year = get_values_in_si("1/s", Angular_Spin_moment_earth_initial)
Angular_moment_lunar_initial_year = get_values_in_si("1/s", Angular_moment_lunar_initial)

def lunar_tidal_torque(axis, unit:str) -> Any:
    """This function uses the semi-major axis of the moon to calculate the
    lunar tidal torque"""
    if unit == "year":
        g = G_const_year
    else:
        g = G_const
    part1 = (g * (M_lunar ** 2)) / axis
    part2 = (R_earth / axis) ** 5
    part3 = K_earth / Tidal_quality_factor
    t_lunar = 1.5 * part1 * part2 * part3
    return t_lunar

# This sets the lunar torque in SI units and years
T_lunar = lunar_tidal_torque(semi_major_axis_lunar_initial, "s")
T_lunar_year = lunar_tidal_torque(semi_major_axis_lunar_initial, "year")


def solar_tidal_torque(t_lunar, semi_lunar):
    """This uses the semi-major axis of the moon and the lunar torque to
    calculate the solar torque"""
    beta = (1 / 4.7) * ((semi_lunar / semi_major_axis_lunar_initial) ** 6)
    return t_lunar * beta

# This gets the solar tidal torque in seconds and years
T_sol_init = solar_tidal_torque(T_lunar, semi_major_axis_lunar_initial)
T_sol_init_year = solar_tidal_torque(T_lunar_year, semi_major_axis_lunar_initial)

def axis_reversal(angular_lunar, unit="year"):
    """Takes the angular momentum of the moon and solves for the semi-major
    axis that corresponds to it"""
    if unit == "s":
        g = G_const
    else:
        g = G_const_year
    bottom = ((g * (M_earth + M_lunar)) * (M_lunar ** 2))
    top = angular_lunar ** 2
    return top / bottom


def integral(t, axis, unit="year"):
    """This is the function that is used by the integrator to integrate the 3
    differential equations from the assignment plus a new differential equation
    I used to calculate the semi-major axis change of the moon"""
    semi_axis = axis_reversal(axis[2], unit)
    t_lunar = lunar_tidal_torque(semi_axis, unit)
    t_solar = solar_tidal_torque(t_lunar, semi_axis)

    d_axis = moon_receding(semi_axis, axis[2])

    d_e_moment = t_solar
    d_e_spin = -1 * t_solar - t_lunar
    d_l_moment = t_lunar
    return d_e_moment, d_e_spin, d_l_moment, d_axis


def integral_solution():
    """This is the function that uses the solve_ivp to integrate the
    differential equations. It can be done in seconds by uncommenting the
    second block, however I did not make everything compatible with it.
    To simplify the equations, I used all values in years to minimise the
    amount of possible iterations. """
    w_naught = (Angular_moment_earth_initial_year,
                Angular_Spin_moment_earth_initial_year,
                Angular_moment_lunar_initial_year, semi_major_axis_lunar_initial)
    evaluation_period = [0, -1 * 4e9]

    # w_naught = (Angular_moment_earth_initial,
    #             Angular_Spin_moment_earth_initial,
    #             Angular_moment_lunar_initial)
    # evaluation_period = [0, -1.26144e+17]

    time_interval = np.linspace(evaluation_period[0], evaluation_period[1], 160)
    solution = solve_ivp(integral, evaluation_period, w_naught, t_eval=time_interval)
    return solution


def time_scales():
    """This function generates the timescales based on current values and then
    converts the result from seconds to years"""
    time_e_moment = Angular_moment_earth_initial / T_sol_init
    time_e_spin = Angular_Spin_moment_earth_initial / (T_sol_init + T_lunar)
    time_l_moment = Angular_moment_lunar_initial / T_lunar
    return time_e_moment / 3.1536e+7, time_e_spin / 3.1536e+7, time_l_moment / 3.1536e+7


# gets the timescales
Time_scale_Earth_moment, Time_scale_Earth_spin, Time_scale_Lunar_moment = time_scales()


def moon_receding(semi, l_lunar):
    """This function is a test to see if the values used are accurate. Since
    it is known that the moon recedes at 3cm/year, getting a similar result
    means that the values are correct.
    """
    t_lunar = lunar_tidal_torque(semi, "s")
    l_lunar = l_lunar / 3.154e+7
    a = 2 * semi
    b = t_lunar / l_lunar
    c = a * b
    c = c * 3.1536e+7
    return c

Receding_moon = str(moon_receding(semi_major_axis_lunar_initial, Angular_moment_lunar_initial_year)) + " m/year"


def length_of_day(spin, unit):
    """This function uses the earths angular spin to calculate the period of
    time needed to complete a day"""
    if unit == "year":
        spin = spin / 3.154e+7
    day_length = np.zeros(62)
    for index in range(len(spin)):
        ang_spin = spin[index]
        time = (2 * np.pi * I_earth) / ang_spin
        day_length[index] = time
    return day_length



def roche_limit_day():
    """Calculates the day length when the moon was at the roche limit"""
    tot = Angular_moment_lunar_initial + Angular_Spin_moment_earth_initial
    roche_angular = M_lunar * math.sqrt((M_earth + M_lunar) * G_const * (18000 * 1000))
    roche_spin = tot - roche_angular
    day_length = length_of_day([roche_spin], "s")
    day_length = day_length / 3600
    return day_length[0]


# Gets the day length at the roche limit
day_length_at_roche = roche_limit_day()


def plot_all_in_one():
    """Plots all three graphs used in one, for easy access to all graphs"""
    data = integral_solution()
    data_y = data.y
    data_1 = data_y[0] / 3.154e+7
    data_2 = data_y[1]
    data_3 = data_y[2] / 3.154e+7
    data_axis = data_y[3] / 1000
    data_t = data.t

    plt.subplot(151)
    plt.plot(data_t, data_1)
    plt.title("Earth's angular momentum with respect to time")
    plt.xlabel("Time (Billion years)")
    plt.ylabel(r"Angular Momentum $((Kg\ m^2)/s)$")

    plt.subplot(152)
    plt.plot(data_t, data_2 / 3.154e+7)
    plt.title("Earth's angular spin with respect to time")
    plt.xlabel("Time (Billion years)")
    plt.ylabel(r"Angular Spin $((Kg\ m^2)/s)$")

    plt.subplot(153)
    plt.plot(data_t, data_3)
    plt.title("Lunar angular momentum with respect to time")
    plt.xlabel("Time (Billion years)")
    plt.ylabel(r"Angular Momentum $((Kg\ m^2)/s)$")

    plt.subplot(154)
    plt.plot(data_t, data_axis)
    plt.title("Lunar semi-major axis with respect to time")
    plt.xlabel("Time (Billion years)")
    plt.ylabel(r"Semi-Major Axis $(Km)$")

    day_length = length_of_day(data_2, "year")
    plt.subplot(155)
    plt.plot(data_t, day_length / 3600)
    plt.title("Day length with respect to time")
    plt.xlabel("Time (Billion years)")
    plt.ylabel(r"Day length $(Hours)$")

    plt.show()


def plot_3_graphs():
    """Plots three separate graphs that splits the graphs in the format in
    which they are used in the paper."""
    data = integral_solution()
    data_y = data.y
    data_1 = data_y[0] / 3.154e+7
    data_2 = data_y[1]
    data_3 = data_y[2] / 3.154e+7
    data_axis = data_y[3] / 1000
    data_t = data.t

    fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
    fig1.suptitle('Earth - Moon system relationships',fontsize=20)
    ax1.plot(data_t, data_1 * 1.0)
    ax1.set_title("Earth's angular momentum WRT time",fontsize=15)
    ax1.set_ylabel(r"Angular Momentum $((Kg\ m^2)/s)$",fontsize=12)

    ax2.plot(data_t, data_2 / 3.154e+7)
    ax2.set_title("Earth's angular spin WRT time", fontsize=15)
    ax2.set_ylabel(r"Angular Spin $((Kg\ m^2)/s)$",fontsize=12)

    ax3.plot(data_t, data_3)
    ax3.set_title("Lunar angular momentum WRT time", fontsize=15)
    ax3.set_xlabel("Time (Billion years)",fontsize=12)
    ax3.set_ylabel(r"Angular Momentum $((Kg\ m^2)/s)$",fontsize=12)
    fig1.set_figheight(8.5)
    fig1.savefig("Relationships between the Earth-Moon system.pdf", format="pdf", bbox_inches="tight")

    fig2, (ax4) = plt.subplots(1)
    fig2.suptitle("Lunar semi-major axis with respect to time")
    ax4.plot(data_t, data_axis)
    ax4.set_xlabel("Time (Billion years)")
    ax4.set_ylabel(r"Semi-Major Axis $(Km)$")
    fig2.set_figwidth(6)
    fig2.set_figheight(4)
    fig2.savefig("Lunar semi-major axis graph.pdf", format="pdf", bbox_inches="tight")

    day_length = length_of_day(data_2, "year")
    fig3, (ax5) = plt.subplots(1)
    fig3.suptitle("Day length with respect to time")
    ax5.plot(data_t, day_length / 3600)
    ax5.set_xlabel("Time (Billion years)")
    ax5.set_ylabel(r"Day length $(Hours)$")
    fig3.set_figwidth(6)
    fig3.set_figheight(4)
    fig3.savefig("Day length over time.pdf", format="pdf", bbox_inches="tight")

    plt.show()

plot_3_graphs()
