import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


def mu(b, I, mu0, mu1):
    """
        Computer per capita recovery rate of infectious individuals
        Parameters
        ----------
        b: number of beds per 10 000 persons
        I: number of infective persons
        mu0: minimum recovery rate
        mu1: maximum recovery rate

        Returns recovery rate mu
        -------

    """
    return mu0 + (mu1 - mu0) * (b / (I + b))


def R0(beta, d, nu, mu1):
    """
        Computes the basic reproduction number.
        Parameters
        ----------
        beta: the average number of adequate contacts per unit time with infectious individuals
        d: per capita natural death rate
        nu: per capita disease-induced death rate
        mu1: maximum recovery rate

        Returns the basic reproduction number R0
        -------

    """
    return beta / (d + nu + mu1)


def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for Hopf bifurcation
    Parameters
    ----------
    I: number of infective persons
    mu0: minimum recovery rate
    mu1: maximum recovery rate
    beta: the average number of adequate contacts per unit time with infectious individuals
    A: recruitment rate of susceptible population
    d: per capita natural death rate
    nu: disease-induced death rate
    b: number of beds per 10 000 persons

    Returns the indicator function for bifurcation
    -------

    """
    c0 = b ** 2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    return c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3


def codimension_saddle_node(A, mu0, mu1, beta, nu, b):
    """
    Computes the codimension of a saddle node.
    Parameters
    ----------
    A: recruitment rate of susceptible population
    mu0: minimum recovery rate
    mu1: maximum recovery rate
    beta: the average number of adequate contacts per unit time with infectious individuals
    nu: disease-induced death rate
    b: number of beds per 10 000 persons
    -------

    """
    if b < (A * (mu1 - mu0)) / (beta * (beta - nu)) or b > (A * (mu1 - mu0)) / (beta * (beta - nu)):
        print("The disease free equilibrium E0 os a saddle-node for codimension 1")
    else:
        print("The disease free equilibrium E0 os an attracting semi-hyperbolic node of codimension 2")


def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
        SIR model including hospitalization and natural death.

        Parameters:
        -----------
        t: time period
        y: list or array [S, I, R]
        mu0: Minimum recovery rate
        mu1: Maximum recovery rate
        beta: average number of adequate contacts per unit time with infectious individuals
        A: recruitment rate of susceptible population (e.g. birth rate)
        d: natural death rate
        nu: disease induced death rate
        b: hospital beds per 10,000 persons
    """

    # Save state variables
    S, I, R = y[:]

    # Calculate the total population
    N = S + I + R

    # Calculate recovery rate
    m = mu(b, I, mu0, mu1)

    # Differential equations
    dSdt = A - d * S - (beta * S * I) / N
    dIdt = -(d + nu) * I - m * I + (beta * S * I) / N
    dRdt = m * I - d * R

    return [dSdt, dIdt, dRdt]

def model_with_tolerance(t, y, mu0, mu1, beta, A, d, nu, b, tol = 1e-10):
    """
        SIR model including hospitalization and natural death. If derivation is within tolerance, set it to zero.

        Parameters:
        -----------
        t: time period
        y: list or array [S, I, R]
        mu0: Minimum recovery rate
        mu1: Maximum recovery rate
        beta: average number of adequate contacts per unit time with infectious individuals
        A: recruitment rate of susceptible population (e.g. birth rate)
        d: natural death rate
        nu: disease induced death rate
        b: hospital beds per 10,000 persons
        tol: tolerance within which the derivation will be considered as zero
    """
    # Save state variables
    S, I, R = y[:]

    # Calculate the total population
    N = S + I + R

    # Calculate recovery rate
    m = mu(b, I, mu0, mu1)

    # Differential equations
    dSdt = A - d * S - (beta * S * I) / N
    dIdt = -(d + nu) * I - m * I + (beta * S * I) / N
    dRdt = m * I - d * R

    result = [dSdt, dIdt, dRdt]
    for i in range(len(result)):
        if np.abs(result[i]) < tol:
            result[i] = 0
    return result