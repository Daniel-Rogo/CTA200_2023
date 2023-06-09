from typing import Any


def f(x: float) -> float:
    """This function evaluates the polynomial X^3 - X^2 - 1.
    """
    return x ** 3 - x ** 2 - 1


def df(x: float) -> float:
    """This function evaluates the derivative of the polynomial X^3 - X^2 - 1.
    """
    return 3 * x ** 2 - 2 * x


def newton(func, der, x_n: float, epsilon=1e-6, max_iter=30) -> Any:
    """This function uses the function  X^3 - X^2 - 1 and calculates its roots
    using the newton raphson method!

    To use this function, input the starting guess for the root. As seen below,
    it takes one extra iteration whe the epsilon value it changed.

    >>> newton(f, df, 182.3)
    Found root in 16 iterations
    1.4655712477257246

    >>> newton(f, df, -13.8)
    Found root in 11 iterations
    1.4655712458336427

    >>> newton(f, df, 182.3, 1e-8)
    Found root in 17 iterations
    1.4655712318767682

    >>> newton(f, df, -13.8, 1e-8)
    Found root in 12 iterations
    1.4655712318767682
    """
    iteration = 0
    while max_iter > 0:
        if abs(func(x_n)) >= epsilon and der(x_n) != 0:
            ratio = func(x_n) / der(x_n)
            x_n = x_n - ratio
            iteration += 1
            max_iter -= 1
        else:
            break
    if max_iter == 0 or abs(func(x_n)) < epsilon:
        print("Found root in " + str(iteration) + " iterations")
        return x_n
    else:
        print("Iteration failed")
        return None
