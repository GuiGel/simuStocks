import logging

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from numpy.polynomial import Polynomial as P

logger = logging.getLogger("simustocks")


class SimuStockError(Exception):
    """Generic exception for SimuStock"""

    def __init__(self, msg, original_exception):
        super(SimuStockError, self).__init__(f"{msg}: {original_exception}")
        self.original_exception = original_exception


class CovNotSymDefPos(SimuStockError):
    def __init__(self, matrix, original_exception, msg=None):
        if msg is None:
            # Set some default useful error message
            msg = (
                "Try to do a Choleski decomposition of a matrix that is not "
                "hermitian positive-definite."
            )
        super(CovNotSymDefPos, self).__init__(msg, original_exception)
        self.matrix = matrix


class RootError(SimuStockError):
    """Basic exception for errors raised during root founding process"""


class Simulation:
    def __init__(self, cov_h: npt.NDArray, er: npt.ArrayLike, m: int) -> None:
        """_summary_

        Args:
            cov_h (npt.NDArray): Historical prices covariance. Matrix of shape
                (k, n) where k is the number of stocks and n the number of
                prices.
            er (npt.ArrayLike): The expected anual returns for each stocks.
                List of shape (k, ) where k is the number of stocks.
            m (int): The number of daily returns to simulate.
        """
        # TODO Check that len(er), len(er) == hc.shape and m > 0
        self.cov_h = cov_h  # historical covariance
        self.er = np.array(
            er
        )  # predicted anual returns  -> must be converted to np.array
        self.m = m
        self.k = len(self.er)  # cov_h (k, k)

        assert len(self.er), len(self.er) == cov_h.shape

    def correlate(self):
        try:
            chol_h = la.cholesky(self.cov_h)
        except la.LinAlgError as e:
            raise CovNotSymDefPos(self.cov_h, e)

        s = np.random.normal(0, 1, size=(self.k, self.m))
        cov_s = np.cov(s)
        chol_f = np.linalg.cholesky(cov_s)
        g = chol_h.dot(np.linalg.inv(chol_f).dot(s))  # (4, n-1)

        # g are daily returns that must all be inferior to 1!
        assert np.all(g < 1), f"Simulated returns not inf to 1!"
        return g

    @staticmethod
    def ld(g: np.ndarray, er: np.ndarray, order: int = 4):
        # g (k, m)
        # er (k)
        # limited development of the function ln(1+x)
        # Returns array of polynomials
        # TODO check that Here 1 + expected returns is always be positive!

        assert np.all(er > -1)
        p = P([0, 1])
        x = np.array(p) + g
        assert order > 1
        lds = x.sum(axis=-1)
        for i in range(2, order):
            lds += (-1) ** (i - 1) / i * (x**i).sum(axis=-1)
        lds -= np.log(1 + er)
        return lds

    def get_root(self, dl: P, min_r: float, max_r: float) -> float:
        # ------------- compute limited development roots
        roots = P.roots(dl)

        # ------------- select real roots
        # In our case roots is dim 1 so np.where is a tuple of just one element.
        no_imag = np.imag(roots) == 0
        real_idxs = np.argwhere(no_imag).flatten()
        real_roots = np.real(np.take(roots, real_idxs))

        # ------------- select the roots that respect the constrains
        w = (1 + real_roots + min_r > 0) & (real_roots + max_r < 1)
        if not np.any(w):
            logger.warning("Not root found that respect the limits!")
        select_roots = real_roots[w]
        if len(select_roots) > 1:  # This permit (ri + root)^n --> 0
            root_arg = np.argmin(select_roots + max_r)
            root = select_roots[root_arg]
        else:
            root = select_roots[0]
        return root

    def __call__(self):
        s = self.correlate()
        lds = self.ld(s, self.er, order=3)

        min_daily_returns = s.min(axis=-1)
        max_daily_returns = s.max(axis=-1)

        alpha = []
        for dl, r_min, r_max in zip(lds, min_daily_returns, max_daily_returns):
            # r_min can be negative
            logger.info(f"{dl=}, {r_min=}, {r_max=}")
            root = self.get_root(dl, r_min, r_max)
            alpha.append(root)  # Todo --> Look for max...

        alpha = np.expand_dims(alpha, 1)
        f = s + alpha
        cov_f = np.cov(f)

        np.allclose(cov_f, self.cov_h)  # Verify that the covariance are the same.

        # Verify that the simulated annual returns are near to the expected ones.
        sr = np.exp(np.log(1 + f).sum(axis=-1)) - 1
        print(sr - self.er <= 1e-7)  # We see that the error is small
        # TODO we can log or emit a warning in this case ?
        return f, cov_f


if __name__ == "__main__":

    logging.basicConfig(encoding="utf-8", level=logging.DEBUG)

    try:
        cov_h = np.ones((3, 3))
        simu = Simulation(cov_h, [0.05, -0.04, 0.1], 50)
        s = simu.correlate()
    except CovNotSymDefPos:
        print(f"fails as expected!")

    print(f"------------------ Simulation \n")
    r1 = np.random.normal(10, 1, size=250)
    r2 = np.random.normal(200, 10, size=250)
    r3 = np.random.normal(500, 1, size=250)

    r = np.vstack((r1, r2, r3))  # The daily returns must be < 1 !

    r = np.diff(r) / r[:, 1:]
    assert np.all(
        np.abs(r) < 1
    )  # does not mean that the simulated returns will be > 1 but probable.

    # assert np.all(r > 0)
    cov_h = np.cov(r)
    print(cov_h)
    simu = Simulation(cov_h, [0.05, -0.04, 0.1], 250)
    f, f_cov = simu()

    print(f"------------------ Simulation \n")
    r1 = np.random.normal(10, 0.2, size=250)
    r2 = np.random.normal(200, 0.2, size=250)
    r3 = np.random.normal(500, 0.2, size=250)

    r = np.vstack((r1, r2, r3))  # The daily returns must be < 1 !

    assert np.all(r > 0)
    cov_h = np.cov(r)
    simu = Simulation(cov_h, [0.05, -0.04, 0.1], 250)
    f, f_cov = simu()
