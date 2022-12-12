import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from numpy.polynomial import Polynomial as P

from simustocks.errors import CovNotSymDefPos
from simustocks.stocks import Stocks

logger = logging.getLogger("simustocks")


class Simulation:
    def __init__(self, stock: Stocks, er: Dict[str, float], m: int) -> None:
        """_summary_

        Args:
            cov_h (npt.NDArray): Historical prices covariance. Matrix of shape
                (k, k) where k is the number of stocks and n the number of
                prices.
            er (npt.ArrayLike): The expected anual returns for each stocks.
                List of shape (k, ) where k is the number of stocks.
            m (int): The number of daily returns to simulate.
        """
        self.k, n = stock.prices.shape
        self.stock = stock
        self.cov_h = stock.cov  # historical covariance
        self.er = np.array(list(er.values()))
        self.m = m

        assert len(self.er), len(self.er) == self.cov_h.shape
        assert m > 0

    @property
    def init_prices(self):
        # Get the initialization price for the simulation, the last
        # prices of the historical prices
        return self.stock.prices[:, -1:]  # (k, 1)

    def correlate(self):
        """Create random vectors that have a given covariance matrix"""
        try:
            chol_h = la.cholesky(self.cov_h)
        except la.LinAlgError as e:
            raise CovNotSymDefPos(self.cov_h, e)

        s = np.random.normal(
            0, 1, size=(self.k, self.m)
        )  # random simulated prices daily returns
        cov_s = np.cov(s)
        chol_f = np.linalg.cholesky(cov_s)
        g = chol_h.dot(np.linalg.inv(chol_f).dot(s))  # (k, m)

        # g are daily returns that must all be inferior to 1!
        if np.all(g < 1):
            logger.warning("Correlated daily returns not all inf to 1!")

        return g  # (k, k)

    @staticmethod
    def ld(g: npt.NDArray, er: npt.NDArray, order: int = 4):
        # g (k, m)
        # er (k)
        # limited development of the function ln(1+x) with x = rc + alpha
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
            logger.warning("Not roots respect the constraints!")
        select_roots = real_roots[w]
        if len(select_roots) > 1:  # This permit (ri + root)^n --> 0
            root_arg = np.argmin(select_roots + max_r)
            root = select_roots[root_arg]
        else:
            root = select_roots[0]
        return root

    def get_returns_adjustment(self, s: npt.NDArray, order: int = 10) -> npt.NDArray:
        # order > 2
        lds = self.ld(s, self.er, order=order)

        min_daily_returns = s.min(axis=-1)
        max_daily_returns = s.max(axis=-1)

        alpha: List[float] = []
        for dl, r_min, r_max in zip(lds, min_daily_returns, max_daily_returns):
            # r_min can be negative
            logger.info(f"{dl=}, {r_min=}, {r_max=}")
            root = self.get_root(dl, r_min, r_max)
            alpha.append(root)  # Todo --> Look for max...

        return np.expand_dims(alpha, axis=1)  # (k, 1)

    def get_future_prices(
        self, init_prices: npt.NDArray, returns: npt.NDArray
    ) -> npt.NDArray:
        returns_extend = np.concatenate(
            [np.ones((self.k, 1)), (returns + 1).cumprod(axis=1)], axis=1
        )  # (k, m + 1)
        prices = (
            returns_extend * init_prices
        ).T  # Reconstruct the price from the last prices values. (k, m + 1)
        return prices  # (k, m + 1)

    def check_returns(self, simulated_returns):
        # Check that the simulated anual returns are near to the expected ones
        sr = Simulation.get_anual_returns_from_daily_returns(simulated_returns)
        check = np.allclose(sr, self.er)
        logger.info(
            f"Are the simulated anual returns equal to the expected ones?  {check}"
        )
        if not check:
            returns_errors = sr - self.er
            stocks_name = self.stock.df.columns.to_list()
            name_returns = dict(zip(stocks_name, returns_errors))
            logger.debug(f"anual returns error: {name_returns}")

    def check_covariance(self, cov_s: npt.NDArray) -> None:
        check = np.allclose(cov_s, self.cov_h)
        logger.debug(
            f"Is the simulated covariance matrix the same as the historical one? {check}"
        )

    @staticmethod
    def get_anual_returns_from_daily_returns(daily_returns: npt.NDArray) -> npt.NDArray:
        # daily_returns (k, m)
        # returns: (k,)
        sar = np.exp(np.log(1 + daily_returns).sum(axis=-1)) - 1
        return sar

    def __call__(
        self, order: int = 10, precision: Optional[float] = None
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # future_price (m + 1, k)
        correlated_returns = self.correlate()
        adjustment = self.get_returns_adjustment(correlated_returns, order=order)
        simulated_returns = correlated_returns + adjustment

        cov_s = np.cov(simulated_returns)
        self.check_covariance(cov_s)

        self.check_returns(simulated_returns)

        future_prices = self.get_future_prices(self.init_prices, simulated_returns)

        return simulated_returns, cov_s, future_prices
