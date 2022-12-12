from typing import Any

import numpy as np
import numpy.typing as npt
import pandera as pa
from pydantic import BaseModel, validator

prices_schema = pa.DataFrameSchema(
    {
        ".*": pa.Column(
            float,
            checks=[
                pa.Check.greater_than_or_equal_to(0, ignore_na=False),
            ],
            regex=True,
        ),
    },
    index=pa.Index(np.dtype("datetime64[ns]"), name="Date"),
)


class Stocks(BaseModel):

    df: Any  # DataFrame (n, k)

    @property
    def prices(self) -> npt.NDArray:
        return self.df.to_numpy()  # shape (n, k)

    @validator("df")
    def schema(cls, v):
        return prices_schema(v)

    @property
    def cov(self) -> npt.NDArray:
        return np.cov(self.returns)  # (k, k)

    @property
    def returns(self) -> npt.NDArray:
        return self.df.pct_change()[1:].to_numpy().T  # (k, n-1)
