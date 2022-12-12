import numpy as np
import numpy.typing as npt
import pandas as pd
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

    df: pd.DataFrame  # DataFrame (n, k)

    @property
    def prices(self) -> npt.NDArray:
        return self.df.to_numpy().T  # shape (k, n)

    @validator("df")
    def schema(cls, v):
        return prices_schema(v)

    @property
    def cov(self) -> npt.NDArray:
        return np.cov(self.returns)  # (k, k)

    @property
    def returns(self) -> npt.NDArray:
        return self.df.pct_change()[1:].to_numpy().T  # (k, n-1)

    class Config:
        arbitrary_types_allowed = True


if __name__ == "__main__":
    prices = pd.DataFrame(
        [[1.0, np.nan, 2.0], [1.0, 2.0, 3.0], [1.0, 3.0, -2, 0]],
        columns=["a", "b", "c", "d"],
    )
    print(prices.iloc[1][0])
    stocks = Stocks(df=prices)
