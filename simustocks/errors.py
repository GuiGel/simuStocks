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
