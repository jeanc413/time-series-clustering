from typing import Any


class CaptureData(list):
    """After being instantiated, everytime this class is called, it will
    store a unique `observation` parameter in the self-list, and return the same value.

    Examples
    --------
    >>> capture = CaptureData()
    >>> capture(10)
    10
    >>> capture(11)
    11
    >>> capture
    [10, 11]

    >>> capture = CaptureData()
    >>> capture(1)
    1
    >>> capture(["a", "b"])
    ['a', 'b']
    >>> capture(30)
    30
    >>> capture
    [1, ['a', 'b'], 30]

    """

    def __int__(self):
        super().__init__()

    def __call__(self, observation: Any):
        """Add any object to class list and return the same value

        Parameters
        ----------
        observation : Any
            Object to be added to the list.

        Returns
        -------
        observation : Any
            Same provided object

        """
        self.append(observation)
        return observation
