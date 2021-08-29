class CryptoquoteAlreadyEncryptedError(Exception):
    """Exception for Cryptoquote that has already been encrypted."""

    pass


class ImproperKeyError(Exception):
    """Key must be 26 unique uppercase alphabetical letters."""

    pass


class EmptyText(Exception):
    """Text length must be greater than zero."""

    pass


class TooShort(Exception):
    """Text must be certain length"""

    pass


class UnknownGraph(Exception):
    """Graph parameter was not single or bigram frequency"""

    pass
