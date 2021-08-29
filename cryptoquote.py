import random
import string
from typing import NoReturn, Optional

import errors


class Quote:
    def __init__(self, value) -> None:
        self.quote = value.upper()

    def __str__(self) -> str:
        return self.quote

    def __iter__(self) -> iter:
        return iter(self.quote)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.quote!r})"


class Key:
    def __init__(self):
        self.key = self._create_key()
        self.mapping = self._create_mapping()

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key!r}, {self.mapping!r})"

    @staticmethod
    def _create_key() -> str:
        """Set Key to a random value or a given value.
        Returns:
            str: Newly created key
        """
        return random.sample(string.ascii_uppercase, len(string.ascii_uppercase))

    def _create_mapping(self) -> dict:
        """Map the standard alphabet to the encrypted alphabet.

        Raises:
            ImproperKeyError: Must have a key to create a map

        Returns:
            dict: Mapping dict with the standard alphabet as keys
            and the encrypted alphabet as values.
        """
        if self.key is None:
            raise errors.ImproperKeyError

        new_key = dict(zip(string.ascii_uppercase, self.key))
        new_key[" "] = " "

        return new_key


class Cryptoquote:
    def __init__(self, quote: Quote, key: Key) -> None:
        self.quote = quote
        self.key = key
        self.crypto = ""
        self._encrypt()

    def __str__(self) -> str:
        return self.crypto

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.quote!r}, {self.key!r}, {self.crypto!r})"
        )

    def _encrypt(self) -> NoReturn:
        """Create a 'encrypted' version of the Quote using Key.

        Raises:
            CryptoquoteAlreadyEncrypted: Cryptoquote can only be encrypted once.

        Returns:
            NoReturn
        """
        if self.crypto:
            raise errors.CryptoquoteAlreadyEncryptedError

        for char in self.quote:
            if char not in string.ascii_uppercase:
                self.crypto += char
                continue
            self.crypto += self.key.mapping[char]
