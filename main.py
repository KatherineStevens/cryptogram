import numpy as np
import string
from cryptoquote import Quote, Cryptoquote, Key
from collections import Counter
from errors import EmptyText, TooShort
from typing import Optional
import graphs


ACCEPTABLE_ALPHA = ' ' + string.ascii_uppercase
ALPHA = list(ACCEPTABLE_ALPHA)
INDEX = {' ': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
         'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
         'Y': 25, 'Z': 26}


def single_frequency(text: str) -> dict:
    """Finds the frequencies of single letters in text.

    Args:
        text (str): String text.

    Returns:
        Letter frequencies.
    """
    freq = {}

    if not text:  # check that there is text
        raise EmptyText

    for char in text:  # for each character in text, add to matching dictionary key or create new if it does not exist
        char = char.upper()
        if char in ACCEPTABLE_ALPHA:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1

    return freq


def common_words(text: str) -> list:
    """Finds common word frequencies.

    Args:
        text (str): String text.

    Returns:
        N most common words in text.
    """
    text = text.upper()
    word_count = Counter(text.split())  # use collections counter to count words and create dictionary

    length = sum([v for k, v in word_count.items()])  # use dict comprehension to add all values to get total words

    if length < 175:  # text is too short to use this function
        raise TooShort

    # creates list of words sorted by ascending frequency
    word_count = sorted(word_count, key=lambda z: word_count[z], reverse=True)

    top_words = word_count[:175]  # keep the top 175 words

    return top_words


def bigram(text: str) -> np.ndarray:
    """Finds frequencies of bigrams of letters in a given text.

    Args:
        text (str): String text.

    Returns:
        2D array of bigram frequencies.
    """
    counts = np.zeros((27, 27))  # create array for counts of pairs

    if len(text) < 2:  # if length is not at least two, this function won't be useful
        raise TooShort

    text = text.upper()  # change all to upper case
    i = 1
    while i < len(text):  # go through text to count pairs
        if text[i - 1] in ACCEPTABLE_ALPHA and text[i] in ACCEPTABLE_ALPHA:
            counts[INDEX[text[i-1]], INDEX[text[i]]] += 1  # this increments pair in count matrix using index dictionary
        i += 1

    return counts


def trigram(text: str) -> np.ndarray:
    """Finds frequencies of trigrams of letters in a given text.

    Args:
        text (str): String text.

    Returns:
        3D array of trigram frequencies.
    """
    tri_counts = np.zeros((27, 27, 27))  # create 3d array for counts of trigrams

    if len(text) < 2:  # if length is not at least two, this function won't be useful
        raise TooShort

    text = text.upper()  # change all to upper case
    i = 2
    while i < len(text):  # go through text to count trigrams
        if all(el in ACCEPTABLE_ALPHA for el in [text[i - 2], text[i - 1], text[i]]):
            tri_counts[INDEX[text[i - 2]], INDEX[text[i - 1]], INDEX[text[i]]] += 1
        i += 1

    return tri_counts


def evaluate(decrypt_bigram: np.ndarray, expect_bigram: np.ndarray) -> int:
    """Evaluates the summation of the absolute differences between each element
    in the decrypted bigram matrix and the expected (english training text)
    bigram matrix.

    Args:
        decrypt_bigram (np.ndarray): 2D array of cipher bigram frequencies.
        expect_bigram (np.ndarray): 2D array of training text bigram frequencies.

    Returns:
        Sum of differences of elements in matrices.
    """
    difference = 0  # initialize value for sum of differences in matrix, this will be the difference of the two matrices
    for i in range(expect_bigram.shape[0]):
        for j in range(expect_bigram.shape[1]):
            # for each element in matrix, find difference of decryption and expected and add to sum
            difference += abs(decrypt_bigram[i, j] - expect_bigram[i, j])

    return difference


def evaluate_tri(decrypt_tri: np.ndarray, expect_tri: np.ndarray) -> int:
    """Evaluates the summation of the absolute differences between each element
    in the decrypted trigram array and the expected (english training text)
    trigram array.

    Args:
        decrypt_tri (np.ndarray): 3D array of cipher trigram frequencies.
        expect_tri (np.ndarray): 3D array of training text trigram frequencies.

    Returns:
        Sum of differences of each element in 3D arrays.
    """
    difference = 0  # initialize value for sum of differences in matrix, this will be the difference of the two matrices
    for i in range(expect_tri.shape[0]):
        for j in range(expect_tri.shape[1]):
            for k in range(expect_tri.shape[2]):
                # for each element in matrix, find difference of decryption and expected and add to sum
                difference += abs(decrypt_tri[i, j, k] - expect_tri[i, j, k])

    return difference


def decrypt(curr_key: dict, cryptogram: str) -> str:
    """Decrypts text using a given Crypto Key

    Args:
        curr_key (dict): Current Crypto Key.
        cryptogram (str): Cipher text.

    Returns:
        Decryption of cipher text using current Crypto Key.
    """
    decrypted = ''
    for letter in cryptogram:
        key = get_key(letter, curr_key)
        if key:
            decrypted += key
        else:
            decrypted += letter

    return decrypted


def get_key(val: str, key_dict: dict) -> Optional[str]:
    """Finds the key for a particular dict value.

    Args:
        val (str): Value of desired key.
        key_dict (dict): Current Crypto Key.

    Returns:
        Associated key of given value, if exists else None.
    """
    for key, value in key_dict.items():
        if val == value:
            return key


def initial_key(text_freq: dict, cipher_freq: dict) -> dict:
    """Guess an initial key by matching the frequencies of the training text and the cipher text.
    This makes a list of letters in order from most to least frequent, i.e. [' ', 'E', 'T',...]

    Args:
        text_freq (dict): Single letter frequencies of training text.
        cipher_freq (dict): Single letter frequencies of cipher text.

    Returns:
        Initial key guess.
    """
    sorted_training = sorted(text_freq, key=lambda z: text_freq[z], reverse=True)
    sorted_crypt = sorted(cipher_freq, key=lambda z: cipher_freq[z], reverse=True)
    key_dict = {}
    alpha_temp = ALPHA.copy()
    i, j = 0, 0
    while i < len(sorted_crypt):  # match values from each sorted list and add to key dictionary
        key_dict[sorted_training[i]] = sorted_crypt[i]
        alpha_temp.remove(sorted_crypt[i])
        i += 1
    while i < len(sorted_training):  # for any letters that were not matched, add the remaining alphabet as values
        key_dict[sorted_training[i]] = alpha_temp[j]
        i += 1
        j += 1

    return key_dict


def solve(crypto: str, text_freq: dict, cipher_freq: dict, expected_bigram: np.ndarray, expected_trigram) -> dict:
    """Solves cryptogram quote by finding most fitting Crypto Key.

    Args:
        crypto (str): Cipher text.
        text_freq (dict): Single letter frequencies of training text.
        cipher_freq (dict): Single letter frequencies of cipher text.
        expected_bigram (np.ndarray): Training text bigram frequencies.
        expected_trigram (np.ndarray): Training text trigram frequencies.

    Returns:
        Solved Crypto Key.
    """
    # for list of letters from most to least frequent:
    c_freq_single = sorted(cipher_freq, key=lambda z: cipher_freq[z], reverse=True)

    # make sure cryptogram frequency list includes all letters:
    alpha_temp = ALPHA.copy()
    for element in c_freq_single:
        if element in alpha_temp:
            alpha_temp.remove(element)
    for element in alpha_temp:
        c_freq_single.append(element)

    # algorithm:
    curr_key = initial_key(text_freq, cipher_freq)  # initialize current key with initial key function
    curr_bigram = bigram(decrypt(curr_key, crypto))  # create current bigram from current key applied on crypto
    difference = evaluate(curr_bigram, expected_bigram)  # evaluate current difference of current bigram and expected
    temp_key = curr_key.copy()  # create a temp key so we can edit and attempt to minimize difference value

    a, b = 1, 1
    while True:  # stops when b == 26
        temp_key = swap(c_freq_single[a], c_freq_single[a + b], temp_key)  # use swap to exchange values in key
        a += 1  # increment a each time we try a new temp key. this will increment both swap values since we use a + b
        if a + b > 26:  # if > 26, we want to reset a and increment b by one so that swaps are one more letter apart
            a = 1
            b += 1
        if b == 26:  # if b is 26, we have tried all swaps
            break
        # exchange corresponding rows and columns, i.e. update temp bigram with temp key
        temp_bigram = bigram(decrypt(temp_key, crypto))
        temp_difference = evaluate(temp_bigram, expected_bigram)  # update temp difference
        if temp_difference >= difference:  # compare difference. if it is larger, don't keep the change to the key
            temp_key = curr_key.copy()
            continue

        # if temp difference was smaller than previous difference, keep new key and new difference value and reset
        # a and b
        a, b = 1, 1
        curr_key = temp_key.copy()
        difference = temp_difference

    # reset values
    curr_trigram = trigram(decrypt(curr_key, crypto))  # create current bigram from current key applied on crypto
    difference = evaluate_tri(curr_trigram, expected_trigram)
    temp_key = curr_key.copy()

    # repeat process with trigrams
    a, b = 1, 1
    while True:  # stops when b == 26
        temp_key = swap(c_freq_single[a], c_freq_single[a + b], temp_key)  # use swap to exchange values in key
        a += 1  # increment a each time we try a new temp key. this will increment both swap values since we use a + b
        if a + b > 26:  # if > 26, we want to reset a and increment b by one so that swaps are one more letter apart
            a = 1
            b += 1
        if b == 26:  # if b is 26, we have tried all swaps
            break
        # exchange corresponding rows and columns, i.e. update temp trigram with temp key
        temp_trigram = trigram(decrypt(temp_key, crypto))
        temp_difference = evaluate_tri(temp_trigram, expected_trigram)  # update temp difference
        if temp_difference >= difference:  # compare difference. if it is larger, don't keep the change to the key
            temp_key = curr_key.copy()
            continue

        # if temp difference was smaller than previous difference, keep new key and new difference value and reset
        # a and b
        a, b = 1, 1
        curr_key = temp_key.copy()
        difference = temp_difference

    return curr_key


def swap(a: str, b: str, temp_key: dict) -> dict:
    """Swaps two key's values in the dictionary according to value parameters.
    Args:
        a (str): First value.
        b (str): Second value.
        temp_key (dict): Crypto Key to swap values in.

    Returns:
        Key after swap.
    """
    temp_a = get_key(a, temp_key)
    temp_b = get_key(b, temp_key)
    temp_key[temp_a] = b
    temp_key[temp_b] = a

    return temp_key


def create_puzzle(cipher_file: str) -> Cryptoquote:
    """Create a puzzle using file.

    Args:
        cipher_file (str): Name of file.

    Returns:
        Puzzle as Cryptoquote object.
    """
    with open(cipher_file) as f:
        cipher_text = ' '.join(f.readlines())

    # create puzzle by creating Cryptoquote object with a new key and a quote from the file
    new_key = Key()
    x = new_key.mapping  # access to true key in the form of a dictionary
    quote = Quote(cipher_text)
    crypto = Cryptoquote(quote=quote, key=new_key)

    return crypto


def decrypt_puzzle(crypto: str, train_text: str) -> str:
    """Decrypts a Cryptoquote puzzle.

    Args:
        crypto (str): Encrypted puzzle string.
        train_text (str): Filename for training text.

    Returns:
        The decrypted puzzle string.
    """

    train_freq = single_frequency(train_text)  # get training text single letter frequencies
    crypto_freq = single_frequency(crypto.crypto)  # get cryptogram text single letter frequencies
    train_bigram = bigram(train_text)  # get training text bigram matrix
    train_bigram /= (len(train_text) / len(crypto.crypto))
    train_trigram = trigram(train_text)
    train_trigram /= (len(train_text) / len(crypto.crypto))

    attempt_key = solve(crypto.crypto, train_freq, crypto_freq, train_bigram, train_trigram)  # get attempt of key
    decrypted_puzzle = decrypt(attempt_key, crypto.crypto)  # decrypt the cipher text with the attempted key

    return decrypted_puzzle


def get_training_text(file_name: str) -> str:
    with open(file_name) as f:
        train_text = ' '.join(f.readlines())

    return train_text


if __name__ == '__main__':
    puzzle = create_puzzle('text/quote.txt')
    training_text = get_training_text("text/HarryPotter.txt")
    decrypted = decrypt_puzzle(puzzle, training_text)
    graphs.show_frequency(single_frequency(puzzle.crypto))
    graphs.save_graph('hello', bigram=bigram(training_text))
    print(decrypted)


