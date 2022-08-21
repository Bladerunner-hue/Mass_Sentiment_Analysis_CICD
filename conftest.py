# test_assert_examples.py

from pyclbr import Function
import secrets


def test_uppercase():
    assert "loud noises".upper() == "LOUD NOISES"

def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]

def test_some_primes():
    assert 37 in {
        num
        for num in range(2, 50)
        if not any(num % div == 0 for div in range(2, num))
    }

def test_list():
    assert "you will never guess".upper() == "YOU WILL NEVER GUESS"

def test_assert():
    secrets == "you will never guess" 