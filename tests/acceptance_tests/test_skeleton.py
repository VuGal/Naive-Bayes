# -*- coding: utf-8 -*-

import pytest
from naive_bayes.skeleton import fib

__author__ = "Wojciech Gałecki, Karolina Rapacz"
__copyright__ = "Wojciech Gałecki, Karolina Rapacz"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
