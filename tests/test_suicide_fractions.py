#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.suicide_fractions import main, suicide_fractions


def _spy(method_to_decorate):
    """Wrap a real method with a MagicMock so calls can be asserted while
    the original implementation still runs."""
    mock = MagicMock(name="groupby method")

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


class TestSuicideFractions(unittest.TestCase):

    def setUp(self):
        self.s = suicide_fractions()

    def test_shape(self):
        self.assertEqual(
            self.s.shape,
            (141,),
            msg="suicide_fractions() returned a Series of shape %r, "
            "expected (141,) - one mean fraction per country."
            % (self.s.shape,),
        )

    def test_type(self):
        self.assertIsInstance(
            self.s,
            pd.Series,
            msg="suicide_fractions() must return a pandas Series, not a "
            "DataFrame or other object. Got %r." % (type(self.s),),
        )
        self.assertEqual(
            self.s.dtype,
            float,
            msg="The dtype of the returned Series should be float, got %r."
            % (self.s.dtype,),
        )

    def test_index(self):
        ind = ["Albania", "Anguilla", "Antigua and Barbuda", "Argentina", "Armenia"]
        self.assertCountEqual(
            self.s.index[:5],
            ind,
            msg="The first five countries in the index should be %r, got %r."
            % (ind, list(self.s.index[:5])),
        )

    def test_nulls(self):
        nulls = self.s.isnull().sum()
        self.assertEqual(
            nulls,
            23,
            msg="Expected 23 missing values (countries with no suicide-rate "
            "data) in the Series, got %d." % (nulls,),
        )

    def test_content(self):
        self.assertAlmostEqual(
            self.s["Albania"],
            0.000035,
            places=6,
            msg="Mean suicide fraction for Albania should be about 0.000035, "
            "got %r." % (self.s["Albania"],),
        )
        self.assertAlmostEqual(
            self.s["Belgium"],
            0.000222,
            places=6,
            msg="Mean suicide fraction for Belgium should be about 0.000222, "
            "got %r." % (self.s["Belgium"],),
        )
        self.assertAlmostEqual(
            self.s["Finland"],
            0.000228,
            places=6,
            msg="Mean suicide fraction for Finland should be about 0.000228, "
            "got %r." % (self.s["Finland"],),
        )

    def test_calls(self):
        method = _spy(pd.core.frame.DataFrame.groupby)
        with patch(
            "src.suicide_fractions.suicide_fractions", wraps=suicide_fractions
        ) as psf, patch.object(
            pd.core.frame.DataFrame, "groupby", new=method
        ), patch(
            "src.suicide_fractions.pd.read_csv", wraps=pd.read_csv
        ) as prc:
            main()
            psf.assert_called_once_with()
            prc.assert_called_once()
            method.mock.assert_called_once()
            args, kwargs = method.mock.call_args
            correct = (len(args) > 0 and args[0] == "country") or (
                "by" in kwargs and kwargs["by"] == "country"
            )
            self.assertTrue(
                correct,
                msg="groupby must be called with 'country' (either as the "
                "first positional argument or as by='country'). Got "
                "args=%r kwargs=%r." % (args, kwargs),
            )


if __name__ == '__main__':
    unittest.main()
