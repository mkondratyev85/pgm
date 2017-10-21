import unittest

from largetime import Time

class TestTime(unittest.TestCase):

    def test_parcing(self):
        t = Time()
        self.assertTrue(t._sec == 0)

        t = Time(100)
        self.assertTrue(t._sec == 100)

        t = Time('100')
        self.assertTrue(t._sec == 100)

        t = Time(' 100 ')
        self.assertTrue(t._sec == 100)

        t = Time(' 100 min ')
        self.assertEqual(t._sec, 100*60)

        self.assertRaises(ValueError,Time,' 100min ')

        self.assertRaises(ValueError,Time,' 100 min hr')

        self.assertRaises(ValueError,Time,' 100 dsin ')

        t = Time(' 100 hr ')
        self.assertTrue(t._sec == 100*60*60)

        t = Time(' 100 days ')
        self.assertTrue(t._sec == 100*60*60*24)

        t = Time(' 100 months ')
        self.assertTrue(t._sec == 100*60*60*24*30.5)

        t = Time(' 100 yr ')
        self.assertTrue(t._sec == 100*60*60*24*365.25)

        t = Time(' 100 Kyr ')
        self.assertTrue(t._sec == 100*60*60*24*365.25*1_000)

        t = Time(' 100 Myr ')
        self.assertTrue(t._sec == 100*60*60*24*365.25*1_000_000)

    def test_adding(self):
        t = Time()
        self.assertEqual(t, 0)
        t2 = Time()
        t3 = Time(1)
        self.assertEqual(t, t2)
        self.assertNotEqual(t, t3)
        self.assertNotEqual(t, 3)

        self.assertEqual(t+1, t3)

        self.assertTrue(t3 < 2)
        self.assertFalse(t3 < -1)
