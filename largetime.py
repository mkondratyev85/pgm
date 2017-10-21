sec = lambda t: t
min = lambda t: 60*t
hr = lambda t: 60*min(t)
days = lambda t: 24*hr(t)
months = lambda t: 30.5*days(t)
yr = lambda t: 365.25 * days(t)
Kyr = lambda t: 1_000 * yr(t)
Myr = lambda t: 1_000_000 * yr(t)

class Time(object):

    _units = { 'sec' : sec,
               'min' : min,
               'hr' : hr,
               'days' : days,
               'months' : months,
               'yr' : yr,
               'Kyr' : Kyr,
               'Myr' : Myr,
               }

    def __init__(self, t=0):
        if type(t) == str:
            t = t.strip()
            t2 = t.split(' ')
            if len(t2) == 1:
                t = t2[0]
            elif len(t2) > 2:
                raise ValueError
            elif t2[1] not in self._units:
                raise ValueError
            else:
                f = self._units[t2[1]]
                t = f(float(t2[0]))
        self._sec = float(t)

    def __bool__(self):
        return True

    def __lt__ (self, other):
        if type(other) == Time:
            return self._sec < other._sec
        else:
            return self._sec < float(other)

    def __eq__ (self, other):
        if type(other) == Time:
            return self._sec == other._sec
        else:
            return self._sec == float(other)

    def __add__(self, other):
        if type(other) == Time:
            return Time(self._sec + other._sec)
        else:
            return Time(self._sec + float(other))

    def __str__(self):
        for unit in self._units:
            f = self._units[unit]
            t = self._sec /f(1)
            if len(str(int(t)))>3:
                continue
            break
        return f"{t:4.7} {unit}"
