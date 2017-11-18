PGM
===

**PGM** - is a tool for simple geodynamical 2D modelling written in Python and
based mostly on a book "Introduction to Numerical Geodynamic Modelling" by T.Gerya.
It uses combination of Finit difference method and marker in cell techique
to solve set of momentum and continuity equations.

To define a problem you can use image in *png* format where manually assign
materials to each color of the image by using **model_constructor** Gui tool.


Requirements
------------

To successfully run PGM you need Python 3.6 with numpy, scipy and matplotlib installed.
PGM was tested on Ubuntu 16.06 and 17.10
To see a full list of requirements see requirements.txt


Installation
------------

### Ubuntu:

First, make sure you use Python 3.6 by running:

'''
    python -V
'''

Clone repository:

'''
    git clone https://github.com/mkondratyev85/pgm
    cd pgm
'''

Install requirements:

'''
    pip3 install -r requirements.txt
'''


### Mac:

Not tested


### Win:

Not tested


Usage
-----


To run benchmark use:

'''
    python3.6 pgm.py benchmarks/slab.py /tmp/slab
'''

That will run modelling that will store each step of modelling in /tmp/slab.
