# RGB to/from Spectral Colorspace

Largley based on ideas from Scott Allen Burns

http://scottburns.us/color-science-projects/

Generates 3 spectral reflectance distributions in range > 0.0 and < 1.0, along with matrixes to convert to/from RGB. Also produces a set of 10 Munsell colors from 5R thru 5RP


```
python3 -m pip install git+https://github.com/uqfoundation/mystic
python3 -m pip install git+https://github.com/colour-science/colour@develop
python3 -m pip install matplotlib
```

edit settings.py to adjust colorspace and parameters (if necessary).  The default is sRGB and 8 spectral channels.  Look in tools.py to see how you can convert to/from and mix pigment colors in log.

run the solver

```
python3 main.py
```
