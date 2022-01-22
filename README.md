# RGB to/from Spectral Colorspace

Largley based on ideas from Scott Allen Burns

http://scottburns.us/color-science-projects/

Generates 3 spectral reflectance distributions in range > 0.0 and < 1.0, along with matrixes to convert to/from RGB. Also produces a set of 10 Munsell colors from 5R thru 5RP


```
python3 -m pip install git+https://github.com/colour-science/colour@develop
python3 -m pip install matplotlib
python3 -m pip install scipy
```

edit settings.py to adjust colorspace and parameters (if necessary).  The default is DisplayP3 and 12 spectral channels.  Look in tools.py to see how you can convert to/from and mix pigment colors in log.

run the solver

```
python3 main.py
```

Output will spit out the primary spectral distrubtions and some matrices.  The `spectral_to_XYZ_m` is the only one you really need, you can combine with any XYZ to RGB matrix to get to RGB from there. `spectral_to_RGB_m` will get you back to the original color space, while `Spectral_to_Device_RGB_m` is intended for weird scenarios like iPad where you need to use sRGB primaries even if your colorspace was originally DisplayP3.

```
differential_evolution step 478: f(x)= 2.32856
cost metric, weighted delta cost, actual delta value
red delta:        0.0497544804228 0.00705368559144
green delta:      0.0295454927939 0.00543557658339
blue delta:       0.114556104878 0.0107030885672
illum xy delta:   0.0144509163821 0.00380143609471
bumpiness:        0.00351695069735 3.51695069735
wave variance     0.0 0.0
illum shape diff  0.038609011688 386.09011688
illum bumpiness   0.0806991056501 80.6991056501
lum drop rg       0.0864492314365 0.294022501582
lum drop rb       1.03911693386 0.101937085198
lum drop gb       0.0451388565592 0.212459070315
mix green delta:  0.260163418226 0.161295820847
mix bl/wh delta:  0.365444285811 0.191165971295
mix purple delta:  0.201112616365 0.141814179956
`touch halt` to exit early with this solution.
---
differential_evolution step 479: f(x)= 2.32856
cost metric, weighted delta cost, actual delta value
red delta:        0.0497544804228 0.00705368559144
green delta:      0.0295454927939 0.00543557658339
blue delta:       0.114556104878 0.0107030885672
illum xy delta:   0.0144509163821 0.00380143609471
bumpiness:        0.00351695069735 3.51695069735
wave variance     0.0 0.0
illum shape diff  0.038609011688 386.09011688
illum bumpiness   0.0806991056501 80.6991056501
lum drop rg       0.0864492314365 0.294022501582
lum drop rb       1.03911693386 0.101937085198
lum drop gb       0.0451388565592 0.212459070315
mix green delta:  0.260163418226 0.161295820847
mix bl/wh delta:  0.365444285811 0.191165971295
mix purple delta:  0.201112616365 0.141814179956
`touch halt` to exit early with this solution.
---

original XYZ targets:  [array([0.412444605778738, 0.212717741970923, 0.019437791408796]), array([0.357643626542645, 0.715197161899879, 0.119291766091722]), array([0.180557785915699, 0.072285096129198, 0.950546004809512])]
final XYZ results: [0.411653956837962 0.214051160858206 0.019937672179341] [0.358549073445655 0.713950369077184 0.118765350156111] [0.183746287138646 0.075267324938449 0.951310403978453] [87.849601534623886 99.597360042767519 118.354854233593628]
optimal (maybe) wavelengths: [462.234773543109952, 513.060176647041089, 565.452706695094548,
 617.968872719492197, 702.024717054742609, 716.710927736806184,
 723.207695305741822, 729.991155510753174]
Spectral red is
[0.016839219848991, 0.056297605942981, 0.034211916841505,
 0.996855372514724, 0.999990000050000, 0.999952101050293,
 0.970044712764112, 0.999990000050000]
Spectral green is
[0.051925270155586, 0.999883682663797, 0.784699164495622,
 0.179017334640088, 0.999990000050000, 0.963949717360944,
 0.941471776849435, 0.957484656296654]
Spectral blue is
[0.999990000050000, 0.069412518874038, 0.018702055417155,
 0.027715468817236, 0.793292459419939, 0.881241745509140,
 0.750431552708948, 0.999990000050000]
spectral_to_XYZ_m is
[[0.160911702209152, 0.010536082264362, 0.335863234341330,
  0.392091779907267, 0.003577037160754, 0.001199280922560,
  0.000744951625635, 0.000502994513244],
 [0.038578653590621, 0.302725534072098, 0.478125520092737,
  0.178394820214319, 0.001291733514223, 0.000433082262815,
  0.000269015687274, 0.000181640565914],
 [0.946532443606047, 0.068580748489464, 0.001308663025418,
  0.000093394132679, 0.000000000000000, 0.000000000000000,
  0.000000000000000, 0.000000000000000]]
spectral_to_RGB_m is
[[-0.009751444231376, -0.465453116724931, 0.352808002267490,
  0.996449910264658, 0.009607180544409, 0.003221020036560,
  0.002000785634738, 0.001350938947967],
 [-0.044257132949613, 0.560541110050266, 0.571469016356550,
  -0.045365696355241, -0.001043770411510, -0.000349947151784,
  -0.000217374935762, -0.000146772432437],
 [1.001540204578923, 0.011324986856812, -0.077459271522419,
  -0.014477621006246, -0.000064493011594, -0.000021622709617,
  -0.000013431283479, -0.000009068865387]]
Spectral_to_Device_RGB_m is
[[-0.009751444231376, -0.465453116724931, 0.352808002267490,
  0.996449910264658, 0.009607180544409, 0.003221020036560,
  0.002000785634738, 0.001350938947967],
 [-0.044257132949613, 0.560541110050266, 0.571469016356550,
  -0.045365696355241, -0.001043770411510, -0.000349947151784,
  -0.000217374935762, -0.000146772432437],
 [1.001540204578923, 0.011324986856812, -0.077459271522419,
  -0.014477621006246, -0.000064493011594, -0.000021622709617,
  -0.000013431283479, -0.000009068865387]]
  ```
  
  
It will also plot color mixes.  The columular output order is  linear sRGB, weighted geometric mean spectral, then perceptual sRGB:
  
![Figure_1](https://user-images.githubusercontent.com/6015639/150627303-476c9959-cd6c-4a2e-8090-1374c4e2859c.png)
![Figure_2](https://user-images.githubusercontent.com/6015639/150627306-16b4896f-90cc-492c-aba1-5eda4e7dc500.png)
![Figure_3](https://user-images.githubusercontent.com/6015639/150627309-50f58cd4-7323-42cf-9755-07512b7e877e.png)
![Figure_4](https://user-images.githubusercontent.com/6015639/150627329-3d1cecc1-c058-4f0c-aeba-4624d415f4fe.png)
![Figure_5](https://user-images.githubusercontent.com/6015639/150627334-6800c07e-9640-4a5a-a967-75957459084b.png)
![Figure_6](https://user-images.githubusercontent.com/6015639/150627339-3843a048-8bd5-462c-a56d-6c51d6683684.png)
![Figure_7](https://user-images.githubusercontent.com/6015639/150627343-be564c25-b760-4b2e-9974-3e9907fc82ec.png)
![Figure_8](https://user-images.githubusercontent.com/6015639/150627346-b768b1ac-15e8-4499-86be-eeb032909e8c.png)
![Figure_9](https://user-images.githubusercontent.com/6015639/150627349-3ae7dac9-567d-4518-be19-cff3d7a47fc6.png)
![Figure_10](https://user-images.githubusercontent.com/6015639/150627353-faace08e-90b7-42f3-ab27-2c00d48f8f04.png)
![Figure_11](https://user-images.githubusercontent.com/6015639/150627358-800a449d-e480-4f0f-90c3-8d7b2954d9b1.png)
![Figure_12](https://user-images.githubusercontent.com/6015639/150627366-59345b5a-b69b-4ba6-b7d9-ef49cd23011c.png)
