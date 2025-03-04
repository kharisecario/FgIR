# gsml

The program workflow are as follow:

## 1. JDX scrapping from NIST

*jdxscrapping*
All the molecules that are in gas phase and has IR spectra are obtained.
> Output: Directory for all IR obtained in JDX scrapping.

## 2. Collect only absorption data

*collectabs*
From all collected IR spectra, it further selected to be the only one with absorption spectra.
> Output: Directory with all copied JDX file for absorbance IR. Filename is in INCHI.

## 3. Read needed information from the JCamp file and interpolate data

*readjcamp* (run automatically with *collectpoint*)
Information such as wavelength, intensity, molecule name, CAS number, INCHI, etc.
> Output: *moldataset.csv* for all the list extracted information.

*collectpoint*
All the intensity vs wavelength data are collected. Wavelength is interpolated by cubic spline interpolation to have uniform number of data and range.
> Output: *XYfeature.csv* and *INCHIlabel.txt* 

## 4. Identify functional group presence with RDkit

*obtainfuncgrouplabel*
From collected INCHI, it put into RDkit to determine its molecule and functional groups that are in the molecules. It collected both as presence and count (how many it have certain functional groups)
> Output: *label_fg_count.csv* and *label_fb_presense.csv*

## 5. Normalize the XYfeature

*getxynorm*
Here, Standard Normal Variate (SNV) are used. It is found to help IR spectra more easily identified. Several other methods are prepared (min-max, min-max+SNV).
> Output: *XYsnv*, *XYnorm*, and *XYsnvnorm*.

## 6. Randomforest

Select label that want to be removed are done here.

*ftirrandomforest*
Used SKlearn and run in CPU. For default parameters, each molecules run for 1-2 minutes. (Single sample)

*randfor_gpu_single*
Used CuML (and SKlearn) and run in GPU. For default parameters, each molecules run for 1-2 s. (Single sample)

*randfor_gpu*
Used CuML (and SKlearn) and run in GPU. Each molecules run for 1-2 times. It loops all the sample.
