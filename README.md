# t21-hack-footprint
Repo for the [TRANSFORM 2021 Hackathon](https://github.com/softwareunderground/transform-2021-hackathon/discussions) project on seismic acquisition footprint noise filtering ([discussion here](https://github.com/softwareunderground/transform-2021-hackathon/discussions/9))

## Slack channel
[t21-hack-footprint](https://app.slack.com/client/T094HBB9T/C01SRMBK50B/thread/C01MZBV60G0-1616097803.005500)

## People
* [@mycarta](https://github.com/mycarta) 
* [@scibbatical](https://github.com/scibbatical)

## Conda setup
```
# To create the conda environment (called "footprint") for this repo, use
#
#     $ conda env create -f environment.yml
#
# To activate the environment, use
#
#     $ conda activate footprint
#
# To deactivate the environment, use
#
#     $ conda deactivate
```

## Description and background
- PowerPoint presentation [here](https://docs.google.com/presentation/d/1xVvdvlpVVrXmgntH651JXqLq-8w0bPWm4E9x2iRnNYk/edit#slide=id.gd23752e0b2_1_0)
- For a definition of acquisition footprint read the [SEG Wiki](https://wiki.seg.org/wiki/Dictionary:Acquisition_footprint_or_imprint)
- For some examples, read this [blog post](https://mycarta.wordpress.com/2017/08/13/what-is-acquisition-footprint-noise-in-seismic-data), and also go to the Resources and reading material below
- Project background: Elwyn and I have done some work putting together a tool to remove acquisition footprint from seismic data; this has been my longest-lived side project. 
- If you are curious about it, please read [Chapter 39](https://github.com/softwareunderground/52things/blob/master/chapters/Niccoli_2.md) of the upcoming [52 Things You Should Know About Geocomputing](https://github.com/softwareunderground/52things) and then head over to the [Tutorial notebook](https://github.com/mycarta/2D_FFT_filter_tutorial/blob/master/52-things_tutorial.ipynb); give it a spin

## Goals
The long term goal (AKA, the "dream") would be to create an open-source tool, ideally part of [Awesome Open Geoscience](https://github.com/softwareunderground/awesome-open-geoscience), that can be used by geophysicists and geologists to remove footprint from seismic data, WHEN 5D interpolation or other costly post-stack processing works are not an options.
The objectives of the hack are:
* The primary objective of this projects would be to find and eliminate performance bottlenecks in the existing code. We already improved computations by switching from `Astropy` convolution to `Scipy` (see [here](https://github.com/mycarta/2D_FFT_filter_tutorial/blob/master/scipy_gaussian_kernel.ipynb) and [here](https://github.com/mycarta/2D_FFT_filter_tutorial/blob/master/speed_up_convolution.ipynb))
* Secondary objective, test with noisier examples (we will need to find open data with footprint, F3 has some)
* Tertiary objective would be to create documentation for the tool, either by expanding/completing the [Tutorial notebook](https://github.com/mycarta/2D_FFT_filter_tutorial/blob/master/52-things_tutorial.ipynb) or a separate document
* Final objective, time permitting, would be to put together a VERY minimal `Panel` app (for example load data from numpy file >> display a time slice and its 2D spectrum >> derive filter >> save filter to numpy file) and deploy
* I am including this for information purposes only: down the road the tool will need a way to automatically recognize and segment time slices of irregular shape into components of polygonal shape (by whatever means)

## Resources and reading material
- [A mathematical review of the Fourier transform - SEG Wiki](https://wiki.seg.org/wiki/A_mathematical_review_of_the_Fourier_transform)
- [An Intuitive Explanation of Fourier Theory](http://apps.usd.edu/coglab/schieber/pdf/Intuitive2DFFT.pdf)
- [The 1-D Fourier transform - SEG Wiki](https://wiki.seg.org/wiki/The_1-D_Fourier_transform)
- [The 2-D Fourier transform - SEG Wiki](https://wiki.seg.org/wiki/The_2-D_Fourier_transform)
- [Elegant SciPy - Chapter 4: Frequency and the Fast Fourier Transform](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html)

## Skills needed
Elwyn and I will be working on this no-matter-what. But we welcome participants that are interested in this project. Useful skills would be:
* Programming / performance optimization /profiling (Numpy broadcasting / Dask, etc.)
* Geophysics / signal analysis
* Testing
* Documentation
* Ideas - any ideas or suggestion is welcome

