# Total-Variation
Total variation image denosing and deblurring.

Python implementation of Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems by Amir Beck and Amir Teboulle (http://www.math.tau.ac.il/~teboulle/papers/tlv.pdf). Link for code on paper no longer works; use the TV_FISTA download from https://sites.google.com/site/amirbeck314/software (this link also has another download for the paper).

Generated the two mat files by following these two parameter sets from guide_tv_deblur.pdf (included in the code download linked above):
- cameraman_Bobs.mat: 
    ```Matlab
    X = double(imread('cameraman.pgm'));
    X = X/255;
    randn('seed',314);
    Bobs=X+2e-2*randn(size(X))
    ```
    Used for denoising task.
- cameraman_Bobs_blurry.mat:
    ```Matlab
    X = double(imread('cameraman.pgm'));
    X = X/255;
    P=1/9*ones(3,3);
    center=[2,2];
    randn(’seed’,314);
    Bobs=imfilter(X,P,’symmetric’)+1e-4*randn(size(X));
    ```
    Used for deblurring/denoising task.
    Note: Due to difference in indexing between Matlab and Python, the center in Python will be `center - [1 1]`. So, in the above example, in Python set center to [1 1]

environment.yaml contains the yaml file to create the conda environment called total-var. Run `conda env create -f environment.yaml`.