# Cancer Histopathology Anomaly Detection: CNN on unsupervised VQ-VAE Reconstruction

In this repository, I will be performing classification of the presence of cancer on histopathological slides through anomaly detection. This will be achieved by first training a VIT VQ-VAE on healthy tissue in an unsupervised manner. The assumption here is that the VQ-VAE will learn how to reconstruct healthy tissue, but will struggle with never-before-seen samples of cancerous cells. 

The reconstruction error will thus be a proxy of how much a sample looks like the distribution that the VQ-VAE was trained on, which was only healthy tissue. This reconstruction error could be directly taken as the likelihood that a sample is cancerous, but a better approach is to use the pixel-wise reconstruction error to train a new CNN in a supervised way.

I will compare 3 different inputs that can be used for the final supervised CNN:
1. Using the pixel-wise reconstruction error-map as the sole input.
2. Using both the original image and the reconstruction, as input. (2 channels)
3. Using the original image, the reconstruction and the pixel-wise error as the input (3 channels)
   
*Note: for the amount of channels I'm assuming an input image with a single channel*
