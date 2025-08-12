## Automated Plate Detection using UNET Model

## About the project

This project implements a system of semantic segmentation using **PyTorch**, based in the **UNET** model to detect automatically patents of vehicles in images.
The objective of the software is process images and retrieve a **binary mask** that show the ubication of the plate.
<p align="center">
  <img src="https://github.com/aguusandy/plate_detection_ai_unet/blob/master/imgs/unet.png" alt="UNET Architecture" width="600"/>
</p>


##  Features
- **Framework**: PyTorch
- **Model**: UNet
- **Images Resize**: 224x224 pixels
- **Size of batches**: 64 images
- **Dataset size**: 3977 images: **Train size**: 80%,  **Validation size**: 10%, **Test size**: 10%


### The software were implemented and tested in:
<ul>
  <li>Python 3.12.4</li>
  <li>PyTorch 2.7.1</li>
</ul>

<p align="center">
  <img src="https://github.com/aguusandy/plate_detection_ai_unet/blob/master/imgs/test_img.png" alt="Mask detected" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/aguusandy/plate_detection_ai_unet/blob/master/imgs/result.png" alt="Result" width="600"/>
</p>
