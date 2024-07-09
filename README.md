# Image Processing Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Filtering Techniques](#filtering-techniques)

## Introduction
This project focuses on image processing using Python. It includes various techniques for processing and analyzing images to extract useful information or to enhance image quality. The core of this project demonstrates applying different types of filters (low-pass, high-pass, band-pass, and band-stop) to an image using Fourier Transform.

## Features
- Image loading and display
- Fourier Transform and inverse Fourier Transform
- Image filtering (Low-pass, High-pass, Band-pass, Band-stop)
- Visualization of phase spectrum
- Application of custom filters

## Installation
To run this project, you'll need to have Python installed along with the following libraries:
- numpy
- matplotlib
- opencv-python
- scipy

You can install these dependencies using pip:
```bash
pip install numpy matplotlib opencv-python scipy

## Usage
- Load and preprocess the image.
- Apply Fourier Transform to convert the image to the frequency domain.
- Create and apply various filters (low-pass, high-pass, band-pass, band-stop).
- Perform inverse Fourier Transform to convert the filtered image back to the spatial domain.
- Visualize the results.
- Filtering Techniques
This project includes the implementation of various filtering techniques using Fourier Transform. Below are the key functions and their descriptions:

- low_pass_filter(shape, cutoff):
Creates a low-pass filter mask.

- high_pass_filter(shape, cutoff):
Creates a high-pass filter mask.

- band_pass_filter(shape, low_cutoff, high_cutoff):
Creates a band-pass filter mask.

- band_stop_filter(shape, low_cutoff, high_cutoff):
Creates a band-stop filter mask.

- apply_filter(image, filter_mask):
Applies the given filter mask to the image.
