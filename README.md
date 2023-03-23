# CSC490H1: Making Your Self-driving Car Perceive the World

This repository contains the starter code for CSC490H1: We then filled functions required to detect cars from LIDAR Data, predict its trajectory five seconds into the future, and collect appropriate metrics and visualizations.
Making Your Self-driving Car Perceive the World.

## Getting started

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ```bash
   curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' > Miniconda.sh
   bash Miniconda.sh
   rm Miniconda.sh
   ```

2. Close and re-open your terminal session.

3. Change directories (`cd`) to where you cloned this repository.

4. Create a new conda environment:

   ```bash
   conda env create --file environment.yml
   ```

   To update the conda environment with new packages:

   ```bash
   conda env update --file environment.yml
   ```

5. Activate your new environment:

   ```bash
   conda activate csc490
   ```

6. Download [PandaSet](https://scale.com/resources/download/pandaset).
   After submitting your request to download the dataset, you will receive an
   email from Scale AI with instructions to download PandaSet in three parts.
   Download Part 1 only. After you have downloaded `pandaset_0.zip`,
   unzip the dataset as follows:

   ```bash
   unzip pandaset_0.zip -d <your_path_to_dataset>
   ```

6. To switch between the Gaussian and Regular Prediction model:
      1. Open /prediction/main.py
      2. In line 22, set MODEL=0 for regular, and MODEL=1 for Gaussian.
