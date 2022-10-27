#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Custom Image Dataset

Course/Term: Neural Networks and Deep Learning (COMP9444) 2022T3
Convener/Lecturer: Alain Blair (blair@cse.unsw.edu.au)
Task: Group Project
Author: Daniel Gotilla (z5343046@unsw.edu.au)

Objective:
Implement a Custom DataSet object to allow our PyTorch network to access data.

Based on PyTorch Docs Tutorial: https://gotil.la/3gp1ZyN
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    """
    Custom Image Dataset Class
    """
    def __init__(self, annotations_file: str, img_dir: str, transform=None):
        """
        Create a Custom Image Dataset object

        Usage:
        dataset_obj = CustomImageDataset('landmarks.txt', '../images/')

        :param annotations_file: address of input file (relative to script)
        :param img_dir: path to image directory (relative to script)
        :param transform: function to be applied to every image requested
        """

        # Read the landmarks file for later querying
        self.img_labels = pd.read_csv(annotations_file,
                                      sep=r'\s+',
                                      header=None)
        self.img_dir: str = img_dir
        self.transform = transform

        # Initialise Maps for the __getitem__ method
        self.genders: dict[str: str] = {
            '0': "male",
            '1': "female"
        }
        self.races: dict[str: str] = {
            '0': "white",
            '1': "black",
            '2': "asian",
            '3': "indian",
            '4': "other"
        }

    def __len__(self) -> int:
        """
        Returns the number of images in the Custom Image Dataset object.

        Usage:
        len(dataset_obj)

        :return: int
        """
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Used by PyTorch to request a given image within the Dataset.

        Usage:
        dataset_obj[42]

        :param idx: number of requested image (should be less than __len__)
        :return: a dictionary with the image and its respective metadata for
        the requested index; it uses the following keys:
            'image': PyTorch image obj for the image file
            'age': int with the age of the person in the image
            'gender': str ('male' or 'female')
            'race': str ('white', 'black', 'asian', 'indian', 'others')
            'datetime': int (YYYYMMDDHHmm)
            'landamarks': Pandas Dataframe obj with 68 pairs of x,y coords
        """

        # Reads file for given index as a tensor image
        imagename = self.img_labels.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, imagename + ".chip.jpg")).float()

        # Applies any transformations to image
        if self.transform:
            image = self.transform(image)

        # Reads the image metadata from the filename
        splitname: list[str] = imagename.split(".")[0].split("_")
        age: int = int(splitname[0])
        gender: str = self.genders[splitname[1]]
        race: str = self.races[splitname[2]]
        datetime: int = int(splitname[3][:13])

        # Reorganises the x,y landmark coordinates as a 68x2 dataframe
        coords = self.img_labels.iloc[idx, 1:].tolist()
        raw_landmarks: list[tuple[int, int]] = []
        for i in range(0, len(coords), 2):
            raw_landmarks.append((coords[i], coords[i+1]))
        landmarks = pd.DataFrame(raw_landmarks, columns=['x', 'y'])

        # Returns dict with requested image and its metadata
        return {'image': image,
                'age': age,
                'gender': gender,
                'race': race,
                'datetime': datetime,
                'landmarks': landmarks}


if __name__ == "__main__":
    # Initialise a Custom Image Dataset object with the consolidated
    # list of landamrks file and the image directory
    UTKFace = CustomImageDataset('landmark_list.txt', 'UTKFace')

    # Retrieve the first image in the Dataset
    image0 = UTKFace[0]

    # Print out the values for the returned dictionary
    print(image0)