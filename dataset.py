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
from os import path
from torch import tensor, div
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
        with open(annotations_file, 'r') as file:
            lines = file.readlines()
        self.img_labels = [line.split() for line in lines]
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

    def __getitem__(self, idx: int):
        """
        Used by PyTorch to request a given image within the Dataset.

        Usage:
        dataset_obj[42]

        :param idx: number of requested image (should be less than __len__)
        :return: the requested image and its metadata as separate variables:
            'image': Scaled PyTorch tensor obj for the image file
            'age': int with the age of the person in the image
            'gender': str ('male' or 'female')
            'race': str ('white', 'black', 'asian', 'indian', 'others')
            'landamarks': PyTorch tensor obj with 68 pairs of x,y coords
        """

        # Reads file for given index as a tensor image
        imagename = self.img_labels[idx][0] + ".chip.jpg"
        image = read_image(path.join(self.img_dir, imagename)).float()
        image_scaled = div(image, 255)

        # Applies any transformations to image
        if self.transform:
            image_scaled = self.transform(image_scaled)

        # Reads the image metadata from the filename
        splitname: list[str] = imagename.split(".")[0].split("_")
        age: int = int(splitname[0])
        gender: str = self.genders[splitname[1]]
        race: str = self.races[splitname[2]]
        # datetime: int = int(splitname[3][:13])    // Not used

        # Reorganises the x,y landmark coordinates as a 68x2 tensor
        coords = self.img_labels[idx][1:]
        raw_landmarks: list[list[int, int]] = []
        for i in range(0, len(coords), 2):
            raw_landmarks.append([int(coords[i]), int(coords[i + 1])])
        landmarks = tensor(raw_landmarks)

        return image_scaled, age, gender, race, landmarks


if __name__ == "__main__":
    # Initialise a Custom Image Dataset object with the consolidated
    # list of landamrks file and the image directory
    UTKFace = CustomImageDataset('landmark_list.txt', 'UTKFace')

    # Retrieve the first image in the Dataset
    image0 = UTKFace[0]

    # Print out the values for the returned dictionary
    print(image0)