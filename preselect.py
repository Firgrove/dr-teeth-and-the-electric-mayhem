#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Landmark Preselection

Course/Term: Neural Networks and Deep Learning (COMP9444) 2022T3
Convener/Lecturer: Alain Blair (blair@cse.unsw.edu.au)
Task: Group Project
Author: Daniel Gotilla (z5343046@unsw.edu.au)

Objective:
Function to generate a landmark file with a subset of the images according
to given filters.
"""
from sys import exit


def preselect_landmarks(landmarks_file: str, *, log=False,
                        age=None, gender=None, race=None) -> None:
    """ Preselect Landmarks

    Iterates through an original file listing images and associated landmarks
    applying the given filters and generates another file with the subset of
    images that passed *all* filters.

    :param landmarks_file: name of original landmark file (str, Required)
    :param log: Whether to log why each image was discarded and to print a
        summary message with the number of images filtered (Default: False)
    :param age: tuple containing min (int) and max (int) values (Default: None)
    :param gender: 'male' or 'female' (str, Default: None)
    :param race: either a str with a single race or a list for multiple races,
        values 'white', 'black', 'asian', 'indian' and 'other' (Default: None)
    :return: nothing, may print Errors, Warnings and log messages to stdout
    """
    with open(landmarks_file, 'r') as file:
        landmarks = file.readlines()

    # Initialise maps
    genders: dict[str: str] = {
        '0': "male",
        '1': "female"
    }
    races: dict[str: str] = {
        '0': "white",
        '1': "black",
        '2': "asian",
        '3': "indian",
        '4': "other"
    }

    # Read parameters for valid filters and capture those for new filename
    filters = []
    if isinstance(age, tuple) and age[0] <= age[1]:
        filters.append(str(age[0]) + "-" + str(age[1]))
    if isinstance(gender, str):
        filters.append(gender)
    if isinstance(race, str):
        race = [race]
    if isinstance(race, list):
        filters.append("-".join(race))

    # Abort if no valid filters were found
    if len(filters) == 0:
        print("Error: No valid filters to apply.")
        exit(1)

    filtered_landmarks: list[str] = []
    for line in landmarks:
        # Iterate over all lines in landmark file and apply filters
        keep = True

        # Retrieve metadata from filename
        imagename = line.split()[0]
        splitname: list[str] = imagename.split(".")[0].split("_")
        # 0 is presumed if no age is provided in metadata, so this may fail
        # minimum age filters greater than 0.
        line_age = int(splitname[0]) if len(splitname) > 0 else 0
        # An image with no gender metadata will fail all gender filters
        if len(splitname) > 1 and splitname[1] in genders:
            line_gender = genders[splitname[1]]
        else:
            line_gender = ""
        # An image without race metadata will fail any race filters
        if len(splitname) > 2 and splitname[2] in races:
            line_race = races[splitname[2]]
        else:
            line_race = ""

        # Check if a given line passes *all* filters
        if isinstance(age, tuple) and (line_age < age[0] or line_age > age[1]):
            if log:
                print(f"Image {imagename} skipped due to age ({line_age}).", )
            keep = False
        if isinstance(gender, str) and line_gender != gender:
            if log:
                print(f"Image {imagename} skipped due to gender ({line_gender}).", )
            keep = False
        if isinstance(race, list) and line_race not in race:
            if log:
                print(f"Image {imagename} skipped due to race ({line_race}).", )
            keep = False

        if keep:
            filtered_landmarks.append(line)

    if len(filtered_landmarks) == 0:
        print("Warning: No images passed all filters.")
        exit(1)

    filtered_landmarks_file = landmarks_file.split(".")[0]
    filtered_landmarks_file += "_" + "_".join(filters) + ".txt"
    with open(filtered_landmarks_file, 'w') as file:
        file.writelines(filtered_landmarks, )
    if log:
        print(len(filtered_landmarks),
              "filtered images saved to file",
              filtered_landmarks_file)


if __name__ == "__main__":
    # You can call the preselect_landmarks function with a sigle filter…
    preselect_landmarks('landmark_list.txt', gender='female')

    # …or many filters. Race can be a single string…
    # preselect_landmarks('landmark_list.txt', age=(0, 50), race='asian')

    # …or a list with many strings.
    # preselect_landmarks('landmark_list.txt', race=['black', 'other'])

    # There's very little checking of parameters so you may want to enable
    # logging to find out why each image was discarded. Note that some files
    # do not have all metadata fields so they will be excluded if any such
    # filters are enabled. All images with fail a non-existant filter value.
    # preselect_landmarks('landmark_list.txt', race='caucasian', log=True)
