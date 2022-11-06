#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Landmark Preselection

Course/Term: Neural Networks and Deep Learning (COMP9444) 2022T3
Convener/Lecturer: Alain Blair (blair@cse.unsw.edu.au)
Task: Group Project
Author: Daniel Gotilla (z5343046@unsw.edu.au)

Objective:
Function to generate a landmark file with a subset of the images according
to given filters and, optionally, up to a given target number.
"""
from sys import exit
from os.path import exists
from random import shuffle, seed


def preselect_landmarks(landmarks_file: str, age=None, gender=None, race=None,
                        *, log: bool = False, randomise: bool = False,
                        randomseed=None, target=None) -> None:
    """ Preselect Landmarks

    Iterates through an original file listing images and associated landmarks
    applying the given filters and generates another file with the subset of
    images that passed *all* filters.

    :param landmarks_file: name of original landmark file (str, Required)
    :param log: Whether to log why each image was discarded and to print a
        summary message with the number of images filtered (Default: False)
    :param target: (maximum) number of images to select based on the provided
        filters. If defined, the function may output two files with suffixes:
        • "_filtered": list of images which meet all filter criteria;
        • "_remainder": list of images which do not meet filter criteria or
            exceed target number;
    :param randomise: shuffle landmarks before preselection? (Default: False)
    :param randomseed: seed value (int) when randomising (Default: None)
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
    if isinstance(gender, str) and gender in genders.values():
        filters.append(gender)
    if isinstance(race, str):
        race = [race]
    if isinstance(race, list) and all(r in races.values() for r in race):
        filters.append("-".join(race))

    # Abort if no valid filters were found or invalid target
    if len(filters) == 0:
        print("Error: No valid filters to apply.")
        exit(1)
    if target is not None and (not isinstance(target, int) or target < 1):
        print("Error: Target needs to be greater than zero.")
        exit(1)

    # Abort if files already exist with target name (avoid overwriting).
    filtered_landmarks_file = landmarks_file.split(".")[0]
    filtered_landmarks_file += "_" + "_".join(filters)
    filtered_landmarks_file += "_filtered" if target else ""
    filtered_landmarks_file += ".txt"
    if exists(filtered_landmarks_file):
        print(f"Error: File '{filtered_landmarks_file}' already exists in "
              f"current directory; delete or rename and run script again.")
        exit(1)
    remainder_landmarks_file = landmarks_file.split(".")[0]
    remainder_landmarks_file += "_" + "_".join(filters) + "_remainder.txt"
    if target and exists(remainder_landmarks_file):
        print(
            f"Error: File '{remainder_landmarks_file}' already exists in "
            f"current directory; delete or rename and run script again.")
        exit(1)

    if randomise:
        if seed is not None:
            seed(randomseed)
        shuffle(landmarks)

    filtered_landmarks: list[str] = []
    remainder_landmarks: list[str] = []
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

        if keep and (target is None or target > len(filtered_landmarks)):
            filtered_landmarks.append(line)
            if log:
                print(f"Image {imagename} added to filtered list.")
        else:
            remainder_landmarks.append(line)
            if log and target:
                print(f"Image {imagename} added to remainder list.")

    if len(filtered_landmarks) == 0:
        print("Warning: No images passed all filters.")
        exit(1)

    with open(filtered_landmarks_file, 'w') as file:
        file.writelines(filtered_landmarks)
    if log:
        print(len(filtered_landmarks),
              "filtered images saved to file",
              filtered_landmarks_file)

    if target and len(remainder_landmarks) != 0:
        with open(remainder_landmarks_file, 'w') as file:
            file.writelines(remainder_landmarks)
        if log:
            print(len(remainder_landmarks),
                  "remainder images saved to file",
                  remainder_landmarks_file)


if __name__ == "__main__":
    # You can call the preselect_landmarks function with a single filter…
    # preselect_landmarks('landmark_list.txt', gender='female')
    # Result: landmark_list_female.txt created with 11,317 images.

    # …or many filters. Race can be a single string…
    # preselect_landmarks('landmark_list.txt', age=(0, 50), race='asian')
    # Result: landmark_list_0-50_asian.txt created with 3,067 images.

    # …or a list with many strings.
    # preselect_landmarks('landmark_list.txt', race=['black', 'other'])
    # Result: landmark_list_black-other.txt created with 6,220 images.

    # You may want to enable logging to find out why each image was discarded
    # and to which file ('filtered' or 'remainder') an image was added. Note:
    # Non-existant filter values will now be ignored as in the below example:
    # preselect_landmarks('landmark_list.txt', race='caucasian', log=True)
    # Result: "Error: No valid filters to apply." message displayed.

    # The new (optional) `target` parameter allows you to specify the maximum
    # number of images (that meet the given filter criteria) to select. If
    # specified, then the output is (usually) two files: a "..._filtered.txt"
    # file with those images that pass all filters (up to the target value)
    # and a "..._remainder.txt" file with all remaining images.
    # preselect_landmarks('landmark_list.txt', gender="female", target=5000)
    # Result: landmark_list_female_filtered.txt created with 5,000 images.
    # Result: landmark_list_female_remainder.txt created with 18,708 images.

    # Images are sorted into the filtered/remainder files in the order they
    # are listed in the original file. The new `randomise` and `randomseed`
    # parameters will shuffle the lines prior to allocating the images; use
    # the same `randomseed` (int) parameter if you want reproducible results.
    # (If `randomseed` is omitted or None, the current system time is used.)
    preselect_landmarks('landmark_list.txt', gender="male", target=1,
                        randomise=True, randomseed=2, log=True)
    # Result: landmark_list_male_filtered.txt with 4_0_1_20170110213311678.jpg
    # Result: landmark_list_male_remainder.txt with remaining 23,707 images.
