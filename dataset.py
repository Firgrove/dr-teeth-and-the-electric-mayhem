from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file,
                                      sep=r'\s+',
                                      header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.genders = {
            '0': "male",
            '1': "female"
        }
        self.races = {
            '0': "white",
            '1': "black",
            '2': "asian",
            '3': "indian",
            '4': "others"
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Reads file for given index as a tensor image
        imagename = self.img_labels.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, imagename))

        # Applies any transformations to image
        if self.transform:
            image = self.transform(image)

        # Reads the image metadata from the filename
        splitname = imagename.split(".")[0].split("_")
        age = int(splitname[0])
        gender = self.genders[splitname[1]]
        race = self.races[splitname[2]]
        datetime = int(splitname[3][:13])

        # Reorganises the x,y landmark coordinates as a 68x2 dataframe
        coords = self.img_labels.iloc[idx, 1:].tolist()
        raw_landmarks = []
        for i in range(0, len(coords), 2):
            raw_landmarks.append((coords[i], coords[i+1]))
        landmarks = pd.DataFrame(raw_landmarks, columns=['x', 'y'])

        # Returns dict with requested image and its metadata
        return {'image': image,
                'age': age,
                'genders': gender,
                'races': race,
                'datetime': datetime,
                'landmarks': landmarks}


if __name__ == "__main__":
    UTKFace = CustomImageDataset('landmark_list.txt', 'UTKFace')
    image0 = UTKFace[0]
    print(image0)