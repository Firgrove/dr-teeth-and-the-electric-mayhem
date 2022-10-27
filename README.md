# Dr. Teeth and the Electric Mayhem

<img alt="Picture of the band" src="The_Electric_Mayhem.jpg">

## Set up

1. If you haven't already, fork this repo and clone it to your machine.
2. You'll also need to download the "Aligned & Cropped Faces" images from UTKFace's [Google Drive](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE?resourcekey=0-01Pth1hq20K4kuGVkp3oBw) (you want "UTKFace.tar.gz" which sits at about 102MB) and copy the file into the repo folder.
3. If you are on a Mac or Unix/Linux machine, you can run the make.sh file to unzip the above file and correct 5 known issues with image naming. **Alternatively, perform the following steps:**
   1. Unzip _UTKFace.tar.gz_ .
   2. Within the "UTKFace" directory, you you will need to correct the following filenames which are missing metadata fields:
       * 24_0_1_20170116220224657 .jpg.chip.jpg -> 24_0_1_20170116220224657.jpg.chip.jpg
       * 55_0_0_20170116232725357jpg.chip.jpg -> 55_0_0_20170116232725357.jpg.chip.jpg
       * 61_1_20170109142408075.jpg.chip.jpg -> 61_1_1_20170109142408075.jpg.chip.jpg
       * 61_1_20170109150557335.jpg.chip.jpg -> 61_1_3_20170109150557335.jpg.chip.jpg
       * 39_1_20170116174525125.jpg.chip.jpg -> 39_1_1_20170116174525125.jpg.chip.jpg

## Preselecting images

If you're running the network on a subset of the dataset, it'll be useful to preselect the images that meet the desired criteria. The _preselect.py_ script has loads of comments and example usage (from line 121) for how to do this.