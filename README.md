# Dr. Teeth and the Electric Mayhem

<img alt="Picture of the band" src="The_Electric_Mayhem.jpg">

## Preselecting images

If you're running the network on a subset of the dataset, it'll be useful to preselect the images that meet the desired criteria. This is how to go about that:

1. If you haven't already, fork this repo and clone it to your machine.
2. You'll also need to download the "Aligned & Cropped Faces" images from UTKFace's [Google Drive](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE?resourcekey=0-01Pth1hq20K4kuGVkp3oBw) (you want "UTKFace.tar.gz" which sits at about 102MB).
3. Unzip and copy the "UTKFace" folder into the repo's directory. (If it is not a sibling to _dataset.py,_ you will need to modify line 120 of that script.)
4. Within "UTKFace," you will need to correct the following filenames which are missing metadata fields:
    * 61_1_20170109142408075.jpg.chip.jpg -> 61_1_1_20170109142408075.jpg.chip.jpg
    * 61_1_20170109150557335.jpg.chip.jpg -> 61_1_3_20170109150557335.jpg.chip.jpg
    * 39_1_20170116174525125.jpg -> 39_1_1_20170116174525125.jpg
5. The _preselect.py_ script has loads of comments and example usage (from line 121).