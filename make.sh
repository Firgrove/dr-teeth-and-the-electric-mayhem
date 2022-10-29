#!/usr/bin/env sh

UTKZIP='UTKFace.tar.gz';
UTKDIR='UTKFace';

	if [ ! -d "$UTKDIR" ]; then
		if [ -f "$UTKZIP" ]; then
		  echo "Unzipping $UTKZIP to $UTKDIR..."
			tar -xvf UTKFace.tar.gz 2>/dev/null
			echo "Done. Feel free to delete $UTKZIP."
		else
		  echo "$UTKDIR directory not found. Please download $UTKZIP from https://gotil.la/3ziDcCX"
		fi
	fi
	if [ -d "$UTKDIR" ]; then
	  echo "$UTKDIR directory found."
	  cnt=0
		if [ -f "$UTKDIR/24_0_1_20170116220224657 .jpg.chip.jpg" ]; then
		    cnt=$((cnt+1))
		    echo "Renaming 24_0_1_20170116220224657 .jpg.chip.jpg\tto 24_0_1_20170116220224657.jpg.chip.jpg"
  			mv "$UTKDIR/24_0_1_20170116220224657 .jpg.chip.jpg" "$UTKDIR/24_0_1_20170116220224657.jpg.chip.jpg"
  	fi
		if [ -f "$UTKDIR/39_1_20170116174525125.jpg.chip.jpg" ]; then
		    cnt=$((cnt+1))
		    echo "Renaming 39_1_20170116174525125.jpg.chip.jpg\tto 39_1_1_20170116174525125.jpg.chip.jpg"
  			mv "$UTKDIR/39_1_20170116174525125.jpg.chip.jpg" "$UTKDIR/39_1_1_20170116174525125.jpg.chip.jpg"
  	fi
		if [ -f "$UTKDIR/61_1_20170109150557335.jpg.chip.jpg" ]; then
		    cnt=$((cnt+1))
		    echo "Renaming 61_1_20170109150557335.jpg.chip.jpg\tto 61_1_3_20170109150557335.jpg.chip.jpg"
  			mv "$UTKDIR/61_1_20170109150557335.jpg.chip.jpg" "$UTKDIR/61_1_3_20170109150557335.jpg.chip.jpg"
  	fi
		if [ -f "$UTKDIR/55_0_0_20170116232725357jpg.chip.jpg" ]; then
		    cnt=$((cnt+1))
		    echo "Renaming 55_0_0_20170116232725357jpg.chip.jpg\tto 55_0_0_20170116232725357.jpg.chip.jpg"
  			mv "$UTKDIR/55_0_0_20170116232725357jpg.chip.jpg" "$UTKDIR/55_0_0_20170116232725357.jpg.chip.jpg"
  	fi
		if [ -f "$UTKDIR/61_1_20170109142408075.jpg.chip.jpg" ]; then
		    cnt=$((cnt+1))
		    echo "Renaming 61_1_20170109142408075.jpg.chip.jpg\tto 61_1_1_20170109142408075.jpg.chip.jpg"
  			mv "$UTKDIR/61_1_20170109142408075.jpg.chip.jpg" "$UTKDIR/61_1_1_20170109142408075.jpg.chip.jpg"
  	fi
  	if [ "$cnt" -eq 0 ]; then
      echo "Nothing to do; all files correctly named."
  	fi
  else
    	  echo "$UTKDIR directory not found."
	fi
