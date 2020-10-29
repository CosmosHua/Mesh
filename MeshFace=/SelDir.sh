#!/bin/bash

#################################################################
SelDir() #copy [beg,end] dirs from src to dst
{
	src=$1; dst=$2; beg=$3; end=$4
	if [ ! -d $dst ]; then mkdir $dst; fi
	for i in $(ls $src);
	do
		#if [ $i -ge $beg ]&&[ $i -le $end ]; then echo "$i" >> List; fi
		if [ $i -ge $beg ]&&[ $i -le $end ]; then cp -r $src/$i $dst; fi
	done
} #0-263,268-473,484-500,221902-226986

src=Tools/CASIA-WebFace
#SelDir $src AI_CV_Train1 0 263
#SelDir $src AI_CV_Train2 268 473
#SelDir $src AI_CV_Test1 484 500
SelDir $src AI_CV_Test2 221902 226986

python3 MeshFace.py &


#################################################################
ReName() #rename *_?.jpg to *_.jpg, then
{
	src=$1; dst=$2;
	if [ ! -d $dst ]; then mkdir $dst; fi
	cd $src; for i in $(ls);
	do
		for j in $i/*_?.jpg; do mv $j ${j%?.jpg}.jpg; done #rename *_?.jpg to *_.jpg
		for j in $(ls $i); do cp $i/$j ../$dst/$i$j; done #copy+rename *.jpg to dst
	done
	cd; #zip -rq $dst.zip $dst &
}

src=AI_CV_Train1; dst=AI_CV_Train_1;
ReName $src $dst &
zip -rq $dst.zip $dst &

cd $dst;ls *[0-9].jpg >> ../$dst.txt; cd
