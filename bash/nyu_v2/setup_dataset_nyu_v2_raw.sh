#!/bin/bash

mkdir -p 'data/nyu_v2/tmp'

nyu_v2_urls=(
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/basements.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bathrooms_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bathrooms_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bathrooms_part3.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bathrooms_part4.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part3.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part4.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part5.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part6.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bedrooms_part7.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bookstore_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bookstore_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/bookstore_part3.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/cafe.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/dining_rooms_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/dining_rooms_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/furniture_stores.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/home_offices.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/kitchens_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/kitchens_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/kitchens_part3.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/libraries.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/living_rooms_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/living_rooms_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/living_rooms_part3.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/living_rooms_part4.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/misc_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/misc_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/offices_part1.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/offices_part2.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/office_kitchens.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/playrooms.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/reception_rooms.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/studies.zip
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/study_rooms.zip
)

for file_url in ${nyu_v2_urls[@]}; do
    wget $file_url -P data/nyu_v2/tmp
done

