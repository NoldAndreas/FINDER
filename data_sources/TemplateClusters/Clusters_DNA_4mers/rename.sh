#!/bin/bash
for fn in $(ls *_cluster_*.txt);
do
   echo "Renaming " $fn "to "${fn:6}
   mv $fn ${fn:6}
done
