#!/bin/bash
for fn in $(ls *_cluster_*.txt);
do
    echo "Renaming "$fn
    mv $fn ${fn:6}
done
