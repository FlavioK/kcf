#!/bin/bash
mkdir -p logs
echo "nvidia" | sudo -S chrt -f 99 build-cufft-openmp-trad/kcf_vot --fit=512 vot2016/ball1
#echo "nvidia" | sudo -S chrt -f 99 build-cufft-openmp-trad/kcf_vot --fit=512 ballshort
#echo "nvidia" | sudo -S chrt -f 99 build-cufft-trad/kcf_vot --fit=512 ballshort
echo "nvidia" | sudo -S chown nvidia:nvidia *.log
mv *.log logs/.
