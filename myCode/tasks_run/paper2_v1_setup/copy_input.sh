#!/bin/bash
#PBS -N p2v1_copy_input
#PBS -P jk53
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=05:00:00
#PBS -l storage=gdata/jk53
#PBS -l wd
set -e
SRC=/g/data/jk53/jinzhu/LUTO/luto-2.0/input/
DST=/g/data/jk53/LUTO_XH/LUTO2_paper2_v1/input/
rsync -a --exclude=.gitignore "$SRC" "$DST"
# SSP3-7.0 has no own climate impact layer: RCP6.0 layer copied under the rcp7p0 name (Brett, 2026-11-13)
for f in ON OFF; do cp -p "$DST/climate_change_impacts_rcp6p0_CO2_FERT_$f.nc" "$DST/climate_change_impacts_rcp7p0_CO2_FERT_$f.nc"; done
echo "rcp7p0 = copy of rcp6p0 (made $(date +%F))" > "$DST/README_rcp7p0_is_rcp6p0.txt"
du -sh "$DST"
echo COPY_DONE
