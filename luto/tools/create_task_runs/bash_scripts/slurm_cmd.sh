#!/bin/bash

# Read the settings_bash file ==> MEM, TIME, THREADS, JOB_NAME
source luto/settings_bash.py
export PATH=$PATH:/usr/local/bin

if [ -f ~/gurobi.lic ]; then
    grb_license=~/gurobi.lic
else
    grb_license=~/gurobi_${NODE}.lic
fi

export GRB_LICENSE_FILE=$grb_license

# Create a temporary script file
SCRIPT=$(mktemp)

# Write the script content to the file, directly substituting variables
cat << 'OUTER_EOF' > $SCRIPT
#!/bin/bash

# Read the settings_bash file ==> NODE, TIME, MEM, THREADS, JOB_NAME
source luto/settings_bash.py
if [[ $NODE == "dgx" ]]; then
    echo "Current node is dgx, syncing files from hpc-fc-b-1:/run/user/219976/python to hpc-dgx-b-1:/run/user/219976/python."
    
    ssh s222552331@hpc-fc-b-1 "rsync -avz --delete /run/user/219976/python/ s222552331@hpc-dgx-b-1:/run/user/219976/python/"
    
    if [ $? -eq 0 ]; then
        echo "Files synced successfully."
    else
        echo "Error occurred during file sync."
        exit 1
    fi
else
    echo "Current node is not dgx, no sync performed."
fi

source /run/user/219976/python/miniforge3/etc/profile.d/conda.sh
conda activate luto

# Run the simulation
python <<-INNER_EOF
import luto.simulation as sim
data = sim.load_data()
sim.run(data=data, base=2010, target=2050)
from luto.tools.write import write_outputs
write_outputs(data)
INNER_EOF

if [ $? -eq 0 ]; then
    /home/s222552331/rsync/sync_files.sh
fi

OUTER_EOF

# Submit the job
sbatch -p ${NODE} \
    --time=${TIME} \
    --mem=${MEM} \
    --cpus-per-task=${THREADS} \
    --job-name=${JOB_NAME} \
    ${SCRIPT}

# Remove the temporary script file
rm $SCRIPT
