#!/bin/bash

# Read the settings_bash file ==> MEM, TIME, THREADS, JOB_NAME
source luto/settings_bash.py
export PATH=$PATH:/usr/local/bin

# Check for gurobi license file
if [ -f ~/gurobi.lic ]; then
    grb_license=~/gurobi.lic
else
    grb_license=~/gurobi_${NODE}.lic
fi

export GRB_LICENSE_FILE=$grb_license

# Create a temporary script file
SCRIPT=$(mktemp)

# Write the script content to the file, directly substituting variables
cat << OUTER_EOF > $SCRIPT
#!/bin/bash

# Read the settings_bash file ==> NODE, TIME, MEM, THREADS, JOB_NAME
source luto/settings_bash.py

if [[ \$NODE != "mem" ]]; then
    if [[ \$NODE == "gc1" ]]; then
        echo "Current node is \$NODE, syncing files from hpc-fc-b-1:/run/user/219976/python to hpc-gc-b-1:/run/user/219976/python."

        ssh s222552331@hpc-fc-b-1 "rsync -avz --delete /run/user/219976/python/ s222552331@hpc-gc-b-1:/run/user/219976/python/"
    elif [[ \$NODE == "gc2" ]]; then
        echo "Current node is \$NODE, syncing files from hpc-fc-b-1:/run/user/219976/python to hpc-gc-b-2:/run/user/219976/python."

        ssh s222552331@hpc-fc-b-1 "rsync -avz --delete /run/user/219976/python/ s222552331@hpc-gc-b-2:/run/user/219976/python/"
    else
        echo "Current node is \$NODE, syncing files from hpc-fc-b-1:/run/user/219976/python to hpc-\$NODE-b-1:/run/user/219976/python."

        ssh s222552331@hpc-fc-b-1 "rsync -avz --delete /run/user/219976/python/ s222552331@hpc-\$NODE-b-1:/run/user/219976/python/"
    fi

    # check rsync was successful
    if [ \$? -eq 0 ]; then
        echo "Files synced successfully."
    else
        echo "Error occurred during file sync."
        exit 1
    fi
else
    echo "Current node is mem, no sync performed."
fi


# Activate conda environment
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

# Sync output files if the simulation was successful
if [ \$? -eq 0 ]; then
    /home/s222552331/rsync/sync_files.sh
fi

OUTER_EOF

# Submit the job using sbatch
if [[ "$NODE" == "gc1" ]]; then
    sbatch -p normal \
    --nodelist=hpc-gc-b-1 \
    --time=${TIME} \
    --mem=${MEM} \
    --cpus-per-task=${THREADS} \
    --job-name=${JOB_NAME} \
    ${SCRIPT}
elif [[ "$NODE" == "gc2" ]]; then
    sbatch -p normal \
    --nodelist=hpc-gc-b-2 \
    --time=${TIME} \
    --mem=${MEM} \
    --cpus-per-task=${THREADS} \
    --job-name=${JOB_NAME} \
    ${SCRIPT}
else
    sbatch -p ${NODE} \
    --time=${TIME} \
    --mem=${MEM} \
    --cpus-per-task=${THREADS} \
    --job-name=${JOB_NAME} \
    ${SCRIPT}
fi

# Optionally remove the temporary script after job submission
# You might want to comment this line if you want to debug the script
# rm $SCRIPT
