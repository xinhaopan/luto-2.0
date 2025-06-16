import os
os.environ['GRB_LICENSE_FILE'] = r'C:\Users\s222552331\gurobi\gurobi_xp.lic'
import luto.simulation as sim

data = sim.load_data()
sim.run(data=data)
pkl_path = f'{data.path}/data_with_solution.gz'
sim.save_data_to_disk(data,pkl_path)
from luto.tools.write import write_outputs
write_outputs(data)