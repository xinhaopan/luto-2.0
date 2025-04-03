import luto.simulation as sim
data = sim.load_data()
sim.run(data=data, base_year=2010, target_year=2050)
pkl_path = f'{data.path}/data_with_solution.gz'
sim.save_data_to_disk(data,pkl_path)
from luto.tools.write import write_outputs
write_outputs(data)