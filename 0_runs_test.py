import luto.simulation as sim
data = sim.load_data()
import luto.simulation as sim
sim.run(data=data, base_year=2010, target_year=2050)
from luto.tools.write import write_outputs
write_outputs(data)