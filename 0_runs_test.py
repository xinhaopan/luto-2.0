
import luto.simulation as sim

data = sim.load_data()
sim.run(data=data)
from luto.tools.write_0 import write_outputs
write_outputs(data)