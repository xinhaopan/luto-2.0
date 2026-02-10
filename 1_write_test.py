from luto.tools.write import write_outputs
import luto.simulation as sim
from luto import settings


save_dir = "output/test"
data = sim.load_data_from_disk(f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
write_outputs(data)