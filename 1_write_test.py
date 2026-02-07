from luto.tools.write import write_outputs
import luto.simulation as sim
from luto import settings


save_dir = "output/2026_02_07__14_35_17_RF13_2010-2050"
data = sim.load_data_from_disk(f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
write_outputs(data)