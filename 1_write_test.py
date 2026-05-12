from luto.tools.write import write_outputs
import luto.simulation as sim
from luto import settings

save_dir = r"output\20260312_Paper3_test\Run_3_SCN_AgS3\output\2026_03_12__10_11_35_RF15_2010-2050"
data = sim.load_data_from_disk(f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
write_outputs(data)