from luto.tools.write import write_outputs
import luto.simulation as sim
from luto import settings

save_dir = r"output\20260210_Paper1_Results_aquila_test\Run_1_GHG_low_BIO_low\output\2026_02_13__06_14_27_RF13_2010-2050"
data = sim.load_data_from_disk(f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
write_outputs(data)