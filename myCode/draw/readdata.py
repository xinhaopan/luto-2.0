import pandas as pd
INPUT_DIR = '../input'
AGEC_CROPS = pd.read_hdf(os.path.join(INPUT_DIR, "agec_crops.h5"))
AGEC_LVSTK = pd.read_hdf(os.path.join(INPUT_DIR, "agec_lvstk.h5"))