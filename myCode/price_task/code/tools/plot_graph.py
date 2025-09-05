# --- 标准库 ---
import os
import re
import math

# --- 第三方库 ---import numpy as np
import pandas as pd
from plotnine import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import seaborn as sns

import cairosvg
from lxml import etree

# --- 本地模块 ---
import tools.config as config
from .tools import get_path, get_year


plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

columns_name = ["cost_ag(M$)", "cost_am(M$)", "cost_non-ag(M$)", "cost_transition_ag2ag(M$)",
                    "cost_amortised_transition_ag2non-ag(M$)","revenue_ag(M$)","revenue_am(M$)","revenue_non-ag(M$)",
                    "GHG_ag(MtCOe2)", "GHG_am(MtCOe2)", "GHG_non-ag(MtCOe2)", "GHG_transition(MtCOe2)",
                    "BIO_ag(M ha)", "BIO_am(M ha)", "BIO_non-ag(M ha)"]
title_name = ["AG Cost", "AM cost", "NON-AG cost", "Transition cost (AG to AG)","Amortised Transition (AG to NON-AG)","AG Revenue","AM revenue","NON-AG revenue",
                    "AG GHG emission", "AM GHG emission", "NON-AG GHG emission", "Transition GHG emision",
                    "AG Biodiversity", "AM Biodiversity", "NON-AG Biodiversity "]
col_map = dict(zip(columns_name, title_name))
df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_Run_3_GHG_high_BIO_off.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_Run_4_GHG_high_BIO_high.xlsx", index_col=0)

# 只保留匹配的列，并重命名
df_ghg = df_ghg[[col for col in df_ghg.columns if col in col_map]]
df_ghg = df_ghg.rename(columns=col_map)

df_ghg_bio = df_ghg_bio[[col for col in df_ghg_bio.columns if col in col_map]]
df_ghg_bio = df_ghg_bio.rename(columns=col_map)

def plot_lines(df, df_ghg):