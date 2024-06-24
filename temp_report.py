from luto import settings
from luto.tools.report.create_report_data import save_report_data
from luto.tools.report.create_html import data2html
from luto.tools.report.create_static_maps import TIF2MAP

path_name = "2024_06_18__10_19_14_soft_mincost_RF10_P1e5_2010-2015_timeseries_82Mt"
path = "output/" + path_name
settings.WRITE_OUTPUT_GEOTIFFS = True
TIF2MAP(path) if settings.WRITE_OUTPUT_GEOTIFFS else None
save_report_data(path)
data2html(path)