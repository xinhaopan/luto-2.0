dd_ccimpact_df = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_2010_2100_cc_dd_ml.h5'))
dd_ccimpact_df.columns = dd_ccimpact_df.columns.droplevel("Region_name")
dd_ccimpact_df = dd_ccimpact_df.loc[:, pd.IndexSlice[:, settings.SSP]]
dd_ccimpact_df.columns = dd_ccimpact_df.columns.droplevel('ssp')