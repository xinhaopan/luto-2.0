start_year = 2010
end_year = 2050
nums = [0]
MODEs = ['timeseries', 'timeseries', 'timeseries', 'snapshot', 'snapshot', 'snapshot']
RESFACTORs = [1, 3, 3, 1, 1, 1]
GHGs = ['1.5C (67%) excl. avoided emis','1.5C (50%) excl. avoided emis','1.8C (67%) excl. avoided emis','1.5C (67%) excl. avoided emis','1.5C (50%) excl. avoided emis','1.8C (67%) excl. avoided emis']
CARBON_PRICEs = [588,506,435,588,506,435]

for i in nums:
    print(f"MODE = {MODEs[i]},RESFACTOR = {RESFACTORs[i]},GHGs = {GHGs[i]}")

    import luto.settings as settings
    # if RESFACTORs[i] == 1:
    #     settings.WRITE_OUTPUT_GEOTIFFS = True
    # else:
    #     settings.WRITE_OUTPUT_GEOTIFFS = False
    settings.OBJECTIVE = 'mincost'
    settings.MODE = MODEs[i]  # 'snapshot''timeseries'
    settings.RESFACTOR = RESFACTORs[i]
    settings.GHG_LIMITS_FIELD = GHGs[i]
    # settings.CARBON_PRICE_PER_TONNE = CARBON_PRICEs[i]

    import luto.simulation as sim
    import luto.tools.write as write

    data = sim.load_data()
    sim.run(data=data, base=start_year, target=end_year)
    write.write_outputs(data)

