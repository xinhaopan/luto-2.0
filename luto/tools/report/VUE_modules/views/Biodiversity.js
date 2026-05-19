window.BiodiversityView = {
  name: 'BiodiversityView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch } = Vue;

    const chartRegister = window.ChartService.chartCategories["Biodiversity"];
    const mapRegister = window.MapService.mapCategories["Biodiversity"];
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Biodiversity";

    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");
    const availableRegionLevels = ['region_state', 'region_NRM'];
    const selectRegionLevel = ref('region_state');

    const availableYears = ref([]);
    const availableUnit = { Biodiversity: "Relative Percentage (Pre-1750 = 100%)" };

    const METRIC_LABELS = {
      'quality': 'Quality', 'GBF2': 'GBF2', 'GBF3_NVIS': 'GBF3 NVIS',
      'GBF4_SNES': 'GBF4 SNES', 'GBF4_ECNES': 'GBF4 ECNES',
      'GBF8_GROUP': 'GBF8 Group', 'GBF8_SPECIES': 'GBF8 Species',
    };
    const METRIC_TO_SETTING = {
      'quality': null, 'GBF2': 'BIODIVERSITY_TARGET_GBF_2',
      'GBF3_NVIS': 'BIODIVERSITY_TARGET_GBF_3_NVIS',
      'GBF4_SNES': 'BIODIVERSITY_TARGET_GBF_4_SNES',
      'GBF4_ECNES': 'BIODIVERSITY_TARGET_GBF_4_ECNES',
      'GBF8_GROUP': 'BIODIVERSITY_TARGET_GBF_8', 'GBF8_SPECIES': 'BIODIVERSITY_TARGET_GBF_8',
    };

    const availableMetrics = ref(['quality']);
    const ALL_CATEGORIES = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    const availableCategories = computed(() => {
      const mr = mapRegister[selectMetric.value] || {};
      return ALL_CATEGORIES.filter(c => mr[c]);
    });
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableSpecies = ref([]);
    const availableLanduse = ref([]);

    const selectMetric = ref("quality");
    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectSpecies = ref("");
    const selectLanduse = ref("");
    const selectBackend = ref("");
    const availableBackends = ref([]);
    const modelBackend = ref("");

    const METRICS_WITH_SPECIES = ['GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8_GROUP', 'GBF8_SPECIES'];
    const hasSpecies = computed(() => METRICS_WITH_SPECIES.includes(selectMetric.value));
    const speciesLabel = computed(() => {
      const m = selectMetric.value;
      if (m === 'GBF3_NVIS') return 'Veg group:';
      if (m === 'GBF4_SNES' || m === 'GBF8_SPECIES') return 'Species:';
      if (m === 'GBF4_ECNES') return 'Community:';
      if (m === 'GBF8_GROUP') return 'Group:';
      return 'Species:';
    });

    const previousSelections = ref({
      "Sum": { species: "", landuse: "" },
      "Ag": { water: "", species: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", species: "", landuse: "" },
      "Non-Ag": { species: "", landuse: "" }
    });

    const SUM_TYPE_LABELS = { 'ALL': 'ALL', 'ag': 'Ag', 'non-ag': 'Non-Ag', 'ag-man': 'Ag Mgt' };
    function formatLanduse(val) {
      return (selectCategory.value === 'Sum') ? (SUM_TYPE_LABELS[val] || val) : val;
    }

    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    const gbf2MaskOverlay = computed(() =>
      (dataLoaded.value && selectMetric.value === 'GBF2') ? (window.BIO_GBF2_MASK || null) : null
    );

    // ── Per-combo map layer loader ──────────────────────────────────────────
    const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);

    const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

    // For the quality metric, chart data has an extra 'backend' outermost key.
    function qualityRoot(name) {
      const isQuality = selectMetric.value === 'quality' && selectBackend.value;
      return isQuality ? window[name]?.[selectBackend.value] : window[name];
    }

    const selectMultiInput = computed(() => {
      if (!dataLoaded.value) return null;
      const metric = selectMetric.value;
      const cat = selectCategory.value;
      const cr = chartRegister[metric];
      const region = selectRegion.value;
      const water = selectWater.value;
      const agMgt = selectAgMgt.value;
      const landuse = selectLanduse.value;
      const species = selectSpecies.value;
      const withSpecies = hasSpecies.value;

      let candidate = null;

      if (cat === 'Sum') {
        const sumEntry = cr?.['Sum'] ?? cr?.['overview']?.['sum'];
        if (!sumEntry) return null;
        const raw = qualityRoot(sumEntry.name)?.[selectRegionLevel.value]?.[region];
        if (!raw) return null;
        candidate = withSpecies ? raw?.[species] : raw;
      } else if (cat === 'Ag') {
        const entry = cr?.['Ag'];
        if (!entry) return null;
        const root = qualityRoot(entry.name);
        const node = withSpecies ? root?.[selectRegionLevel.value]?.[region]?.[species] : root?.[selectRegionLevel.value]?.[region];
        candidate = node?.[water];
      } else if (cat === 'Ag Mgt') {
        const entry = cr?.['Ag Mgt'];
        if (!entry) return null;
        const root = qualityRoot(entry.name);
        const node = withSpecies ? root?.[selectRegionLevel.value]?.[region]?.[species] : root?.[selectRegionLevel.value]?.[region];
        candidate = node?.[agMgt]?.[water];
      } else if (cat === 'Non-Ag') {
        const entry = cr?.['Non-Ag'];
        if (!entry) return null;
        const raw = qualityRoot(entry.name)?.[selectRegionLevel.value]?.[region];
        candidate = withSpecies ? raw?.[species] : raw;
      }

      if (!candidate || Array.isArray(candidate) || candidate['Area'] === undefined) return null;

      if (landuse !== 'ALL' && cat !== 'Sum') {
        return {
          Percent: (candidate.Percent || []).filter(s => s.name === landuse),
          Area:    (candidate.Area    || []).filter(s => s.name === landuse),
        };
      }
      return candidate;
    });

    const selectMultiYAxis = {
      'Area': 'Area-weighted Score (ha)',
      'Percent': 'Relative Percentage (Pre-1750 = 100%)',
    };

    const selectChartData = computed(() => {
      const metric = selectMetric.value;
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const region = selectRegion.value;
      const species = selectSpecies.value;
      const withSpecies = hasSpecies.value;
      if (!dataLoaded.value) return {};
      const cr = chartRegister[metric];
      const dataRoot = qualityRoot(cr?.[cat]?.["name"]);
      const regionNode = dataRoot?.[selectRegionLevel.value]?.[region];
      const chartData = withSpecies ? regionNode?.[species] : regionNode;
      let seriesData;

      if (cat === "Sum") {
        const sumEntry = cr?.['Sum'] ?? cr?.['overview']?.['sum'];
        const rawSumData = qualityRoot(sumEntry?.['name'])?.[selectRegionLevel.value]?.[region];
        const candidate = withSpecies ? rawSumData?.[species] : rawSumData;
        const isMultiInput = candidate && !Array.isArray(candidate) && candidate['Area'] !== undefined;
        seriesData = isMultiInput ? [] : (candidate || []);
      } else if (cat === "Ag") {
        const agData = chartData?.[water];
        seriesData = Array.isArray(agData) ? agData.filter(s => landuse === "ALL" || s.name === landuse) : [];
      } else if (cat === "Ag Mgt") {
        const amData = chartData?.[agMgt]?.[water];
        seriesData = Array.isArray(amData) ? amData.filter(s => landuse === "ALL" || s.name === landuse) : [];
      } else if (cat === "Non-Ag") {
        seriesData = Array.isArray(chartData) ? chartData.filter(s => landuse === "ALL" || s.name === landuse) : [];
      }
      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        plotOptions: { column: { stacking: 'normal' } },
        yAxis: { title: { text: availableUnit["Biodiversity"] } },
        series: seriesData || [],
      };
    });

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    // ── Helpers ──────────────────────────────────────────────────────────────
    function getTree(metric, cat) {
      const rawTree = window[mapRegister[metric]?.[cat]?.indexName]?.tree ?? (cat === "Non-Ag" ? [] : {});
      // For quality metric the outermost tree key is backend — slice it so doCascade
      // sees the same structure as every other metric (lm → lu, am → lm → lu, etc.).
      if (metric === 'quality' && selectBackend.value && !Array.isArray(rawTree)) {
        return rawTree[selectBackend.value] ?? (cat === "Non-Ag" ? [] : {});
      }
      return rawTree;
    }

    async function ensureIndexLoaded(metric, cat) {
      const entry = mapRegister[metric]?.[cat];
      if (entry && !window[entry.indexName]) {
        isLoadingData.value = true;
        await loadScript(entry.indexPath, entry.indexName, VIEW_NAME);
        isLoadingData.value = false;
      }
    }

    // Build combo array for ensureComboLayer
    function buildCombo(cat, tree, withSpecies) {
      const metric = selectMetric.value;
      const isQuality = metric === 'quality';
      const b = isQuality ? [selectBackend.value] : [];
      if (cat === "Sum") {
        return [...b, ...(withSpecies ? [selectSpecies.value, selectLanduse.value] : [selectLanduse.value])];
      } else if (cat === "Ag") {
        return [...b, ...(withSpecies ? [selectWater.value, selectSpecies.value, selectLanduse.value] : [selectWater.value, selectLanduse.value])];
      } else if (cat === "Ag Mgt") {
        return [...b, ...(withSpecies ? [selectAgMgt.value, selectWater.value, selectSpecies.value, selectLanduse.value] : [selectAgMgt.value, selectWater.value, selectLanduse.value])];
      } else if (cat === "Non-Ag") {
        return [...b, ...(withSpecies ? [selectSpecies.value, selectLanduse.value] : [selectLanduse.value])];
      }
      return [...b, selectLanduse.value];
    }

    async function doCascade(cat) {
      const metric = selectMetric.value;
      const tree = getTree(metric, cat);
      const withSpecies = hasSpecies.value;
      const curW = selectWater.value, curL = selectLanduse.value;
      const curAm = selectAgMgt.value, curSp = selectSpecies.value;

      if (cat === "Sum") {
        availableAgMgt.value = []; availableWater.value = [];
        if (withSpecies && !Array.isArray(tree)) {
          availableSpecies.value = Object.keys(tree);
          const prevSp = previousSelections.value["Sum"]?.species || curSp;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = tree[selectSpecies.value] || [];
        } else {
          availableSpecies.value = [];
          selectSpecies.value = '';
          availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
        }
        const prevLU = previousSelections.value["Sum"]?.landuse || curL;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (cat === "Ag") {
        if (withSpecies) {
          // tree: { water: { species: [lu] } }
          availableWater.value = Object.keys(tree);
          const prevW = previousSelections.value["Ag"]?.water || curW;
          selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
          availableSpecies.value = Object.keys(tree[selectWater.value] || {});
          const prevSp = previousSelections.value["Ag"]?.species || curSp;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = tree[selectWater.value]?.[selectSpecies.value] || [];
        } else {
          // tree: { water: [lu] }
          availableSpecies.value = []; selectSpecies.value = '';
          availableWater.value = Object.keys(tree);
          const prevW = previousSelections.value["Ag"]?.water || curW;
          selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
          availableLanduse.value = tree[selectWater.value] || [];
        }
        const prevLU = previousSelections.value["Ag"]?.landuse || curL;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (cat === "Ag Mgt") {
        if (withSpecies) {
          // tree: { am: { lm: { species: [lu] } } }
          availableAgMgt.value = Object.keys(tree);
          const prevAm = previousSelections.value["Ag Mgt"]?.agMgt || curAm;
          selectAgMgt.value = (prevAm && availableAgMgt.value.includes(prevAm)) ? prevAm : (availableAgMgt.value[0] || '');
          availableWater.value = Object.keys(tree[selectAgMgt.value] || {});
          const prevW = previousSelections.value["Ag Mgt"]?.water || curW;
          selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
          availableSpecies.value = Object.keys(tree[selectAgMgt.value]?.[selectWater.value] || {});
          const prevSp = previousSelections.value["Ag Mgt"]?.species || curSp;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value]?.[selectSpecies.value] || [];
        } else {
          // tree: { am: { lm: [lu] } }
          availableSpecies.value = []; selectSpecies.value = '';
          availableAgMgt.value = Object.keys(tree);
          const prevAm = previousSelections.value["Ag Mgt"]?.agMgt || curAm;
          selectAgMgt.value = (prevAm && availableAgMgt.value.includes(prevAm)) ? prevAm : (availableAgMgt.value[0] || '');
          availableWater.value = Object.keys(tree[selectAgMgt.value] || {});
          const prevW = previousSelections.value["Ag Mgt"]?.water || curW;
          selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
          availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value] || [];
        }
        const prevLU = previousSelections.value["Ag Mgt"]?.landuse || curL;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (cat === "Non-Ag") {
        availableAgMgt.value = []; availableWater.value = [];
        if (withSpecies && !Array.isArray(tree)) {
          // tree: { species: [lu] }
          availableSpecies.value = Object.keys(tree);
          const prevSp = previousSelections.value["Non-Ag"]?.species || curSp;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = tree[selectSpecies.value] || [];
        } else {
          availableSpecies.value = []; selectSpecies.value = '';
          availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
        }
        const prevLU = previousSelections.value["Non-Ag"]?.landuse || curL;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
      }

      const entry = mapRegister[metric]?.[cat];
      if (entry?.layerPrefix) {
        await ensureComboLayer(entry.layerPrefix, buildCombo(cat, tree, withSpecies));
      }
    }

    async function loadAllCharts() {
      const pending = [];
      for (const metric of availableMetrics.value) {
        const metricCr = chartRegister[metric];
        for (const [key, val] of Object.entries(metricCr || {})) {
          if (key === 'overview') {
            for (const entry of Object.values(val || {})) {
              if (entry?.name && !window[entry.name]) pending.push(loadScript(entry.path, entry.name, VIEW_NAME));
            }
          } else if (val?.name && !window[val.name]) {
            pending.push(loadScript(val.path, val.name, VIEW_NAME));
          }
        }
      }
      if (pending.length > 0) await Promise.allSettled(pending);
    }

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);

      const runScenario = {};
      (window.Supporting_info.model_run_settings || []).forEach(s => { runScenario[s.parameter] = s.val; });

      METRIC_LABELS['GBF3_NVIS'] = 'GBF3 (NVIS)';

      const enabledMetrics = ['quality'];
      for (const [metric, settingKey] of Object.entries(METRIC_TO_SETTING)) {
        if (metric === 'quality') continue;
        if (settingKey && runScenario[settingKey] !== 'off' && mapRegister[metric]) {
          enabledMetrics.push(metric);
        }
      }
      availableMetrics.value = enabledMetrics;

      if (enabledMetrics.includes('GBF2')) {
        const mask = mapRegister['GBF2']['mask'];
        await loadScript(mask.path, mask.name, VIEW_NAME);
      }

      availableYears.value = window.Supporting_info.years;
      selectYear.value = availableYears.value[0] || 2020;

      const initMetric = enabledMetrics[0];
      const initMr = mapRegister[initMetric] || {};
      const initCat = ALL_CATEGORIES.find(c => initMr[c]) || "Ag";

      await Promise.all([ensureIndexLoaded(initMetric, initCat), loadAllCharts()]);

      selectMetric.value = initMetric;

      // Discover available backends from BIO_quality_Ag top-level keys
      if (window["BIO_quality_Ag"]) {
        const backends = Object.keys(window["BIO_quality_Ag"]);
        availableBackends.value = backends;
        const mbSetting = (window.Supporting_info.model_run_settings || []).find(s => s.parameter === "BIO_QUALITY_LAYER")?.val || "";
        modelBackend.value = mbSetting;
        selectBackend.value = backends.includes(mbSetting) ? mbSetting : (backends[0] || "");
      }

      await doCascade(initCat);
      selectCategory.value = initCat;
      dataLoaded.value = true;
    });

    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };
    watch(yearIndex, (i) => { selectYear.value = availableYears.value[i]; });
    watch(selectRegionLevel, () => { selectRegion.value = 'AUSTRALIA'; });

    watch(selectMetric, async (newMetric) => {
      const mr = mapRegister[newMetric] || {};
      let cat = selectCategory.value;
      if (!mr[cat]) {
        cat = ALL_CATEGORIES.find(c => mr[c]) || cat;
        selectCategory.value = cat;
        return;
      }
      await ensureIndexLoaded(newMetric, cat);
      await doCascade(cat);
    });

    watch(selectCategory, async (newCat, oldCat) => {
      if (oldCat === "Sum") previousSelections.value["Sum"] = { species: selectSpecies.value, landuse: selectLanduse.value };
      if (oldCat === "Ag") previousSelections.value["Ag"] = { water: selectWater.value, species: selectSpecies.value, landuse: selectLanduse.value };
      if (oldCat === "Ag Mgt") previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, species: selectSpecies.value, landuse: selectLanduse.value };
      if (oldCat === "Non-Ag") previousSelections.value["Non-Ag"] = { species: selectSpecies.value, landuse: selectLanduse.value };
      await ensureIndexLoaded(selectMetric.value, newCat);
      await doCascade(newCat);
    });

    watch(selectAgMgt, async (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const metric = selectMetric.value;
      const tree = getTree(metric, "Ag Mgt");
      const withSpecies = hasSpecies.value;
      availableWater.value = Object.keys(tree[newAgMgt] || {});
      const prevW = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
      if (withSpecies) {
        availableSpecies.value = Object.keys(tree[newAgMgt]?.[selectWater.value] || {});
        const prevSp = previousSelections.value["Ag Mgt"].species;
        selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
        availableLanduse.value = tree[newAgMgt]?.[selectWater.value]?.[selectSpecies.value] || [];
      } else {
        availableLanduse.value = tree[newAgMgt]?.[selectWater.value] || [];
      }
      const prevL = previousSelections.value["Ag Mgt"].landuse;
      selectLanduse.value = (prevL && availableLanduse.value.includes(prevL)) ? prevL : (availableLanduse.value[0] || '');
      const entry = mapRegister[metric]?.["Ag Mgt"];
      if (entry?.layerPrefix) await ensureComboLayer(entry.layerPrefix, buildCombo("Ag Mgt", tree, withSpecies));
    });

    watch(selectWater, async (newWater) => {
      const cat = selectCategory.value;
      if (cat !== "Ag" && cat !== "Ag Mgt") return;
      previousSelections.value[cat].water = newWater;
      const metric = selectMetric.value;
      const tree = getTree(metric, cat);
      const withSpecies = hasSpecies.value;
      if (cat === "Ag") {
        if (withSpecies) {
          availableSpecies.value = Object.keys(tree[newWater] || {});
          const prevSp = previousSelections.value["Ag"].species;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = tree[newWater]?.[selectSpecies.value] || [];
        } else {
          availableLanduse.value = tree[newWater] || [];
        }
      } else { // Ag Mgt
        if (withSpecies) {
          availableSpecies.value = Object.keys(tree[selectAgMgt.value]?.[newWater] || {});
          const prevSp = previousSelections.value["Ag Mgt"].species;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = tree[selectAgMgt.value]?.[newWater]?.[selectSpecies.value] || [];
        } else {
          availableLanduse.value = tree[selectAgMgt.value]?.[newWater] || [];
        }
      }
      const prevL = previousSelections.value[cat].landuse;
      selectLanduse.value = (prevL && availableLanduse.value.includes(prevL)) ? prevL : (availableLanduse.value[0] || '');
      const entry = mapRegister[metric]?.[cat];
      if (entry?.layerPrefix) await ensureComboLayer(entry.layerPrefix, buildCombo(cat, tree, withSpecies));
    });

    watch(selectSpecies, async (newSpecies) => {
      if (!hasSpecies.value || !newSpecies) return;
      const cat = selectCategory.value;
      const metric = selectMetric.value;
      const tree = getTree(metric, cat);
      previousSelections.value[cat].species = newSpecies;
      if (cat === "Sum") {
        availableLanduse.value = tree[newSpecies] || [];
      } else if (cat === "Ag") {
        availableLanduse.value = tree[selectWater.value]?.[newSpecies] || [];
      } else if (cat === "Ag Mgt") {
        availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value]?.[newSpecies] || [];
      } else if (cat === "Non-Ag") {
        availableLanduse.value = tree[newSpecies] || [];
      }
      if (!availableLanduse.value.includes(selectLanduse.value)) {
        selectLanduse.value = availableLanduse.value[0] || '';
      }
      const entry = mapRegister[metric]?.[cat];
      if (entry?.layerPrefix) await ensureComboLayer(entry.layerPrefix, buildCombo(cat, tree, true));
    });

    watch(selectLanduse, async (newLanduse) => {
      const cat = selectCategory.value;
      const metric = selectMetric.value;
      previousSelections.value[cat] = { ...(previousSelections.value[cat] || {}), landuse: newLanduse };
      const tree = getTree(metric, cat);
      const withSpecies = hasSpecies.value;
      const entry = mapRegister[metric]?.[cat];
      if (entry?.layerPrefix) await ensureComboLayer(entry.layerPrefix, buildCombo(cat, tree, withSpecies));
    });

    watch(selectBackend, async () => {
      if (selectMetric.value !== 'quality' || !selectCategory.value) return;
      await doCascade(selectCategory.value);
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableRegionLevels, selectRegionLevel,
      METRIC_LABELS, availableYears, availableMetrics, availableCategories,
      availableAgMgt, availableWater, availableSpecies, availableLanduse,
      selectMetric, selectCategory, selectAgMgt, selectWater, selectSpecies, selectLanduse,
      selectBackend, availableBackends, modelBackend,
      hasSpecies, speciesLabel, formatLanduse,
      selectMapData, selectChartData, selectMultiInput, selectMultiYAxis, gbf2MaskOverlay,
      dataLoaded, isLoadingData, isDrawerOpen, toggleDrawer,
    };
    const _fn = v => String(v).trim().replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    _state.mapFileName = computed(() =>
      [VIEW_NAME, selectMetric.value, selectCategory.value, selectAgMgt.value, selectWater.value, selectSpecies.value, selectLanduse.value, selectYear.value]
        .filter(Boolean).map(_fn).filter(Boolean).join('__')
    );
    window._debug[VIEW_NAME] = _state;
    return _state;
  },
  template: /*html*/`
    <div class="relative w-full h-screen">

      <!-- Region level tabs + Region selection dropdown -->
      <div class="absolute w-[262px] top-24 left-[20px] z-[9999] max-w-xs">
        <!-- Drawer-style region level tabs -->
        <div class="flex gap-1 ml-2 mb-0">
          <button v-for="lvl in availableRegionLevels" :key="lvl"
            @click="selectRegionLevel = lvl"
            class="px-2 py-0.5 text-[0.65rem] font-medium rounded-t-md border border-b-0 transition-colors"
            :class="selectRegionLevel === lvl
              ? 'bg-white/90 border-gray-300 text-sky-600'
              : 'bg-white/40 border-gray-200 text-gray-500 hover:bg-white/60'">
            {{ lvl === 'region_state' ? 'State' : 'NRM' }}
          </button>
        </div>
        <!-- Dropdown panel -->
        <div class="bg-white/70 rounded-lg shadow-lg">
          <filterable-dropdown
            :key="selectRegionLevel"
            :region-type="selectRegionLevel === 'region_state' ? 'STATE' : 'NRM'">
          </filterable-dropdown>
        </div>
      </div>

      <!-- Year slider -->
      <div class="absolute top-[240px] left-[20px] z-[1001] w-[262px] bg-white/70 p-2 rounded-lg items-center">
        <p class="text-[0.8rem]">Year: <strong>{{ selectYear }}</strong></p>
        <el-slider
          v-if="availableYears && availableYears.length > 0"
          v-model="yearIndex"
          size="small"
          :min="0"
          :max="availableYears.length - 1"
          :step="1"
          :format-tooltip="index => availableYears[index]"
          :show-stops="true"
          @input="(index) => { yearIndex = index; selectYear = availableYears[index]; }"
        />
      </div>

      <!-- Data selection controls container -->
      <div class="absolute top-[325px] left-[20px] w-[320px] z-[1001] flex flex-col space-y-3 bg-white/70 p-2 rounded-lg">

        <!-- Metric buttons (always visible) -->
        <div class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Metric:</span>
          <button v-for="(val, key) in availableMetrics" :key="key"
            @click="selectMetric = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectMetric === val}">
            {{ METRIC_LABELS[val] || val }}
          </button>
        </div>

        <!-- Backend layer buttons (quality metric only) -->
        <div v-if="dataLoaded && selectMetric === 'quality' && availableBackends.length > 1" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Backend:</span>
          <button v-for="val in availableBackends" :key="val"
            @click="selectBackend = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1 border border-transparent"
            :class="{
              'bg-sky-500 text-white': selectBackend === val,
              '!border-dashed !border-sky-700': val === modelBackend,
            }">
            {{ val }}
          </button>
        </div>

        <!-- Category buttons -->
        <div class="flex space-x-1">
          <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
          <button v-for="(val, key) in availableCategories" :key="key"
            @click="selectCategory = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
            :class="{'bg-sky-500 text-white': selectCategory === val}">
            {{ val }}
          </button>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category) -->
        <div v-if="dataLoaded && selectCategory === 'Ag Mgt' && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
          <button v-for="(val, key) in availableAgMgt" :key="key"
            @click="selectAgMgt = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectAgMgt === val}">
            {{ val }}
          </button>
        </div>

        <!-- Water options (Ag and Ag Mgt only) -->
        <div v-if="selectCategory !== 'Non-Ag' && selectCategory !== 'Sum' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- Landuse options (for Sum: shows Type dimension values) -->
        <div v-if="dataLoaded" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">{{ selectCategory === 'Sum' ? 'Type:' : 'Landuse:' }}</span>
          <button v-for="(val, key) in availableLanduse" :key="key"
            @click="selectLanduse = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectLanduse === val}">
            {{ formatLanduse(val) }}
          </button>
        </div>

        <!-- Backend legend note -->
        <div v-if="dataLoaded && selectMetric === 'quality' && availableBackends.length > 1"
          style="font-size:0.6rem; line-height:1.3;" class="flex items-center gap-1 text-gray-500">
          <span class="inline-block px-1 py-0.5 rounded border border-dashed border-sky-700 text-[0.6rem] leading-none">abc</span>
          <span>= model-selected backend</span>
        </div>
      </div>

      <!-- Map container with slide-out chart drawer -->
      <div style="position: relative; width: 100%; height: 100%; overflow: hidden;">

        <!-- Species / VegGroup / Community options (GBF3/GBF4/GBF8 only, including Sum category) — floating bottom-right scroll panel -->
        <div v-if="dataLoaded && availableSpecies.length > 0"
          class="absolute bottom-[20px] right-[20px] z-[1001] w-[280px] max-h-[260px] bg-white/85 rounded-lg shadow-md p-2 flex flex-col"
          :class="{ 'right-[440px]': isDrawerOpen }"
          style="transition: right 0.3s ease;">
          <div class="text-[0.8rem] font-medium mb-1 flex-shrink-0">{{ speciesLabel }}</div>
          <div class="flex flex-wrap gap-1 overflow-y-auto pr-1">
            <button v-for="(val, key) in availableSpecies" :key="key"
              @click="selectSpecies = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1 text-left"
              :class="{'bg-sky-500 text-white': selectSpecies === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Loading overlay shown while lazy-loading a new map file -->
        <div v-if="isLoadingData"
          class="absolute inset-0 z-[2000] flex items-center justify-center bg-white/60 backdrop-blur-sm">
          <div class="flex flex-col items-center gap-2 text-gray-600 text-sm font-medium">
            <svg class="animate-spin h-8 w-8 text-sky-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
            Loading map data…
          </div>
        </div>

        <!-- Map component takes full space -->
        <regions-map
          :mapData="selectMapData"
          :file-name="mapFileName"
          :region-type="selectRegionLevel === 'region_state' ? 'STATE' : 'NRM'"
          :overlayGeoJSON="gbf2MaskOverlay"
          :show-legend="!isDrawerOpen"
          style="width: 100%; height: 100%;">
        </regions-map>

        <!-- Drawer toggle button -->
        <button
          @click="toggleDrawer"
          class="absolute top-5 z-[1001] p-2.5 bg-white border border-gray-300 rounded cursor-pointer transition-all duration-300 ease-in-out"
          :class="isDrawerOpen ? 'right-[420px]' : 'right-5'">
          {{ isDrawerOpen ? '→' : '←' }}
        </button>

        <!-- Chart drawer positioned relative to map -->
        <div
          :style="{
            position: 'absolute',
            height: '50px',
            top: '10px',
            bottom: '10px',
            right: isDrawerOpen ? '0px' : '-100%',
            width: '66.666%',
            background: 'transparent',
            transition: 'right 0.3s ease',
            zIndex: 1000,
            padding: '60px 20px 20px 20px',
            boxSizing: 'border-box'
          }">
          <chart-container
            :chartData="selectChartData"
            :multiInput="selectMultiInput"
            :multiYAxis="selectMultiYAxis"
            :draggable="true"
            :zoomable="true"
            style="width: 100%; height: 200px;">
          </chart-container>
        </div>
      </div>

    </div>
  `,
};
