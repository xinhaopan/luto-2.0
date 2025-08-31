// Ranking Cards Element
// This component displays ranking cards with progress indicators for various metrics

window.RankingCards = {
  props: {
    rankingData: {
      type: Object,
      required: true
    },
    selectRegion: {
      type: String,
      required: true
    },
    selectYear: {
      type: Number,
      required: true
    },
  },


  template: `
    <div class="flex flex-wrap gap-4 justify-between h-[230px]">
      <!-- Economics Card -->
      <div class="flex-1 rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#e6ba7f] to-[#eacca2]" >
        <h4 class="text-white text-center text-lg mb-2">Economics</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData['Economics'][selectRegion]['Total']['value'][selectYear] }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Australian Dollar</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Cost</span>
            <span>{{ rankingData['Economics'][selectRegion]['Cost']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Revenue</span>
            <span>{{ rankingData['Economics'][selectRegion]['Revenue']['value'][selectYear] }}</span>
          </div>
        </div>
      </div>
      
      <!-- Area Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-blue-400 to-cyan-400" >
        <h4 class="text-white text-center text-lg mb-2">Area</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData['Area'][selectRegion]['Total']['value'][selectYear] }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Hectares</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ rankingData['Area'][selectRegion]['Agricultural Landuse']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ rankingData['Area'][selectRegion]['Agricultural Management']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ rankingData['Area'][selectRegion]['Non-Agricultural Landuse']['value'][selectYear] }}</span>
          </div>
        </div>
      </div>
      
      <!-- GHG Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-green-400 to-green-500" >
        <h4 class="text-white text-center text-lg mb-2">GHG Impact</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData['GHG'][selectRegion]['Total']['value'][selectYear] }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">tCO2e</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Emissions</span>
            <span>{{ rankingData['GHG'][selectRegion]['GHG emissions']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Reduction</span>
            <span>{{ rankingData['GHG'][selectRegion]['GHG sequestrations']['value'][selectYear] }}</span>
          </div>
        </div>
      </div>
      
      <!-- Water Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-rose-400 to-amber-300" >
        <h4 class="text-white text-center text-lg mb-2">Water Usage</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData['Water'][selectRegion]['Total']['value'][selectYear] }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Megaliters</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ rankingData['Water'][selectRegion]['Agricultural Landuse']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ rankingData['Water'][selectRegion]['Agricultural Management']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>NonAg</span>
            <span>{{ rankingData['Water'][selectRegion]['Non-Agricultural Landuse']['value'][selectYear] }}</span>
          </div>
        </div>
      </div>
      
      <!-- Biodiversity Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#918be9] to-[#e2cbfa]" >
        <h4 class="text-white text-center text-lg mb-2">Biodiversity</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData['Biodiversity'][selectRegion]['Total']['value'][selectYear] }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Priority Weighted Hectares</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ rankingData['Biodiversity'][selectRegion]['Agricultural Landuse']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ rankingData['Biodiversity'][selectRegion]['Agricultural Management']['value'][selectYear] }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ rankingData['Biodiversity'][selectRegion]['Non-Agricultural land-use']['value'][selectYear] }}</span>
          </div>
        </div>
      </div>
    </div>
  `,
};