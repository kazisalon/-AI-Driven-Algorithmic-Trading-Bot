TradingChart.vue
<template>
	<v-card>
		<v-card-title>Performance Chart</v-card-title>
		<v-card-text>
			<div
				v-if="isLoading"
				class="d-flex justify-center align-center"
				style="height: 400px"
			>
				<v-progress-circular indeterminate />
			</div>
			<div v-else-if="chartData">
				<Line :data="chartData" :options="chartOptions" />
			</div>
			<div v-else class="text-center py-8">No data available</div>
		</v-card-text>
	</v-card>
</template>

<script>
import { Line } from "vue-chartjs";
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend,
} from "chart.js";

ChartJS.register(
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend
);

export default {
	name: "TradingChart",
	components: { Line },
	props: {
		chartData: Object,
		isLoading: Boolean,
	},
	setup() {
		const chartOptions = {
			responsive: true,
			maintainAspectRatio: false,
			scales: {
				y: {
					beginAtZero: true,
					title: {
						display: true,
						text: "Portfolio Value ($)",
					},
				},
				x: {
					title: {
						display: true,
						text: "Date",
					},
				},
			},
		};

		return { chartOptions };
	},
};
</script>
