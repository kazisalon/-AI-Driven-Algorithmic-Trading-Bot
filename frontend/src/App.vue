<template>
	<v-app>
		<v-main>
			<v-container fluid>
				<v-row>
					<v-col cols="12" md="4">
						<trading-controls
							@start-training="handleTraining"
							@start-backtest="handleBacktest"
							:is-loading="isLoading"
						/>
						<trading-stats :performance-data="performanceData" class="mt-4" />
					</v-col>
					<v-col cols="12" md="8">
						<trading-chart :chart-data="chartData" :is-loading="isLoading" />
					</v-col>
				</v-row>
				<v-row>
					<v-col cols="12">
						<trading-history :trading-history="tradingHistory" />
					</v-col>
				</v-row>
			</v-container>
		</v-main>
	</v-app>
</template>
