<template>
	<v-card>
		<v-card-title>Performance Statistics</v-card-title>
		<v-card-text>
			<v-list v-if="performanceData">
				<v-list-item>
					<v-list-item-title>Initial Balance</v-list-item-title>
					<v-list-item-subtitle
						>${{
							formatNumber(performanceData.initialBalance)
						}}</v-list-item-subtitle
					>
				</v-list-item>
				<v-list-item>
					<v-list-item-title>Final Balance</v-list-item-title>
					<v-list-item-subtitle
						>${{
							formatNumber(performanceData.finalBalance)
						}}</v-list-item-subtitle
					>
				</v-list-item>
				<v-list-item>
					<v-list-item-title>Total Return</v-list-item-title>
					<v-list-item-subtitle :class="getReturnClass">
						{{ formatNumber(performanceData.totalReturn) }}%
					</v-list-item-subtitle>
				</v-list-item>
			</v-list>
			<div v-else class="text-center py-4">No performance data available</div>
		</v-card-text>
	</v-card>
</template>
