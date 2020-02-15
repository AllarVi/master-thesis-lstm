from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class Plots:
    @staticmethod
    def neural_network_patents_over_time(data):
        plot_data = data.copy()

        fig, ax = plt.subplots()

        plt.style.use('fivethirtyeight')

        plot_data['year-month'] = [
            datetime(year, month, 1) for year, month in zip(
                plot_data['patent_date'].dt.year, plot_data['patent_date'].dt.month)
        ]

        monthly = plot_data.groupby('year-month')['patent_number'].count().reset_index()
        monthly.set_index('year-month')['patent_number'].plot(figsize=(16, 8))

        ax.set_ylabel('Number of Patents')
        ax.set_xlabel('Date')
        ax.set_title('Neural Network Patents over Time')
        fig.savefig('./plots/neural-network-patents-over-time.png')

    @staticmethod
    def neural_network_patents_by_year(data):
        plot_data = data.copy()

        fig, ax = plt.subplots()

        plt.style.use('fivethirtyeight')

        plot_data['year-month'] = [
            datetime(year, month, 1) for year, month in zip(
                plot_data['patent_date'].dt.year, plot_data['patent_date'].dt.month)
        ]

        monthly = plot_data.groupby('year-month')['patent_number'].count().reset_index()
        monthly.groupby(monthly['year-month'].dt.year)['patent_number'].sum().plot.bar(
            color='red', edgecolor='k', figsize=(12, 6))

        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Patents')
        ax.set_title('Neural Network Patents by Year')
        fig.savefig('./plots/neural-network-patents-by-year.png')
