import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI dependencies
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
from datetime import datetime
import csv

from oeneastdpt import NULL_SENTINEL


class SimulationStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mass_transferred = []
        self.movements = []
        self.replications = []
        self.communications = []
        self.reorientations = []
        self.population_size = []
        self.avg_nodes = []
        self.avg_synapses = []
        self.lowest_nodes = []
        self.highest_nodes = []
        self.lowest_synapses = []
        self.highest_synapses = []
        self.ticks = []

        # Temporary storage for current tick
        self.current_tick_data = {
            'mass_transferred': 0,
            'movements': 0,
            'replications': 0,
            'communications': 0,
            'reorientations': 0
        }

    def update(self, myxomatrix, tick):
        population_size = sum(len(species[2]) for species in myxomatrix.species_manager.species)
        total_nodes = 0
        total_synapses = 0
        min_nodes = float('inf')
        max_nodes = 0
        min_synapses = float('inf')
        max_synapses = 0

        for species in myxomatrix.species_manager.species:
            for agent in species[2]:
                nodes = sum(1 for neuron_type in agent.genetic_representation.neuron_types
                                     if agent.genetic_representation._decode_neuron_type(neuron_type) == 'hidden')
                synapses = len([syn for syn in agent.genetic_representation.synapse_pre if syn != NULL_SENTINEL])

                min_nodes = min(min_nodes, nodes)
                max_nodes = max(max_nodes, nodes)
                total_nodes += nodes

                min_synapses = min(min_synapses, synapses)
                max_synapses = max(max_synapses, synapses)
                total_synapses += synapses

        self.population_size.append(population_size)
        self.avg_nodes.append(total_nodes / population_size if population_size > 0 else 0)
        self.avg_synapses.append(total_synapses / population_size if population_size > 0 else 0)
        self.lowest_nodes.append(min_nodes if min_nodes != float('inf') else 0)
        self.highest_nodes.append(max_nodes)
        self.lowest_synapses.append(min_synapses if min_synapses != float('inf') else 0)
        self.highest_synapses.append(max_synapses)
        self.ticks.append(tick)

        # Add current tick data to respective lists
        self.mass_transferred.append(self.current_tick_data['mass_transferred'])
        self.movements.append(self.current_tick_data['movements'])
        self.replications.append(self.current_tick_data['replications'])
        self.communications.append(self.current_tick_data['communications'])
        self.reorientations.append(self.current_tick_data['reorientations'])

        # Reset current tick data
        self.current_tick_data = {key: 0 for key in self.current_tick_data}

    def record_action(self, action_type, value=1):
        if action_type in self.current_tick_data:
            self.current_tick_data[action_type] += value

    def get_stats(self):
        return {
            'ticks': self.ticks,
            'mass_transferred': self.mass_transferred,
            'movements': self.movements,
            'replications': self.replications,
            'communications': self.communications,
            'reorientations': self.reorientations,
            'population_size': self.population_size,
            'avg_nodes': self.avg_nodes,
            'avg_synapses': self.avg_synapses,
            'lowest_nodes': self.lowest_nodes,
            'highest_nodes': self.highest_nodes,
            'lowest_synapses': self.lowest_synapses,
            'highest_synapses': self.highest_synapses
        }

    def print_current_stats(self, tick):
        print(f"Tick {tick}:")
        print(f"  Mass Transferred: {self.mass_transferred[-1]:.2f}")
        print(f"  Movements: {self.movements[-1]}")
        print(f"  Replications: {self.replications[-1]}")
        print(f"  Communications: {self.communications[-1]}")
        print(f"  Reorientations: {self.reorientations[-1]}")
        print(f"  Population Size: {self.population_size[-1]}")
        print(f"  Avg Nodes: {self.avg_nodes[-1]:.2f}")
        print(f"  Avg Synapses: {self.avg_synapses[-1]:.2f}")
        print(f"  Nodes Range: {self.lowest_nodes[-1]} - {self.highest_nodes[-1]}")
        print(f"  Synapses Range: {self.lowest_synapses[-1]} - {self.highest_synapses[-1]}")


class SimulationDataAnalyser:
    def __init__(self, simulation_stats=None):
        self.simulation_stats = simulation_stats
        self.base_folder = "simulation_results"
        self.ensure_base_folder_exists()
        self.current_run_folder = None

    def ensure_base_folder_exists(self):
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

    def create_run_folder(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_folder = os.path.join(self.base_folder, f"run_{timestamp}")
        os.makedirs(self.current_run_folder)

    def save_data(self):
        if not self.current_run_folder:
            self.create_run_folder()

        filename = os.path.join(self.current_run_folder, "simulation_data.csv")

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Tick', 'Population', 'Avg_Nodes', 'Avg_Synapses', 'Mass_Transferred',
                             'Movements', 'Replications', 'Communications', 'Reorientations'])

            stats = self.simulation_stats.get_stats()
            for i in range(len(stats['ticks'])):
                writer.writerow([
                    stats['ticks'][i],
                    stats['population_size'][i],
                    stats['avg_nodes'][i],
                    stats['avg_synapses'][i],
                    stats['mass_transferred'][i],
                    stats['movements'][i],
                    stats['replications'][i],
                    stats['communications'][i],
                    stats['reorientations'][i]
                ])

        print(f"Data saved to {filename}")
        return filename

    def create_single_run_plot(self, data, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data, label='Value', color='blue')
        ax.set_title(title)
        ax.set_xlabel('Ticks')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        return fig

    def generate_and_save_single_run_plots(self):
        if not self.current_run_folder:
            self.create_run_folder()

        stats = self.simulation_stats.get_stats()

        plots_to_generate = [
            ('population_size', 'Population Size', 'Agent Count'),
            ('avg_nodes', 'Average Number of Neurons', 'Neuron Count'),
            ('avg_synapses', 'Average Number of Synapses', 'Synapse Count'),
            ('mass_transferred', 'Mass Transferred', 'Mass Units'),
            ('movements', 'Number of Movements', 'Movement Count'),
            ('replications', 'Number of Replications', 'Replication Count'),
            ('communications', 'Number of Communications', 'Communication Count'),
            ('reorientations', 'Number of Reorientations', 'Reorientation Count')
        ]

        for stat_key, title, ylabel in plots_to_generate:
            fig = self.create_single_run_plot(stats[stat_key], title, ylabel)
            filename = os.path.join(self.current_run_folder, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(filename)
            plt.close(fig)
            print(f"Plot saved to {filename}")

    def analyse_single_run(self):
        self.create_run_folder()
        self.save_data()
        self.generate_and_save_single_run_plots()
        print("Single run data analysis and plotting completed.")

    def load_all_data(self):
        data = {}
        for root, dirs, files in os.walk(self.base_folder):
            for file in files:
                if file == "simulation_data.csv":
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            for key, value in row.items():
                                if key not in data:
                                    data[key] = []
                                data[key].append(float(value))
        return data

    def create_multi_run_plot(self, data, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))

        data_array = np.array(data)
        ticks = range(data_array.shape[1])

        best_run = np.min(data_array, axis=0)
        worst_run = np.max(data_array, axis=0)
        mean_run = np.mean(data_array, axis=0)
        std_dev = np.std(data_array, axis=0)
        ci95 = stats.t.interval(0.95, len(data_array) - 1, loc=mean_run, scale=std_dev / np.sqrt(len(data_array)))

        ax.plot(ticks, best_run, label='Best Run', color='green')
        ax.plot(ticks, worst_run, label='Worst Run', color='red')
        ax.plot(ticks, mean_run, label='Mean', color='blue')
        ax.fill_between(ticks, mean_run - std_dev, mean_run + std_dev, color='blue', alpha=0.2, label='1 SD')
        ax.fill_between(ticks, ci95[0], ci95[1], color='yellow', alpha=0.3, label='95% CI')

        ax.set_title(title)
        ax.set_xlabel('Ticks')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        return fig

    def analyse_all_runs(self):
        data = self.load_all_data()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        multi_run_folder = os.path.join(self.base_folder, f"multi_run_analysis_{timestamp}")
        os.makedirs(multi_run_folder)

        plots_to_generate = [
            ('Population', 'Population Size', 'Agent Count'),
            ('Avg_Nodes', 'Average Number of Neurons', 'Neuron Count'),
            ('Avg_Synapses', 'Average Number of Synapses', 'Synapse Count'),
            ('Mass_Transferred', 'Mass Transferred', 'Mass Units'),
            ('Movements', 'Number of Movements', 'Movement Count'),
            ('Replications', 'Number of Replications', 'Replication Count'),
            ('Communications', 'Number of Communications', 'Communication Count'),
            ('Reorientations', 'Number of Reorientations', 'Reorientation Count')
        ]

        for stat_key, title, ylabel in plots_to_generate:
            if stat_key in data:
                fig = self.create_multi_run_plot(data[stat_key], title, ylabel)
                filename = os.path.join(multi_run_folder, f"multi_run_{title.lower().replace(' ', '_')}.png")
                plt.savefig(filename)
                plt.close(fig)
                print(f"Multi-run plot saved to {filename}")

        # Save the combined data
        combined_data_file = os.path.join(multi_run_folder, "combined_simulation_data.csv")
        with open(combined_data_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.keys())
            for i in range(len(next(iter(data.values())))):
                writer.writerow([data[key][i] for key in data.keys()])

        print(f"Combined data saved to {combined_data_file}")
        print("Multi-run data analysis and plotting completed.")