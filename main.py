import random
import pygame
import pygame_gui
import gc
import os
import threading
import time
import shutil
import glob
from videorecorder import VideoRecorder
from oeneastdpt import genome_pool, OENEASTDPTNetwork, crossover, compatible, NULL_SENTINEL
from interface import EnvironmentSetupInterface
from dataanalysis import SimulationStats, SimulationDataAnalyser

# Constants
CELL_SIZE = 6
SETUP_WINDOW_WIDTH = 1400
SETUP_WINDOW_HEIGHT = 860
FPS = 30

# Hyperparameters
CONSUMPTION_RATE = 0.5
BASAL_METABOLIC_COST = 0.002
NEURON_COST = 0.0002
SYNAPSE_COST = 0.00002
REPLICATION_SURVIVABILITY_SCALAR = 6
REPLICATION_COST = 0.1
MOVEMENT_COST = 0.05
REPLICATION_COOLDOWN = 6
TRANSFER_RATE = 0.2
CROSSOVER_PROBABILITY = 0.2
MUTATION_PROBABILITY = 0.08
COMPATIBILITY_THRESHOLD = 0.16
NUM_COMMUNICATION_SIGNALS = 4
PROGENITOR_UPDATE_THRESHOLD = 500

# Video progress globals
video_progress_message = ""
video_progress_percentage = 0

# Global seed variable
current_seed = None


def set_global_seed(seed):
    """Set the global seed for all random operations."""
    global current_seed
    current_seed = seed
    random.seed(seed)


def get_current_seed():
    """Get the current seed."""
    return current_seed


class Wall:
    def __init__(self, cell):
        self.cell = cell


class GrowthRegion:
    def __init__(self, x_min, y_min, x_max, y_max, num_plants, tau_min, tau_max, r_min, r_max):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.num_plants = num_plants
        self.tau_min, self.tau_max = tau_min, tau_max
        self.r_min, self.r_max = r_min, r_max


class Plant:  # UPDATE DESCRIPTION IN THE PAPER
    def __init__(self, cell, growth_region):
        self.cell = cell
        self.growth_region = growth_region
        self.growth_timer = 0
        self.had_agent_last_tick = False
        self.grow_resource()

    def update(self, environment):
        if self.growth_timer > 0:
            self.growth_timer -= 1
        elif self.cell.resource_level == 0:
            if not self.cell.agent and self.had_agent_last_tick:
                new_cell = self.find_empty_adjacent_cell(environment)
                if new_cell:
                    environment.move_plant(self, new_cell)
                self.reset_growth_timer()
            elif self.growth_timer == 0 and not self.cell.agent:
                self.grow_resource()

        self.had_agent_last_tick = bool(self.cell.agent)

    def reset_growth_timer(self):
        self.growth_timer = random.randint(self.growth_region.tau_min, self.growth_region.tau_max)

    def grow_resource(self):
        self.cell.resource_level = random.uniform(self.growth_region.r_min, self.growth_region.r_max)

    def find_empty_adjacent_cell(self, environment):
        adjacent_cells = [self.cell.north, self.cell.south, self.cell.east, self.cell.west]
        empty_cells = [cell for cell in adjacent_cells
                       if environment.is_within_region(cell, self.growth_region)
                       and not cell.plant and not cell.wall]
        return random.choice(empty_cells) if empty_cells else None


class Agent:
    ORIENTATIONS = ['Northward', 'Eastward', 'Southward', 'Westward']

    def __init__(self, genetic_representation, cell, orientation, mass,
                 num_communication_signals=NUM_COMMUNICATION_SIGNALS):
        self.cell = cell
        self.orientation = orientation
        self.mass = mass
        self.genetic_representation = genetic_representation
        self.control_network = OENEASTDPTNetwork(genetic_representation)
        self.replication_cooldown = REPLICATION_COOLDOWN
        self.num_communication_signals = num_communication_signals

        self.communication_inputs = {
            'ahead': [0] * num_communication_signals,
            'left': [0] * num_communication_signals,
            'behind': [0] * num_communication_signals,
            'right': [0] * num_communication_signals
        }
        self.communication_outputs = [False] * (4 * num_communication_signals)

        self.reorientation_outputs = [False, False, False]  # Left, Back, Right
        self.replication_outputs = [False, False, False, False]  # Ahead, Left, Back, Right
        self.mass_transfer_outputs = [False, False, False, False]  # Ahead, Left, Back, Right

        self.species_colour = (0, 0, 0)

        self.alive = True

        self.north_neighbour = None
        self.south_neighbour = None
        self.east_neighbour = None
        self.west_neighbour = None
        self.update_neighbour_compatibility()

        self.metabolic_cost = (
                BASAL_METABOLIC_COST +
                NEURON_COST * len([ntype for ntype in genetic_representation.neuron_types if ntype == 'hidden']) +
                SYNAPSE_COST * len([syn for syn in genetic_representation.synapse_pre if syn != NULL_SENTINEL])
        )

        self.replication_threshold = (
                REPLICATION_COST +
                (2 * self.metabolic_cost) * REPLICATION_SURVIVABILITY_SCALAR
        )

        self.stats = None

    def update_neighbour_compatibility(self):
        directions = ['north', 'south', 'east', 'west']
        for direction in directions:
            neighbour_cell = getattr(self.cell, direction)
            neighbour_agent = neighbour_cell.agent
            if neighbour_agent:
                is_compatible = compatible(self.genetic_representation, neighbour_agent.genetic_representation,
                                           COMPATIBILITY_THRESHOLD)
                setattr(self, f"{direction}_neighbour", neighbour_agent if is_compatible else None)
                opposite = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}[direction]
                setattr(neighbour_agent, f"{opposite}_neighbour", self if is_compatible else None)

    def update(self):
        self.consume_resources()
        self.metabolise()
        self.process_neural_network()
        self.reorient()
        self.transfer_mass()
        new_agent = self.replicate()
        self.communicate()

        if self.mass <= 0:
            self.kill()

        return new_agent

    def consume_resources(self):
        consumed = min(self.cell.resource_level, CONSUMPTION_RATE, 1 - self.mass)
        self.mass += consumed
        self.cell.resource_level -= consumed

    def metabolise(self):
        self.mass = max(0, self.mass - self.metabolic_cost)

    def process_neural_network(self):
        inputs = self.get_neural_inputs()
        outputs = self.control_network.process(inputs)
        self.interpret_outputs(outputs)

    def get_neural_inputs(self):
        inputs = [
            self.cell.resource_level,
            1 if self.get_relative_neighbour('ahead') else 0,
            1 if self.get_relative_neighbour('left') else 0,
            1 if self.get_relative_neighbour('behind') else 0,
            1 if self.get_relative_neighbour('right') else 0,
            1 if self.get_relative_cell('ahead').wall else 0,
            1 if self.get_relative_cell('left').wall else 0,
            1 if self.get_relative_cell('behind').wall else 0,
            1 if self.get_relative_cell('right').wall else 0,
        ]
        for direction in ['ahead', 'left', 'behind', 'right']:
            inputs.extend(self.communication_inputs[direction])
        return inputs

    def interpret_outputs(self, outputs):
        self.reorientation_outputs = outputs[:3]
        self.replication_outputs = outputs[3:7]
        self.mass_transfer_outputs = outputs[7:11]
        self.communication_outputs = outputs[11:]

    def reorient(self):
        true_outputs = [i for i, output in enumerate(self.reorientation_outputs) if output]

        if true_outputs:
            # Randomly select one of the true outputs
            direction = random.choice(true_outputs)

            self.orientation = self.ORIENTATIONS[(self.ORIENTATIONS.index(self.orientation) + direction + 1) % 4]

            if self.stats:
                self.stats.record_action('reorientations')

    def replicate(self):
        new_agent = None
        if self.mass > self.replication_threshold and self.replication_cooldown == 0:
            valid_directions = [i for i, signal in enumerate(self.replication_outputs) if signal]
            valid_directions = [i for i in valid_directions
                                if not (c := self.get_relative_cell(['ahead', 'right', 'behind', 'left'][i])).wall
                                and not c.agent]
            if valid_directions:
                i = random.choice(valid_directions)
                direction = ['ahead', 'right', 'behind', 'left'][i]
                target_cell = self.get_relative_cell(direction)
                new_orientation = self.ORIENTATIONS[(self.ORIENTATIONS.index(self.orientation) + i) % 4]
                new_mass = (self.mass - REPLICATION_COST) / 2
                new_genome = self.genetic_representation.copy(genome_pool)
                if random.random() < MUTATION_PROBABILITY:
                    new_genome.mutate()
                new_agent = target_cell.initialise_agent(new_genome, new_orientation,
                                                         new_mass, self.num_communication_signals)
                self.mass = new_mass
                self.replication_cooldown = REPLICATION_COOLDOWN
                new_agent.replication_cooldown = REPLICATION_COOLDOWN

                if new_agent:
                    if self.stats:
                        self.stats.record_action('replications')

        if self.replication_cooldown > 0:
            self.replication_cooldown -= 1
        return new_agent

    def transfer_mass(self):
        for i, signal in enumerate(self.mass_transfer_outputs):
            if signal:
                direction = ['ahead', 'right', 'behind', 'left'][i]
                target_agent = self.get_relative_neighbour(direction)
                if target_agent:
                    transferred = min(self.mass, TRANSFER_RATE, 1 - target_agent.mass)
                    self.mass -= transferred
                    target_agent.mass += transferred
                    if random.random() < CROSSOVER_PROBABILITY:
                        self.genetic_representation, target_agent.genetic_representation = crossover(
                            self.genetic_representation, target_agent.genetic_representation)
                    if self.stats:
                        self.stats.record_action('mass_transferred', transferred)

    def communicate(self):  # OPPOSITE IS WRONG
        for i, direction in enumerate(['ahead', 'left', 'behind', 'right']):
            target_agent = self.get_relative_neighbour(direction)
            if target_agent:
                opposite_direction = {
                    'ahead': 'behind',
                    'left': 'right',
                    'behind': 'ahead',
                    'right': 'left'
                }[direction]
                start_index = i * self.num_communication_signals
                signals = self.communication_outputs[start_index:start_index + self.num_communication_signals]
                target_agent.receive_communication(signals, opposite_direction)
                if self.stats:
                    self.stats.record_action('communications')

    def receive_communication(self, signals, direction):
        for i, signal in enumerate(signals):
            self.communication_inputs[direction][i] = 1 if signal else 0

    def kill(self):
        self.alive = False
        self.cell.agent = None

        self.dissociate_neighbours_with_self()

        self.cell = None
        self.north_neighbour = None
        self.south_neighbour = None
        self.east_neighbour = None
        self.west_neighbour = None

        self.control_network.clear()
        genome_pool.release(self.genetic_representation)
        # self.genetic_representation = None

    def dissociate_neighbours_with_self(self):
        directions = ['north', 'south', 'east', 'west']
        opposites = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        for direction in directions:
            neighbour_cell = getattr(self.cell, direction)
            if neighbour_cell and neighbour_cell.agent:
                opposite = opposites[direction]
                setattr(neighbour_cell.agent, f"{opposite}_neighbour", None)

    def get_relative_cell(self, direction):
        if self.orientation == 'Northward':
            return {
                'ahead': self.cell.north,
                'left': self.cell.west,
                'behind': self.cell.south,
                'right': self.cell.east
            }[direction]
        elif self.orientation == 'Eastward':
            return {
                'ahead': self.cell.east,
                'left': self.cell.north,
                'behind': self.cell.west,
                'right': self.cell.south
            }[direction]
        elif self.orientation == 'Southward':
            return {
                'ahead': self.cell.south,
                'left': self.cell.east,
                'behind': self.cell.north,
                'right': self.cell.west
            }[direction]
        elif self.orientation == 'Westward':
            return {
                'ahead': self.cell.west,
                'left': self.cell.south,
                'behind': self.cell.east,
                'right': self.cell.north
            }[direction]

    def get_relative_neighbour(self, direction):
        if self.orientation == 'Northward':
            return {
                'ahead': self.north_neighbour,
                'left': self.west_neighbour,
                'behind': self.south_neighbour,
                'right': self.east_neighbour
            }[direction]
        elif self.orientation == 'Eastward':
            return {
                'ahead': self.east_neighbour,
                'left': self.north_neighbour,
                'behind': self.west_neighbour,
                'right': self.south_neighbour
            }[direction]
        elif self.orientation == 'Southward':
            return {
                'ahead': self.south_neighbour,
                'left': self.east_neighbour,
                'behind': self.north_neighbour,
                'right': self.west_neighbour
            }[direction]
        elif self.orientation == 'Westward':
            return {
                'ahead': self.west_neighbour,
                'left': self.south_neighbour,
                'behind': self.east_neighbour,
                'right': self.north_neighbour
            }[direction]


class MovingAgent(Agent):
    def __init__(self, genetic_representation, cell, orientation, mass,
                 num_communication_signals=NUM_COMMUNICATION_SIGNALS):
        super().__init__(genetic_representation, cell, orientation, mass, num_communication_signals)
        self.movement_outputs = [False, False, False, False]  # Ahead, Left, Back, Right
        self.action_cooldown = 0  # Combined cooldown for both movement and replication

    def update(self):
        self.consume_resources()
        self.metabolise()
        self.process_neural_network()
        self.reorient()
        self.transfer_mass()
        new_agent = self.move_or_replicate()
        self.communicate()

        if self.mass <= 0:
            self.kill()

        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        return new_agent

    def process_neural_network(self):
        inputs = self.get_neural_inputs()
        outputs = self.control_network.process(inputs)
        self.interpret_outputs(outputs)

    def interpret_outputs(self, outputs):
        self.reorientation_outputs = outputs[:3]
        self.replication_outputs = outputs[3:7]
        self.mass_transfer_outputs = outputs[7:11]
        self.movement_outputs = outputs[11:15]
        self.communication_outputs = outputs[15:]

    def move_or_replicate(self):
        if self.action_cooldown == 0:
            valid_actions = []
            for i, (move_signal, replicate_signal) in enumerate(zip(self.movement_outputs, self.replication_outputs)):
                target_cell = self.get_relative_cell(['ahead', 'right', 'behind', 'left'][i])
                if not target_cell.wall and not target_cell.agent:
                    if move_signal and self.mass > MOVEMENT_COST:
                        valid_actions.append(('move', i))
                    if replicate_signal and self.mass > self.replication_threshold:
                        valid_actions.append(('replicate', i))

            if valid_actions:
                action, direction = random.choice(valid_actions)
                if action == 'move':
                    return self.perform_move(direction)
                else:
                    return self.perform_replicate(direction)

        return None

    def perform_move(self, direction):
        target_cell = self.get_relative_cell(['ahead', 'right', 'behind', 'left'][direction])

        # Remove agent from current cell
        self.dissociate_neighbours_with_self()
        self.cell.agent = None

        # Update agent's cell and orientation
        self.cell = target_cell
        self.orientation = self.ORIENTATIONS[(self.ORIENTATIONS.index(self.orientation) + direction) % 4]

        # Place agent in new cell
        target_cell.agent = self

        # Apply movement cost
        self.mass -= MOVEMENT_COST

        # Set cooldown
        self.action_cooldown = REPLICATION_COOLDOWN + 1

        # Update neighbor compatibility
        self.update_neighbour_compatibility()

        # Update stats
        if self.stats:
            self.stats.record_action('movements')

        return None

    def perform_replicate(self, direction):
        target_cell = self.get_relative_cell(['ahead', 'right', 'behind', 'left'][direction])
        new_orientation = self.ORIENTATIONS[(self.ORIENTATIONS.index(self.orientation) + direction) % 4]
        new_mass = (self.mass - REPLICATION_COST) / 2
        new_genome = self.genetic_representation.copy(genome_pool)
        if random.random() < MUTATION_PROBABILITY:
            new_genome.mutate()

        # Create a new MovingAgent instead of a regular Agent
        new_agent = MovingAgent(new_genome, target_cell, new_orientation, new_mass, self.num_communication_signals)
        target_cell.agent = new_agent  # Set the new agent in the target cell

        self.mass = new_mass
        self.action_cooldown = REPLICATION_COOLDOWN + 1
        new_agent.action_cooldown = REPLICATION_COOLDOWN

        if new_agent:
            if self.stats:
                self.stats.record_action('replications')

        return new_agent


def random_colour():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


class SpeciesManager:
    def __init__(self, update_threshold):
        self.species = []  # List of tuples (g_i, c_i, A_i, n_i)
        self.update_threshold = update_threshold

    def process(self, agent):
        for i, (g_i, c_i, A_i, n_i) in enumerate(self.species):
            if compatible(agent.genetic_representation, g_i, COMPATIBILITY_THRESHOLD):
                A_i.append(agent)
                self.species[i] = (g_i, c_i, A_i, n_i + 1)
                agent.species_colour = c_i

                if n_i + 1 >= self.update_threshold:
                    self.species[i] = (agent.genetic_representation, c_i, A_i, n_i + 1)

                return

        # If no compatible species found, create a new entry
        if not self.species:  # first agent ever, force yellow
            new_colour = (255, 255, 0)  # bright yellow
        else:  # all later species
            new_colour = random_colour()
        self.species.append((agent.genetic_representation, new_colour, [agent], 1))
        agent.species_colour = new_colour

    def update(self):
        for i, (g_i, c_i, A_i, n_i) in enumerate(self.species):
            # Remove dead agents but keep the total count
            alive_agents = [agent for agent in A_i if agent.alive]

            if alive_agents:
                # Update species with only living agents
                self.species[i] = (g_i, c_i, alive_agents, n_i)
            else:
                # Mark empty species for removal
                self.species[i] = None

        # Remove empty species
        self.species = [s for s in self.species if s is not None]

    def get_species_stats(self):
        return [(g, c, len(A), n) for g, c, A, n in self.species]

    def get_all_agents(self):
        return [agent for species in self.species for agent in species[2] if agent.alive]


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.north = None
        self.south = None
        self.east = None
        self.west = None
        self.plant = None
        self.agent = None
        self.wall = None
        self.resource_level = 0

    def initialise_agent(self, genetic_representation, orientation, mass, num_communication_signals):
        agent = Agent(genetic_representation, self, orientation, mass, num_communication_signals)
        self.agent = agent
        return agent


class Myxomatrix:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[Cell(x, y) for x in range(width)] for y in range(height)]
        self.stitch()
        self.plants = []
        self.growth_regions = []
        self.species_manager = SpeciesManager(PROGENITOR_UPDATE_THRESHOLD)
        self.walls = []
        self.simulation_stats = SimulationStats()

    def stitch(self):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                cell.north = self.grid[(y - 1) % self.height][x]
                cell.south = self.grid[(y + 1) % self.height][x]
                cell.east = self.grid[y][(x + 1) % self.width]
                cell.west = self.grid[y][(x - 1) % self.width]

    def unstitch(self):
        for row in self.grid:
            for cell in row:
                cell.north = None
                cell.south = None
                cell.east = None
                cell.west = None

    def add_growth_region(self, growth_region):
        self.growth_regions.append(growth_region)

        plants_placed = 0
        attempts = 0
        max_attempts = growth_region.num_plants * 10

        while plants_placed < growth_region.num_plants and attempts < max_attempts:
            cell = self.get_random_empty_cell()
            if cell and self.is_within_region(cell, growth_region):
                plant = Plant(cell, growth_region)
                self.add_plant(plant)
                plants_placed += 1
            attempts += 1

        if plants_placed < growth_region.num_plants:
            print(
                f"Warning: Could not place all plants in growth region. Placed {plants_placed} out of {growth_region.num_plants}.")

    def add_plant(self, plant):
        self.plants.append(plant)
        plant.cell.plant = plant

    def add_agent(self, genetic_representation, cell, orientation, mass, num_communication_signals):
        agent = cell.initialise_agent(genetic_representation, orientation, mass, num_communication_signals)
        self.species_manager.process(agent)
        return agent

    def add_moving_agent(self, genetic_representation, cell, orientation, mass, num_communication_signals):
        agent = MovingAgent(genetic_representation, cell, orientation, mass, num_communication_signals)
        cell.agent = agent
        self.species_manager.process(agent)
        return agent

    def add_wall(self, x, y):
        cell = self.grid[y][x]
        if not cell.wall and not cell.plant and not cell.agent:
            wall = Wall(cell)
            cell.wall = wall
            self.walls.append(wall)

    def remove_agent(self, agent):
        agent.cell.agent = None

    def get_plant(self, x, y):
        return self.grid[y][x].plant

    def get_agent(self, x, y):
        return self.grid[y][x].agent

    def get_wall(self, x, y):
        return self.grid[y][x].wall

    def move_plant(self, plant, new_cell):
        plant.cell.plant = None
        plant.cell = new_cell
        new_cell.plant = plant

    def is_within_region(self, cell, region):
        return region.x_min <= cell.x <= region.x_max and region.y_min <= cell.y <= region.y_max

    def get_random_empty_cell_in_region(self, region):
        empty_cells = []
        for y in range(region.y_min, region.y_max + 1):
            for x in range(region.x_min, region.x_max + 1):
                cell = self.grid[y][x]
                if not cell.agent and not cell.plant and not cell.wall:
                    empty_cells.append(cell)
        return random.choice(empty_cells) if empty_cells else None

    def update(self):
        random.shuffle(self.plants)
        for plant in self.plants:
            plant.update(self)
        all_agents = self.species_manager.get_all_agents()
        random.shuffle(all_agents)
        new_agents = []
        for agent in all_agents:
            if agent.alive:
                agent.stats = self.simulation_stats
                new_agent = agent.update()
                if new_agent:
                    new_agents.append(new_agent)

        for new_agent in new_agents:
            self.species_manager.process(new_agent)

        self.species_manager.update()
        self.simulation_stats.update(self, len(all_agents))

    def get_random_empty_cell(self):
        empty_cells = [cell for row in self.grid for cell in row if not cell.agent and not cell.plant and not cell.wall]
        return random.choice(empty_cells) if empty_cells else None


def draw_simulation(screen, myxomatrix, max_resource_level):
    cell_size = CELL_SIZE

    screen.fill((0, 0, 0))  # Fill screen with black

    grid_width = myxomatrix.width * cell_size
    grid_height = myxomatrix.height * cell_size

    resource_surface = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA)
    agent_surface = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA)
    wall_surface = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA)

    # Draw resources and walls
    for y in range(myxomatrix.height):
        for x in range(myxomatrix.width):
            cell = myxomatrix.grid[y][x]
            if cell.wall:
                pygame.draw.rect(wall_surface, (160, 160, 160),
                                 (x * cell_size, y * cell_size, cell_size, cell_size))
            elif cell.resource_level > 0:  # Only draw resources if they exist
                # Pure green for resources
                resource_intensity = cell.resource_level / max_resource_level
                green = int(200 * resource_intensity + 30)
                color = (0, green, 0)
                pygame.draw.rect(resource_surface, color,
                                 (x * cell_size, y * cell_size, cell_size, cell_size))

    # Draw agents with enhanced visibility
    for species in myxomatrix.species_manager.species:
        base_color = species[1]
        for agent in species[2]:
            # Enhance the base color while maintaining original hue
            r, g, b = base_color

            # Scale colors to be more vibrant but maintain relationships
            max_component = max(r, g, b, 1)
            scale = 255 / max_component
            r = int(min(255, r * scale * 0.7 + 40))
            g = int(min(255, g * scale * 0.7 + 40))
            b = int(min(255, b * scale * 0.7 + 40))

            # Set minimum opacity but keep mass differences visible
            min_opacity = 80
            mass_opacity = int(175 * agent.mass)
            opacity = min(255, min_opacity + mass_opacity)

            pygame.draw.rect(agent_surface, (r, g, b, opacity),
                             (agent.cell.x * cell_size, agent.cell.y * cell_size,
                              cell_size, cell_size))

            # Draw orientation marker
            orientation_offset = {
                'Northward': (cell_size // 2, 0),
                'Eastward': (cell_size, cell_size // 2),
                'Southward': (cell_size // 2, cell_size),
                'Westward': (0, cell_size // 2)
            }
            x_offset, y_offset = orientation_offset[agent.orientation]
            marker_size = max(cell_size // 5, 2)
            pygame.draw.circle(agent_surface, (255, 255, 255, 255),
                               (agent.cell.x * cell_size + x_offset,
                                agent.cell.y * cell_size + y_offset),
                               marker_size)

    # Blend the surfaces together
    screen.blit(resource_surface, (0, 0))
    screen.blit(wall_surface, (0, 0))
    screen.blit(agent_surface, (0, 0))

    pygame.display.flip()


def find_max_neurons_and_synapses(myxomatrix):
    max_hidden_neurons = 0
    max_synapses = 0
    for species in myxomatrix.species_manager.species:
        for agent in species[2]:
            hidden_neurons = sum(1 for neuron_type in agent.genetic_representation.neuron_types
                                 if agent.genetic_representation._decode_neuron_type(neuron_type) == 'hidden')
            synapses = len([syn for syn in agent.genetic_representation.synapse_pre if syn != NULL_SENTINEL])

            max_hidden_neurons = max(max_hidden_neurons, hidden_neurons)
            max_synapses = max(max_synapses, synapses)

    return max_hidden_neurons, max_synapses


def initialise_environment(config, myxomatrix):
    """Initialise the environment with walls, growth regions and agents."""
    # Add walls
    if config['walls_top_bottom']:
        for x in range(config['width']):
            myxomatrix.add_wall(x, 0)
            myxomatrix.add_wall(x, config['height'] - 1)
    if config['walls_left_right']:
        for y in range(config['height']):
            myxomatrix.add_wall(0, y)
            myxomatrix.add_wall(config['width'] - 1, y)

    # Calculate center of the grid
    center_x = config['width'] // 2
    center_y = config['height'] // 2
    center_cell = myxomatrix.grid[center_y][center_x]

    first_agent_spawned = False

    # Add growth regions and spawn agents into them
    for region in config['growth_regions']:
        growth_region = GrowthRegion(
            region['x_min'], region['y_min'],
            region['x_max'], region['y_max'],
            region['num_plants'],
            region['tau_min'], region['tau_max'],
            region['r_min'], region['r_max']
        )
        myxomatrix.add_growth_region(growth_region)

        # Spawn agents in this region
        for i in range(region['num_agents']):
            # For the very first agent across all regions, try to spawn at center
            if not first_agent_spawned and myxomatrix.is_within_region(center_cell, growth_region):
                if not center_cell.agent and not center_cell.plant and not center_cell.wall:
                    cell = center_cell
                    first_agent_spawned = True
                    print(f"First agent spawned at center: ({center_x}, {center_y})")
                else:
                    # Center is occupied, fall back to random placement
                    cell = myxomatrix.get_random_empty_cell_in_region(growth_region)
                    if cell:
                        first_agent_spawned = True
                        print(f"Center occupied, first agent spawned at: ({cell.x}, {cell.y})")
            else:
                # Regular random placement for all other agents
                cell = myxomatrix.get_random_empty_cell_in_region(growth_region)

            if cell:
                genome = genome_pool.get()
                genome.configure(9, 31)
                orientation = random.choice(Agent.ORIENTATIONS)
                mass = 1
                if config['agents_can_move']:
                    agent = myxomatrix.add_moving_agent(genome, cell, orientation, mass, NUM_COMMUNICATION_SIGNALS)
                else:
                    agent = myxomatrix.add_agent(genome, cell, orientation, mass, NUM_COMMUNICATION_SIGNALS)
                cell.resource_level = 10

    print_controls()


def reinitialise_simulation(config, myxomatrix):
    """Reset the simulation to its initial state."""
    # Clear existing simulation
    myxomatrix.unstitch()
    myxomatrix.plants.clear()
    myxomatrix.growth_regions.clear()
    myxomatrix.walls.clear()

    # Clear all agents and their references
    for species in myxomatrix.species_manager.species:
        for agent in species[2]:
            agent.kill()

    myxomatrix.species_manager.species.clear()

    # Reset the simulation stats
    myxomatrix.simulation_stats = SimulationStats()

    # Create a new data analyzer with the fresh stats
    data_analyser = SimulationDataAnalyser(myxomatrix.simulation_stats)

    # Reinitialise grid
    myxomatrix.grid = [[Cell(x, y) for x in range(myxomatrix.width)] for y in range(myxomatrix.height)]
    myxomatrix.stitch()

    # Reinitialise environment
    initialise_environment(config, myxomatrix)

    return data_analyser


def show_initialisation_text(screen, font, new_window_width, new_window_height):
    text = font.render("Initialising environment...", True, (255, 255, 255))
    text_rect = text.get_rect(center=(new_window_width // 2, new_window_height // 2))
    screen.fill((0, 0, 0))
    screen.blit(text, text_rect)
    pygame.display.flip()


def cleanup_temp_video_frames():
    """Clean up any leftover temporary video frames from previous sessions."""
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        try:
            # Find all session directories
            session_dirs = glob.glob(os.path.join(temp_dir, "*"))
            frames_deleted = 0
            dirs_deleted = 0

            for session_dir in session_dirs:
                if os.path.isdir(session_dir):
                    # Count frames before deletion
                    frame_files = glob.glob(os.path.join(session_dir, "frame_*.png"))
                    frames_deleted += len(frame_files)

                    # Remove the entire session directory
                    shutil.rmtree(session_dir)
                    dirs_deleted += 1

            if frames_deleted > 0:
                print(f"Cleaned up {frames_deleted} temporary video frames from {dirs_deleted} sessions")

            # Remove temp_frames directory if it's empty
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)

        except Exception as e:
            print(f"Warning: Could not clean up temporary video frames: {e}")


def cleanup_on_exit(video_recorder):
    """Clean up resources when the program exits."""
    try:
        # If recording is in progress, cancel it
        if video_recorder.is_recording():
            print("Cancelling video recording due to program exit...")
            video_recorder.cancel_recording()

        # Clean up any remaining temp frames
        cleanup_temp_video_frames()

    except Exception as e:
        print(f"Warning during cleanup: {e}")


def main(fade_time=1.0, logo_display_time=2.0):
    # Clean up any leftover temp frames from previous sessions
    cleanup_temp_video_frames()

    pygame.init()
    screen = pygame.display.set_mode((SETUP_WINDOW_WIDTH, SETUP_WINDOW_HEIGHT))
    pygame.display.set_caption("The Myxomatrix")
    clock = pygame.time.Clock()

    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    original_logo = pygame.image.load(logo_path).convert_alpha()

    video_recorder = VideoRecorder(
        fps=FPS,
        logo_path=logo_path,
        intro_sec=logo_display_time + 2 * fade_time,
        fade_sec=fade_time)

    font = pygame.font.Font(None, 36)

    def crop_transparent(surface):
        # Get the bounding box of non-transparent pixels
        bbox = surface.get_bounding_rect()
        # Crop the surface to the bounding box
        return surface.subsurface(bbox)

    # Remove any potential border by cropping the image
    original_logo = crop_transparent(original_logo)

    # Calculate the scaling factor to fit the logo within 1/3 of the screen while maintaining aspect ratio
    logo_scale = min(SETUP_WINDOW_WIDTH / (3 * original_logo.get_width()),
                     SETUP_WINDOW_HEIGHT / (3 * original_logo.get_height()))

    # Scale the logo
    logo_size = (int(original_logo.get_width() * logo_scale),
                 int(original_logo.get_height() * logo_scale))
    logo = pygame.transform.smoothscale(original_logo, logo_size)

    # Calculate logo position to center it
    logo_pos = ((SETUP_WINDOW_WIDTH - logo_size[0]) // 2,
                (SETUP_WINDOW_HEIGHT - logo_size[1]) // 2)

    def fade_logo(start_alpha, end_alpha, fade_time):
        start_time = pygame.time.get_ticks()
        while True:
            current_time = pygame.time.get_ticks()
            elapsed = (current_time - start_time) / 1000.0
            if elapsed >= fade_time:
                alpha = end_alpha
            else:
                alpha = int(start_alpha + (end_alpha - start_alpha) * (elapsed / fade_time))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False

            screen.fill((0, 0, 0))

            # Create a copy of the logo with the current alpha
            faded_logo = logo.copy()
            faded_logo.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)

            screen.blit(faded_logo, logo_pos)
            pygame.display.flip()
            clock.tick(60)

            if elapsed >= fade_time:
                break

        return True

    # Fade in, hold, and fade out the logo
    if not fade_logo(0, 255, fade_time):
        return
    start_time = pygame.time.get_ticks()
    while (pygame.time.get_ticks() - start_time) / 1000.0 < logo_display_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        clock.tick(60)
    if not fade_logo(255, 0, fade_time):
        return

    screen.fill((0, 0, 0))
    pygame.display.flip()

    # Run the simulation setup wizard
    manager = pygame_gui.UIManager((SETUP_WINDOW_WIDTH, SETUP_WINDOW_HEIGHT))
    setup = EnvironmentSetupInterface(screen, manager)
    config = setup.run()
    if config is None:
        pygame.quit()
        return

    # Handle seed setup
    seed_input = config.get('seed', 'RANDOM')
    if seed_input == "RANDOM":
        # Generate a random seed
        actual_seed = int(time.time() * 1000000) % 2147483647  # Large random seed
        print(f"Generated random seed: {actual_seed}")
    else:
        # Use the provided seed
        try:
            actual_seed = int(seed_input)
            print(f"Using provided seed: {actual_seed}")
        except ValueError:
            # If seed is not a valid integer, generate a hash from the string
            actual_seed = abs(hash(seed_input)) % 2147483647
            print(f"Using string-based seed '{seed_input}' -> {actual_seed}")

    # Set the global seed for deterministic behavior
    set_global_seed(actual_seed)

    # Calculate window size based on grid dimensions and CELL_SIZE
    window_width = config['width'] * CELL_SIZE
    window_height = config['height'] * CELL_SIZE
    screen = pygame.display.set_mode((window_width, window_height))

    # Display initialisation text
    show_initialisation_text(screen, font, window_width, window_height)

    # Initialize the simulation with the config
    myxomatrix = Myxomatrix(config['width'], config['height'])
    data_analyser = SimulationDataAnalyser(myxomatrix.simulation_stats)

    # Apply hyperparameters
    global CONSUMPTION_RATE, BASAL_METABOLIC_COST, NEURON_COST, SYNAPSE_COST, REPLICATION_COST
    global REPLICATION_SURVIVABILITY_SCALAR, REPLICATION_COOLDOWN, MOVEMENT_COST, TRANSFER_RATE
    global CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, COMPATIBILITY_THRESHOLD, NUM_COMMUNICATION_SIGNALS
    global PROGENITOR_UPDATE_THRESHOLD

    hyperparameters = config['hyperparameters']
    CONSUMPTION_RATE = hyperparameters['Consumption Rate']
    BASAL_METABOLIC_COST = hyperparameters['Basal Metabolic Cost']
    NEURON_COST = hyperparameters['Neuron Cost']
    SYNAPSE_COST = hyperparameters['Synapse Cost']
    REPLICATION_COST = hyperparameters['Replication Cost']
    REPLICATION_SURVIVABILITY_SCALAR = hyperparameters['Replication Survivability Scalar']
    REPLICATION_COOLDOWN = int(hyperparameters['Replication Cooldown'])
    MOVEMENT_COST = hyperparameters['Movement Cost']
    TRANSFER_RATE = hyperparameters['Transfer Rate']
    CROSSOVER_PROBABILITY = hyperparameters['Crossover Probability']
    MUTATION_PROBABILITY = hyperparameters['Mutation Probability']
    COMPATIBILITY_THRESHOLD = hyperparameters['Compatibility Threshold']
    NUM_COMMUNICATION_SIGNALS = int(hyperparameters['Number of Communication Signals'])
    PROGENITOR_UPDATE_THRESHOLD = int(hyperparameters['Progenitor Update Threshold'])

    # Initialise the environment
    initialise_environment(config, myxomatrix)

    gc.disable()

    max_hidden_neurons, max_synapses = find_max_neurons_and_synapses(myxomatrix)
    print(f"Tick 0 (SEED {get_current_seed()}):")
    print(f"  Max Hidden Neurons: {max_hidden_neurons}")
    print(f"  Max Synapses: {max_synapses}")
    print(f"Number of species: {len(myxomatrix.species_manager.species)}")
    print(f"Total agents: {sum(len(species[2]) for species in myxomatrix.species_manager.species)}")

    running = True
    tick_counter = 0
    data_analyser = SimulationDataAnalyser(myxomatrix.simulation_stats)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Clean up before quitting
                cleanup_on_exit(video_recorder)
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Display initialisation text
                    show_initialisation_text(screen, font, window_width, window_height)

                    # Generate a new random seed for restart
                    new_seed = int(time.time() * 1000000) % 2147483647
                    set_global_seed(new_seed)
                    print(f"Restart with new random seed: {new_seed}")

                    # Reset simulation and get new data analyser
                    data_analyser = reinitialise_simulation(config, myxomatrix)
                    tick_counter = 0
                    print("Simulation reinitialised!")

                    # Print initial statistics
                    max_hidden_neurons, max_synapses = find_max_neurons_and_synapses(myxomatrix)
                    print(f"Tick 0 (SEED {get_current_seed()}):")
                    print(f"  Max Hidden Neurons: {max_hidden_neurons}")
                    print(f"  Max Synapses: {max_synapses}")
                    print(f"Number of species: {len(myxomatrix.species_manager.species)}")
                    print(f"Total agents: {sum(len(species[2]) for species in myxomatrix.species_manager.species)}")

                elif event.key == pygame.K_q:
                    # Quit to setup interface
                    cleanup_on_exit(video_recorder)
                    pygame.quit()
                    main(fade_time, logo_display_time)
                    return

                elif event.key == pygame.K_v:
                    if video_recorder.is_processing():
                        print("Video is still being processed. Please wait...")
                    else:
                        output_file = video_recorder.toggle_recording(progress_callback=video_progress_callback)
                        if output_file:  # toggle_recording returns filename when stopping
                            print(f"Video processing started. Output will be: {output_file}")
                        else:  # toggle_recording returns None when starting
                            print("Recording started - press V to stop, C to cancel")

                elif event.key == pygame.K_c:
                    if video_recorder.is_recording():
                        video_recorder.cancel_recording()
                        print("Recording cancelled!")
                    else:
                        print("No recording in progress to cancel")

        myxomatrix.update()

        if tick_counter % 1000 == 0:
            gc.collect()

        tick_counter += 1
        if tick_counter % 100 == 0:
            max_hidden_neurons, max_synapses = find_max_neurons_and_synapses(myxomatrix)
            print(f"Tick {tick_counter} (SEED {get_current_seed()}):")
            print(f"  Max Hidden Neurons: {max_hidden_neurons}")
            print(f"  Max Synapses: {max_synapses}")

        draw_simulation(screen, myxomatrix, 10)

        # Display video processing progress if applicable
        if video_recorder.is_processing():
            font = pygame.font.Font(None, 24)
            progress_text = font.render(f"Processing Video: {video_progress_percentage}%", True, (255, 255, 255))
            detail_text = font.render(video_progress_message, True, (255, 255, 255))

            # Create a semi-transparent overlay
            overlay = pygame.Surface((screen.get_width(), 60))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            # Draw progress text
            screen.blit(progress_text, (10, 10))
            screen.blit(detail_text, (10, 35))
        video_recorder.save_frame(screen)

    # Clean up before final exit
    cleanup_on_exit(video_recorder)

    # Plot run
    data_analyser.analyse_single_run()
    # data_analyser.analyse_all_runs()

    gc.enable()
    gc.collect()
    pygame.quit()


def video_progress_callback(message, percentage):
    global video_progress_message, video_progress_percentage
    video_progress_message = message
    video_progress_percentage = percentage

def print_controls():
    """Print the available controls for the simulation."""
    print("\n" + "="*50)
    print("          SIMULATION CONTROLS")
    print("="*50)
    print("R - Restart simulation with new random seed")
    print("Q - Quit to setup interface")
    print("V - Start/Stop video recording")
    print("C - Cancel current video recording")
    print("ESC/Close Window - Exit program")
    print("="*50)
    print("Video files are saved as MP4 in 'simulation_recordings' in the current directory")
    print("="*50 + "\n")


if __name__ == "__main__":
    main(fade_time=0.2, logo_display_time=2.0)
