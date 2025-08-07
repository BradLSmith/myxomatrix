import copy
import weakref
import math
import array
import random
from copy import deepcopy
from objectpool import ObjectPool

# Define sentinel value for null positions
NULL_SENTINEL = -1


class OENEASTDPGenome:
    def __init__(self):
        self.n_input_neurons = 0
        self.n_output_neurons = 0
        self.neuron_out_synapses = []
        self.neuron_in_synapses = []
        self.neuron_types = array.array('b')
        self.neuron_biases = array.array('f')
        self.neuron_taus = array.array('f')
        self.neuron_refractory_periods = array.array('f')
        self.synapse_pre = array.array('i')
        self.synapse_post = array.array('i')
        self.synapse_weights = array.array('f')
        self.synapse_pre_rates = array.array('f')
        self.synapse_post_rates = array.array('f')
        self.reset()

    def reset(self):
        self.n_input_neurons = 0
        self.n_output_neurons = 0

        # Clear existing data
        self.neuron_out_synapses.clear()
        self.neuron_in_synapses.clear()
        self.neuron_types = array.array('b')
        self.neuron_biases = array.array('f')
        self.neuron_taus = array.array('f')
        self.neuron_refractory_periods = array.array('f')
        self.synapse_pre = array.array('i')
        self.synapse_post = array.array('i')
        self.synapse_weights = array.array('f')
        self.synapse_pre_rates = array.array('f')
        self.synapse_post_rates = array.array('f')

    def configure(self, n_input_neurons, n_output_neurons):
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons

        # Initialise input neurons
        for _ in range(n_input_neurons):
            self.add_neuron('input')

        # Initialise output neurons
        for _ in range(n_output_neurons):
            self.add_neuron('output')

    def add_neuron(self, neuron_type, bias=None, tau=10, refractory_period=1.0):
        if bias is None:
            bias = random.uniform(-1, 1)

        # Look for a null sentinel position to create the new neuron in
        for i, neuron in enumerate(self.neuron_types):
            if neuron == NULL_SENTINEL:
                self.neuron_in_synapses[i] = []
                self.neuron_out_synapses[i] = []
                self.neuron_types[i] = self._encode_neuron_type(neuron_type)
                self.neuron_biases[i] = bias
                self.neuron_taus[i] = tau
                self.neuron_refractory_periods[i] = refractory_period
                return i

        # Else if no null sentinel present, extend the lists
        self.neuron_in_synapses.append([])
        self.neuron_out_synapses.append([])
        self.neuron_types.append(self._encode_neuron_type(neuron_type))
        self.neuron_biases.append(bias)
        self.neuron_taus.append(tau)
        self.neuron_refractory_periods.append(refractory_period)
        return len(self.neuron_types) - 1

    def add_synapse(self, pre_index, post_index, weight=None, pre_rate=0.0, post_rate=0.0):
        if weight is None:
            weight = random.uniform(-1, 1)

        # Look for a null sentinel position to create the new synapse in
        for i, syn in enumerate(self.synapse_pre):
            if syn == NULL_SENTINEL:
                self.synapse_pre[i] = pre_index
                self.synapse_post[i] = post_index
                self.synapse_weights[i] = weight
                self.synapse_pre_rates[i] = pre_rate
                self.synapse_post_rates[i] = post_rate
                self.associate_neurons_with_synapse(i)
                return i

        # Else if no null sentinel present, extend the lists
        self.synapse_pre.append(pre_index)
        self.synapse_post.append(post_index)
        self.synapse_weights.append(weight)
        self.synapse_pre_rates.append(pre_rate)
        self.synapse_post_rates.append(post_rate)
        idx = len(self.synapse_pre) - 1
        self.associate_neurons_with_synapse(idx)
        return idx

    def associate_neurons_with_synapse(self, synapse_index):
        pre_neuron_index = self.synapse_pre[synapse_index]
        post_neuron_index = self.synapse_post[synapse_index]
        self.neuron_out_synapses[pre_neuron_index].append(synapse_index)
        self.neuron_in_synapses[post_neuron_index].append(synapse_index)

    def dissociate_neurons_with_synapse(self, synapse_index):
        pre_neuron_index = self.synapse_pre[synapse_index]
        post_neuron_index = self.synapse_post[synapse_index]
        self.neuron_out_synapses[pre_neuron_index].remove(synapse_index)
        self.neuron_in_synapses[post_neuron_index].remove(synapse_index)

    @staticmethod
    def _encode_neuron_type(neuron_type):
        return {'input': 0, 'hidden': 1, 'output': 2}[neuron_type]

    @staticmethod
    def _decode_neuron_type(encoded_type):
        return ['input', 'hidden', 'output'][encoded_type]

    def mutate(self):
        mutation_ops = [
            self.mutate_neuron_biases,
            self.mutate_neuron_taus,
            self.mutate_neuron_refractory_periods,
            self.mutate_synapse_weights,
            self.mutate_synapse_pre_rates,
            self.mutate_synapse_post_rates,
            self.mutate_add_neuron,
            self.mutate_add_synapse,
            self.mutate_delete_neurons,
            self.mutate_delete_synapses,
            self.mutate_duplicate_neurons
        ]

        weights = [15, 5, 5, 15, 5, 5, 10, 15, 5, 10, 0]

        chosen_op = random.choices(mutation_ops, weights=weights, k=1)[0]
        chosen_op()

    def mutate_synapse_weights(self, rate=0.2):
        for i in range(len(self.synapse_pre)):
            if self.synapse_pre[i] != NULL_SENTINEL and random.random() < rate:
                self.synapse_weights[i] += random.gauss(0, 0.1)

    def mutate_synapse_pre_rates(self, rate=0.2):
        for i in range(len(self.synapse_pre)):
            if self.synapse_pre[i] != NULL_SENTINEL and random.random() < rate:
                self.synapse_pre_rates[i] += random.gauss(0, 0.01)

    def mutate_synapse_post_rates(self, rate=0.2):
        for i in range(len(self.synapse_pre)):
            if self.synapse_pre[i] != NULL_SENTINEL and random.random() < rate:
                self.synapse_post_rates[i] += random.gauss(0, 0.01)

    def mutate_neuron_biases(self, rate=0.1):
        for i in range(len(self.neuron_types)):
            if self.neuron_types[i] != NULL_SENTINEL and random.random() < rate:
                self.neuron_biases[i] += random.gauss(0, 0.1)

    def mutate_neuron_taus(self, rate=0.1):
        for i in range(len(self.neuron_types)):
            if self.neuron_types[i] != NULL_SENTINEL and random.random() < rate:
                self.neuron_taus[i] += random.gauss(0, 0.1)

    def mutate_neuron_refractory_periods(self, rate=0.1):
        for i in range(len(self.neuron_types)):
            if self.neuron_types[i] != NULL_SENTINEL and random.random() < rate:
                self.neuron_refractory_periods[i] += random.gauss(0, 0.1)
                self.neuron_refractory_periods[i] = max(0.1, self.neuron_refractory_periods[i])

    def mutate_add_neuron(self):
        new_neuron_index = self.add_neuron('hidden',
                                           bias=random.uniform(-1, 1),
                                           tau=random.uniform(1, 20),
                                           refractory_period=random.uniform(0.1, 2.0))
        valid_synapses = [i for i, pre in enumerate(self.synapse_pre) if pre != NULL_SENTINEL]
        if valid_synapses:
            synapse_to_split = random.choice(valid_synapses)
            pre = self.synapse_pre[synapse_to_split]
            post = self.synapse_post[synapse_to_split]
            weight = self.synapse_weights[synapse_to_split]
            pre_rate = self.synapse_pre_rates[synapse_to_split]
            post_rate = self.synapse_post_rates[synapse_to_split]

            self.delete_specific_synapse(synapse_to_split)

            # Use the same weight and rates for new synapses
            self.add_synapse(pre, new_neuron_index,
                             weight=weight,
                             pre_rate=pre_rate,
                             post_rate=post_rate)
            self.add_synapse(new_neuron_index, post,
                             weight=weight,
                             pre_rate=pre_rate,
                             post_rate=post_rate)
        else:
            self._connect_isolated_neuron(new_neuron_index)

    def mutate_add_synapse(self):
        valid_neurons = [i for i, type in enumerate(self.neuron_types) if type != NULL_SENTINEL]
        if len(valid_neurons) > 0:
            pre_neuron = random.choice(valid_neurons)
            post_neuron = random.choice(valid_neurons)
            if pre_neuron != post_neuron:
                self.add_synapse(pre_neuron, post_neuron,
                                 weight=random.uniform(-1, 1),
                                 pre_rate=random.uniform(0, 0.1),
                                 post_rate=random.uniform(0, 0.1))

    def mutate_delete_neurons(self):
        self.mutate_delete_neurons_percentage(random.randint(5, 20))

    def _mutate_delete_neurons_random(self):
        for i in range(self.n_input_neurons + self.n_output_neurons, len(self.neuron_types)):
            if self.neuron_types[i] != NULL_SENTINEL and random.random() < 0.05:
                self._delete_neuron(i)

    def _mutate_delete_neurons_n_point(self):
        hidden_neurons = [i for i in range(self.n_input_neurons + self.n_output_neurons, len(self.neuron_types))
                          if self.neuron_types[i] != NULL_SENTINEL]

        if not hidden_neurons:
            return  # No hidden neurons to delete

        n_points = min(random.randint(1, 3), len(hidden_neurons))
        points = sorted(random.sample(hidden_neurons, n_points))
        points.append(len(self.neuron_types))

        start = self.n_input_neurons + self.n_output_neurons
        for end in points:
            if random.random() < 0.5:
                for i in range(start, end):
                    if self.neuron_types[i] != NULL_SENTINEL:
                        self._delete_neuron(i)
            start = end

    def mutate_delete_neurons_percentage(self, percentage=10):
        hidden_neurons = [i for i in range(self.n_input_neurons + self.n_output_neurons, len(self.neuron_types))
                          if self.neuron_types[i] != NULL_SENTINEL]

        if not hidden_neurons:
            return  # No hidden neurons to delete

        num_to_delete = max(1, int(len(hidden_neurons) * percentage / 100))
        neurons_to_delete = random.sample(hidden_neurons, num_to_delete)

        for neuron_index in neurons_to_delete:
            self._delete_neuron(neuron_index)

    def _delete_neuron(self, index):
        # Delete the neuron's synapses
        for i in range(len(self.synapse_pre)):
            if self.synapse_pre[i] == index or self.synapse_post[i] == index:
                self.delete_specific_synapse(i)

        # Delete neuron
        self.neuron_in_synapses[index] = []
        self.neuron_out_synapses[index] = []
        self.neuron_types[index] = NULL_SENTINEL
        self.neuron_biases[index] = 0
        self.neuron_taus[index] = 0
        self.neuron_refractory_periods[index] = 0

    def mutate_delete_synapses(self):
        self.mutate_delete_synapses_percentage(random.randint(5, 20))

    def _mutate_delete_synapses_random(self):
        for i in range(len(self.synapse_pre)):
            if self.synapse_pre[i] != NULL_SENTINEL and random.random() < 0.05:
                self.delete_specific_synapse(i)

    def _mutate_delete_synapses_n_point(self):
        valid_synapses = [i for i, pre in enumerate(self.synapse_pre) if pre != NULL_SENTINEL]

        if not valid_synapses:
            return  # No valid synapses to delete

        n_points = min(random.randint(1, 5), len(valid_synapses))
        points = sorted(random.sample(valid_synapses, n_points))
        points.append(len(self.synapse_pre))

        start = 0
        for end in points:
            if random.random() < 0.5:
                for i in range(start, end):
                    if self.synapse_pre[i] != NULL_SENTINEL:
                        self.delete_specific_synapse(i)
            start = end

    def mutate_delete_synapses_percentage(self, percentage=10):
        valid_synapses = [i for i, pre in enumerate(self.synapse_pre) if pre != NULL_SENTINEL]

        if not valid_synapses:
            return  # No valid synapses to delete

        num_to_delete = max(1, int(len(valid_synapses) * percentage / 100))
        synapses_to_delete = random.sample(valid_synapses, num_to_delete)

        for synapse_index in synapses_to_delete:
            self.delete_specific_synapse(synapse_index)

    def mutate_duplicate_neurons(self):
        method = random.choice(["random", "n_point"])
        if method == "random":
            self._mutate_duplicate_neurons_random()
        else:
            self._mutate_duplicate_neurons_n_point()

    def _mutate_duplicate_neurons_random(self):
        for i in range(self.n_input_neurons + self.n_output_neurons, len(self.neuron_types)):
            if self.neuron_types[i] != NULL_SENTINEL and random.random() < 0.05:
                self._duplicate_neuron(i)

    def _mutate_duplicate_neurons_n_point(self):
        hidden_neurons = [i for i in range(self.n_input_neurons + self.n_output_neurons, len(self.neuron_types))
                          if self.neuron_types[i] != NULL_SENTINEL]

        if not hidden_neurons:
            return  # No hidden neurons to duplicate

        n_points = min(random.randint(1, 3), len(hidden_neurons))
        points = sorted(random.sample(hidden_neurons, n_points))
        points.append(len(self.neuron_types))

        start = self.n_input_neurons + self.n_output_neurons
        for end in points:
            if random.random() < 0.5:
                for i in range(start, end):
                    if self.neuron_types[i] != NULL_SENTINEL:
                        self._duplicate_neuron(i)
            start = end

    def _duplicate_neuron(self, index):
        new_neuron_index = self.add_neuron(
            self._decode_neuron_type(self.neuron_types[index]),
            self.neuron_biases[index],
            self.neuron_taus[index],
            self.neuron_refractory_periods[index]
        )

        # Duplicate incoming synapses
        for syn_index in self.neuron_in_synapses[index]:
            if self.synapse_pre[syn_index] != NULL_SENTINEL:
                pre_neuron = self.synapse_pre[syn_index]
                weight = self.synapse_weights[syn_index]
                pre_rate = self.synapse_pre_rates[syn_index]
                post_rate = self.synapse_post_rates[syn_index]

                new_syn_index = self.add_synapse(
                    pre_neuron,
                    new_neuron_index,
                    weight,
                    pre_rate,
                    post_rate
                )

        # Duplicate outgoing synapses
        for syn_index in self.neuron_out_synapses[index]:
            if self.synapse_post[syn_index] != NULL_SENTINEL:
                post_neuron = self.synapse_post[syn_index]
                weight = self.synapse_weights[syn_index]
                pre_rate = self.synapse_pre_rates[syn_index]
                post_rate = self.synapse_post_rates[syn_index]

                new_syn_index = self.add_synapse(
                    new_neuron_index,
                    post_neuron,
                    weight,
                    pre_rate,
                    post_rate
                )

    def delete_specific_synapse(self, index):
        if 0 <= index < len(self.synapse_pre) and self.synapse_pre[index] != NULL_SENTINEL:
            self.dissociate_neurons_with_synapse(index)
            self.synapse_pre[index] = NULL_SENTINEL
            self.synapse_post[index] = NULL_SENTINEL
            self.synapse_weights[index] = 0
            self.synapse_pre_rates[index] = 0
            self.synapse_post_rates[index] = 0

    def _connect_isolated_neuron(self, neuron_index):
        valid_neurons = [i for i, type in enumerate(self.neuron_types) if type != NULL_SENTINEL]
        if len(valid_neurons) >= 2:
            input_neuron = random.choice(valid_neurons)
            output_neuron = random.choice(valid_neurons)
            self.add_synapse(input_neuron, neuron_index)
            self.add_synapse(neuron_index, output_neuron)
        elif len(valid_neurons) == 1:
            other_neuron = valid_neurons[0]
            self.add_synapse(other_neuron, neuron_index)
            self.add_synapse(neuron_index, other_neuron)

    def copy(self, genome_pool):
        new_genome = genome_pool.get()
        new_genome.n_input_neurons = self.n_input_neurons
        new_genome.n_output_neurons = self.n_output_neurons
        new_genome.neuron_out_synapses = copy.deepcopy(self.neuron_out_synapses)
        new_genome.neuron_in_synapses = copy.deepcopy(self.neuron_in_synapses)
        new_genome.neuron_types = array.array('b', self.neuron_types)
        new_genome.neuron_biases = array.array('f', self.neuron_biases)
        new_genome.neuron_taus = array.array('f', self.neuron_taus)
        new_genome.neuron_refractory_periods = array.array('f', self.neuron_refractory_periods)
        new_genome.synapse_pre = array.array('i', self.synapse_pre)
        new_genome.synapse_post = array.array('i', self.synapse_post)
        new_genome.synapse_weights = array.array('f', self.synapse_weights)
        new_genome.synapse_pre_rates = array.array('f', self.synapse_pre_rates)
        new_genome.synapse_post_rates = array.array('f', self.synapse_post_rates)
        return new_genome

    def print_stats(self):
        # Count neurons
        total_neurons = len(self.neuron_types)
        active_neurons = sum(1 for nt in self.neuron_types if nt != NULL_SENTINEL)
        sentinel_neurons = total_neurons - active_neurons

        # Count synapses
        total_synapses = len(self.synapse_pre)
        active_synapses = sum(1 for pre in self.synapse_pre if pre != NULL_SENTINEL)
        sentinel_synapses = total_synapses - active_synapses

        print(f"Genome Statistics:")
        print(f"Neurons: {active_neurons} active, {sentinel_neurons} sentinel, {total_neurons} total")
        print(f"Synapses: {active_synapses} active, {sentinel_synapses} sentinel, {total_synapses} total")

    def _extend_neurons(self, new_length):
        current_length = len(self.neuron_types)
        if new_length > current_length:
            extension_length = new_length - current_length
            self.neuron_in_synapses.extend([[] for _ in range(extension_length)])
            self.neuron_out_synapses.extend([[] for _ in range(extension_length)])
            self.neuron_types.extend([NULL_SENTINEL] * extension_length)
            self.neuron_biases.extend([0.0] * extension_length)
            self.neuron_taus.extend([0.0] * extension_length)
            self.neuron_refractory_periods.extend([0.0] * extension_length)

    def _extend_synapses(self, new_length):
        current_length = len(self.synapse_pre)
        if new_length > current_length:
            extension_length = new_length - current_length
            self.synapse_pre.extend([NULL_SENTINEL] * extension_length)
            self.synapse_post.extend([NULL_SENTINEL] * extension_length)
            self.synapse_weights.extend([0.0] * extension_length)
            self.synapse_pre_rates.extend([0.0] * extension_length)
            self.synapse_post_rates.extend([0.0] * extension_length)

    def validate_synapse(self, pre_index, post_index):
        # Check if both neurons exist
        if (pre_index >= len(self.neuron_types) or post_index >= len(self.neuron_types) or
                self.neuron_types[pre_index] == NULL_SENTINEL or self.neuron_types[post_index] == NULL_SENTINEL):
            return False

        # Check if a synapse already exists between these neurons
        for synapse_index in self.neuron_out_synapses[pre_index]:
            if self.synapse_post[synapse_index] == post_index:
                return False

        return True


# Object pool for genomes
genome_pool = ObjectPool(
        create_func=OENEASTDPGenome,
        reset_func=OENEASTDPGenome.reset,
        initial_size=100
    )


class LeakyIntegrateFireNeuron:
    def __init__(self):
        self.reset()

    def reset(self):
        # Reset to default values
        self.type = None
        self.bias = 0.0
        self.tau = 0.0
        self.refractory_period = 0
        self.membrane_potential = 0.0
        self.last_spike_step = -1000
        self.incoming_synapses = []
        self.outgoing_synapses = []

    def configure(self, neuron_type, bias, tau, refractory_period):
        self.type = neuron_type
        self.bias = bias
        self.tau = tau
        self.refractory_period = refractory_period

    def receive_spike(self, input_value, current_step):
        if current_step - self.last_spike_step > self.refractory_period:
            self.membrane_potential += input_value

    def update(self, current_step):
        if current_step - self.last_spike_step <= self.refractory_period:
            return False  # Neuron is in refractory period, cannot spike

        # Leaky integrate-and-fire model
        self.membrane_potential *= math.exp(-1 / self.tau)
        self.membrane_potential += self.bias

        # Check for spike
        if self.membrane_potential >= 1.0:
            self.last_spike_step = current_step
            self.membrane_potential = 0.0
            return True
        return False


class SpikeTimeDependentPlasticSynapse:
    def __init__(self):
        self.reset()

    def reset(self):
        # Reset to default values
        self.pre_neuron = None
        self.post_neuron = None
        self.weight = 0.0
        self.pre_rate = 0.0
        self.post_rate = 0.0
        self.pre_tau = 20.0
        self.post_tau = 20.0
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.max_weight = 1.0
        self.min_weight = -1.0

    def configure(self, pre_neuron, post_neuron, weight, pre_rate, post_rate):
        self.pre_neuron = weakref.ref(pre_neuron) if pre_neuron else None
        self.post_neuron = weakref.ref(post_neuron) if post_neuron else None
        self.weight = weight
        self.pre_rate = pre_rate
        self.post_rate = post_rate

    def pre_spike(self, current_step):
        self.pre_trace += 1.0
        post_neuron = self.post_neuron()
        if post_neuron:
            post_neuron.receive_spike(self.weight, current_step)

    def post_spike(self, current_step):
        self.post_trace += 1.0

    def update(self, current_step):
        # Decay traces
        self.pre_trace *= math.exp(-1 / self.pre_tau)
        self.post_trace *= math.exp(-1 / self.post_tau)

        # STDP weight update
        dw = self.post_rate * self.pre_trace - self.pre_rate * self.post_trace
        self.weight += dw
        self.weight = max(self.min_weight, min(self.max_weight, self.weight))


class OENEASTDPTNetwork:
    neuron_pool = ObjectPool(
        create_func=LeakyIntegrateFireNeuron,
        reset_func=LeakyIntegrateFireNeuron.reset,
        initial_size=100
    )
    synapse_pool = ObjectPool(
        create_func=SpikeTimeDependentPlasticSynapse,
        reset_func=SpikeTimeDependentPlasticSynapse.reset,
        initial_size=500
    )

    def __init__(self, genome):
        self.genome = genome
        self.neurons = []
        self.synapses = []
        self.neuron_index_map = {}
        self.current_step = 0
        self.build_network()
        self.output_neurons = [n for n in self.neurons if n.type == 'output']

    def build_network(self):
        # Create neurons based on the genome's neuron data
        for i in range(len(self.genome.neuron_types)):
            if self.genome.neuron_types[i] != NULL_SENTINEL:
                neuron_type = self.genome._decode_neuron_type(self.genome.neuron_types[i])
                bias = self.genome.neuron_biases[i]
                tau = max(1.0, self.genome.neuron_taus[i])
                refractory_period = self.genome.neuron_refractory_periods[i]

                neuron = self.neuron_pool.get()
                neuron.configure(neuron_type, bias, tau, refractory_period)

                self.neurons.append(neuron)
                self.neuron_index_map[i] = len(self.neurons) - 1

        # Create synapses based on the genome's synapse data
        for i in range(len(self.genome.synapse_pre)):
            pre_index = self.genome.synapse_pre[i]
            post_index = self.genome.synapse_post[i]
            if pre_index != NULL_SENTINEL and post_index != NULL_SENTINEL:
                if pre_index in self.neuron_index_map and post_index in self.neuron_index_map:
                    pre_neuron = self.neurons[self.neuron_index_map[pre_index]]
                    post_neuron = self.neurons[self.neuron_index_map[post_index]]
                    weight = self.genome.synapse_weights[i]
                    pre_rate = self.genome.synapse_pre_rates[i]
                    post_rate = self.genome.synapse_post_rates[i]

                    synapse = self.synapse_pool.get()
                    synapse.configure(pre_neuron, post_neuron, weight, pre_rate, post_rate)

                    self.synapses.append(synapse)
                    pre_neuron.outgoing_synapses.append(weakref.ref(synapse))
                    post_neuron.incoming_synapses.append(weakref.ref(synapse))

    def process(self, inputs, iterations=1):
        outputs = [0] * len(self.output_neurons)

        for _ in range(iterations):
            self.current_step += 1

            # Set input neuron activations
            for i, input_value in enumerate(inputs):
                if i < len(self.neurons) and self.neurons[i].type == 'input':
                    if input_value > 0:
                        self.neurons[i].receive_spike(input_value, self.current_step)

            # Update all neurons
            for neuron in self.neurons:
                if neuron.update(self.current_step):
                    # If neuron spiked, activate outgoing synapses
                    for synapse_ref in neuron.outgoing_synapses:
                        synapse = synapse_ref()
                        if synapse:
                            synapse.pre_spike(self.current_step)

            # Update all synapses
            for synapse in self.synapses:
                synapse.update(self.current_step)

            # Collect outputs
            for i, neuron in enumerate(self.output_neurons):
                if neuron.last_spike_step == self.current_step:
                    outputs[i] += 1

        return outputs

    def clear(self):
        """
        Clear all references in the network and return objects to their respective pools.
        """
        for neuron in self.neurons:
            neuron.incoming_synapses.clear()
            neuron.outgoing_synapses.clear()
            self.neuron_pool.release(neuron)

        for synapse in self.synapses:
            synapse.pre_neuron = None
            synapse.post_neuron = None
            self.synapse_pool.release(synapse)

        self.neurons.clear()
        self.synapses.clear()
        self.neuron_index_map.clear()
        self.output_neurons.clear()
        self.genome = None


def attribute_crossover(parent1, parent2, method='uniform', n=2, prob=0.5):
    child1, child2 = parent1.copy(genome_pool), parent2.copy(genome_pool)

    # Neuron attribute crossover
    if method == 'uniform':
        for i in range(min(len(child1.neuron_biases), len(child2.neuron_biases))):
            if random.random() < prob:
                child1.neuron_biases[i], child2.neuron_biases[i] = child2.neuron_biases[i], child1.neuron_biases[i]
                child1.neuron_taus[i], child2.neuron_taus[i] = child2.neuron_taus[i], child1.neuron_taus[i]
                child1.neuron_refractory_periods[i], child2.neuron_refractory_periods[i] = child2.neuron_refractory_periods[i], child1.neuron_refractory_periods[i]
    elif method == 'n_point':
        points = sorted(random.sample(range(1, min(len(child1.neuron_biases), len(child2.neuron_biases))), n))
        swap = False
        for i in range(min(len(child1.neuron_biases), len(child2.neuron_biases))):
            if i in points:
                swap = not swap
            if swap:
                child1.neuron_biases[i], child2.neuron_biases[i] = child2.neuron_biases[i], child1.neuron_biases[i]
                child1.neuron_taus[i], child2.neuron_taus[i] = child2.neuron_taus[i], child1.neuron_taus[i]
                child1.neuron_refractory_periods[i], child2.neuron_refractory_periods[i] = child2.neuron_refractory_periods[i], child1.neuron_refractory_periods[i]
    elif method == 'n_random':
        for _ in range(n):
            i = random.randint(0, min(len(child1.neuron_biases), len(child2.neuron_biases)) - 1)
            child1.neuron_biases[i], child2.neuron_biases[i] = child2.neuron_biases[i], child1.neuron_biases[i]
            child1.neuron_taus[i], child2.neuron_taus[i] = child2.neuron_taus[i], child1.neuron_taus[i]
            child1.neuron_refractory_periods[i], child2.neuron_refractory_periods[i] = child2.neuron_refractory_periods[i], child1.neuron_refractory_periods[i]

    # Synapse attribute crossover
    if method == 'uniform':
        for i in range(min(len(child1.synapse_weights), len(child2.synapse_weights))):
            if random.random() < prob:
                child1.synapse_weights[i], child2.synapse_weights[i] = child2.synapse_weights[i], child1.synapse_weights[i]
                child1.synapse_pre_rates[i], child2.synapse_pre_rates[i] = child2.synapse_pre_rates[i], child1.synapse_pre_rates[i]
                child1.synapse_post_rates[i], child2.synapse_post_rates[i] = child2.synapse_post_rates[i], child1.synapse_post_rates[i]
    elif method == 'n_point':
        points = sorted(random.sample(range(1, min(len(child1.synapse_weights), len(child2.synapse_weights))), n))
        swap = False
        for i in range(min(len(child1.synapse_weights), len(child2.synapse_weights))):
            if i in points:
                swap = not swap
            if swap:
                child1.synapse_weights[i], child2.synapse_weights[i] = child2.synapse_weights[i], child1.synapse_weights[i]
                child1.synapse_pre_rates[i], child2.synapse_pre_rates[i] = child2.synapse_pre_rates[i], child1.synapse_pre_rates[i]
                child1.synapse_post_rates[i], child2.synapse_post_rates[i] = child2.synapse_post_rates[i], child1.synapse_post_rates[i]
    elif method == 'n_random':
        for _ in range(n):
            i = random.randint(0, min(len(child1.synapse_weights), len(child2.synapse_weights)) - 1)
            child1.synapse_weights[i], child2.synapse_weights[i] = child2.synapse_weights[i], child1.synapse_weights[i]
            child1.synapse_pre_rates[i], child2.synapse_pre_rates[i] = child2.synapse_pre_rates[i], child1.synapse_pre_rates[i]
            child1.synapse_post_rates[i], child2.synapse_post_rates[i] = child2.synapse_post_rates[i], child1.synapse_post_rates[i]

    return child1, child2


def _cross_segment(target, source, start, end):
    # Delete neurons and their associated synapses in the target genome
    for i in range(start, end):
        if target.neuron_types[i] != NULL_SENTINEL:
            target._delete_neuron(i)

    # Copy neurons from source to target
    for i in range(start, end):
        if source.neuron_types[i] != NULL_SENTINEL:
            target.neuron_in_synapses[i] = []
            target.neuron_out_synapses[i] = []
            target.neuron_types[i] = source.neuron_types[i]
            target.neuron_biases[i] = source.neuron_biases[i]
            target.neuron_taus[i] = source.neuron_taus[i]
            target.neuron_refractory_periods[i] = source.neuron_refractory_periods[i]

    # Copy synapses associated with the copied neurons
    for i in range(start, end):
        if source.neuron_types[i] != NULL_SENTINEL:
            # Handle outgoing synapses
            for synapse_index in source.neuron_out_synapses[i]:
                if source.synapse_pre[synapse_index] != NULL_SENTINEL:
                    pre = source.synapse_pre[synapse_index]
                    post = source.synapse_post[synapse_index]
                    weight = source.synapse_weights[synapse_index]
                    pre_rate = source.synapse_pre_rates[synapse_index]
                    post_rate = source.synapse_post_rates[synapse_index]

                    if target.validate_synapse(pre, post):
                        new_synapse_index = target.add_synapse(pre, post, weight, pre_rate, post_rate)
                        if new_synapse_index is not None:
                            target.neuron_out_synapses[pre].append(new_synapse_index)
                            target.neuron_in_synapses[post].append(new_synapse_index)

            # Handle incoming synapses
            for synapse_index in source.neuron_in_synapses[i]:
                if source.synapse_post[synapse_index] != NULL_SENTINEL:
                    pre = source.synapse_pre[synapse_index]
                    post = source.synapse_post[synapse_index]
                    weight = source.synapse_weights[synapse_index]
                    pre_rate = source.synapse_pre_rates[synapse_index]
                    post_rate = source.synapse_post_rates[synapse_index]

                    if target.validate_synapse(pre, post):
                        new_synapse_index = target.add_synapse(pre, post, weight, pre_rate, post_rate)
                        if new_synapse_index is not None:
                            target.neuron_out_synapses[pre].append(new_synapse_index)
                            target.neuron_in_synapses[post].append(new_synapse_index)


def topology_crossover(parent1, parent2, n_points=2):
    # Copy the parent genomes
    child1 = parent1.copy(genome_pool)
    child2 = parent2.copy(genome_pool)

    # Extend the shorter lists to match the longer ones
    max_neurons = max(len(child1.neuron_types), len(child2.neuron_types))
    max_synapses = max(len(child1.synapse_pre), len(child2.synapse_pre))

    child1._extend_neurons(max_neurons)
    child2._extend_neurons(max_neurons)
    child1._extend_synapses(max_synapses)
    child2._extend_synapses(max_synapses)

    # Randomly assign crossover points
    crossover_points = sorted(random.sample(range(1, max_neurons), n_points))
    crossover_points = [0] + crossover_points + [max_neurons]

    # Perform crossover
    for i in range(len(crossover_points) - 1):
        start = crossover_points[i]
        end = crossover_points[i + 1]

        if i % 2 == 0:  # Keep this segment in child1, replace in child2
            _cross_segment(child2, child1, start, end)
        else:  # Keep this segment in child2, replace in child1
            _cross_segment(child1, child2, start, end)

    return child1, child2


def crossover(parent1, parent2, attribute_method='uniform', topology_method='uniform',
              attribute_n=2, topology_n=2, attribute_prob=0.5, topology_prob=0.5):
    if random.random() < 0.5:
        # Perform attribute crossover
        child1, child2 = attribute_crossover(parent1, parent2, method=attribute_method,
                                             n=attribute_n, prob=attribute_prob)
    else:
        # Perform topology crossover
        child1, child2 = topology_crossover(parent1, parent2, n_points=topology_n)

    return child1, child2


def calculate_genetic_disparity(genome1, genome2, c_n=1.0, c_b=0.5, c_s=1.0, c_w=0.33, c_pre=0.0825, c_post=0.0825):
    # Calculate neuron disparity
    matching_neurons, disjointed_neurons = find_matching_and_disjointed_neurons(genome1, genome2)
    total_neurons = len(disjointed_neurons) + len(matching_neurons)

    if total_neurons == 0:
        delta_n = 0  # If there are no neurons, consider the neuron disparity as 0
    else:
        D_n = len(disjointed_neurons) / total_neurons

        delta_b = sum(abs(genome1.neuron_biases[i] - genome2.neuron_biases[i]) for i in matching_neurons)
        delta_b /= total_neurons

        delta_n = (c_n * D_n + c_b * delta_b) / 2

    # Calculate synapse disparity
    matching_synapses, disjointed_synapses = find_matching_and_disjointed_synapses(genome1, genome2)
    total_synapses = len(disjointed_synapses) + len(matching_synapses)

    if total_synapses == 0:
        delta_s = 0  # If there are no synapses, consider the synapse disparity as 0
    else:
        D_s = len(disjointed_synapses) / total_synapses

        delta_w = sum(abs(genome1.synapse_weights[i] - genome2.synapse_weights[i]) for i in matching_synapses)
        delta_pre = sum(abs(genome1.synapse_pre_rates[i] - genome2.synapse_pre_rates[i]) for i in matching_synapses)
        delta_post = sum(abs(genome1.synapse_post_rates[i] - genome2.synapse_post_rates[i]) for i in matching_synapses)

        delta_w /= total_synapses
        delta_pre /= total_synapses
        delta_post /= total_synapses

        delta_s = (c_s * D_s + c_w * delta_w + c_pre * delta_pre + c_post * delta_post) / 4

    # Calculate overall genetic disparity
    if total_neurons == 0 and total_synapses == 0:
        return 0  # If both genomes are empty, consider them identical
    elif total_neurons == 0:
        return delta_s  # If there are no neurons, only consider synapse disparity
    elif total_synapses == 0:
        return delta_n  # If there are no synapses, only consider neuron disparity
    else:
        return (delta_s + delta_n) / 2


def find_matching_and_disjointed_neurons(genome1, genome2):
    matching = []
    disjointed = []

    for i in range(max(len(genome1.neuron_types), len(genome2.neuron_types))):
        if i < len(genome1.neuron_types) and i < len(genome2.neuron_types):
            if (genome1.neuron_types[i] != NULL_SENTINEL and genome2.neuron_types[i] != NULL_SENTINEL
                    and genome1.neuron_in_synapses[i] == genome2.neuron_in_synapses[i]
                    and genome1.neuron_out_synapses[i] == genome2.neuron_out_synapses[i]):
                matching.append(i)
            else:
                disjointed.append(i)
        else:
            disjointed.append(i)

    return matching, disjointed


def find_matching_and_disjointed_synapses(genome1, genome2):
    matching = []
    disjointed = []

    for i in range(max(len(genome1.synapse_pre), len(genome2.synapse_pre))):
        if i < len(genome1.synapse_pre) and i < len(genome2.synapse_pre):
            if (genome1.synapse_pre[i] != NULL_SENTINEL and genome2.synapse_pre[i] != NULL_SENTINEL and
                    genome1.synapse_post[i] != NULL_SENTINEL and genome2.synapse_post[i] != NULL_SENTINEL):
                matching.append(i)
            else:
                disjointed.append(i)
        else:
            disjointed.append(i)

    return matching, disjointed


def compatible(genome1, genome2, threshold):
    return calculate_genetic_disparity(genome1, genome2) <= threshold



### TESTS ###

import unittest
import random

class TestOENEASTDPT(unittest.TestCase):
    def setUp(self):
        # Create two simple genomes for testing
        self.genome1 = genome_pool.get()
        self.genome1.configure(3, 2)
        self.genome2 = genome_pool.get()
        self.genome2.configure(3, 2)

        # Add neurons and synapses to both genomes
        for _ in range(2):
            self.genome1.add_neuron('hidden')
            self.genome2.add_neuron('hidden')

        # Add some synapses
        for _ in range(10):
            pre = random.randint(0, 6)
            post = random.randint(0, 6)
            if pre != post:
                self.genome1.add_synapse(pre, post, weight=random.uniform(-1, 1))
                self.genome2.add_synapse(pre, post, weight=random.uniform(-1, 1))

        # Mutate each genome 1000 times
        for _ in range(1000):
            self.genome1.mutate()
            self.genome2.mutate()

    def test_create_and_run_networks(self):
        network1 = OENEASTDPTNetwork(self.genome1)
        network2 = OENEASTDPTNetwork(self.genome2)

        # Run networks for 100 ticks
        for _ in range(100):
            inputs = [random.random() for _ in range(3)]
            output1 = network1.process(inputs)
            output2 = network2.process(inputs)

            self.assertIsInstance(output1, list)
            self.assertIsInstance(output2, list)

    def test_mutation(self):
        original = self.genome1.copy(genome_pool)
        for _ in range(1000):
            self.genome1.mutate()

        self.assertNotEqual(self.genome1.neuron_types, original.neuron_types)
        self.assertNotEqual(self.genome1.neuron_biases, original.neuron_biases)
        self.assertNotEqual(self.genome1.synapse_weights, original.synapse_weights)

    def test_copy_and_mutate(self):
        copy = self.genome1.copy(genome_pool)
        for _ in range(1000):
            copy.mutate()

        self.assertNotEqual(copy.neuron_types, self.genome1.neuron_types)
        self.assertNotEqual(copy.neuron_biases, self.genome1.neuron_biases)
        self.assertNotEqual(copy.synapse_weights, self.genome1.synapse_weights)

    def test_genetic_disparity(self):
        disparity = calculate_genetic_disparity(self.genome1, self.genome2)
        self.assertGreater(disparity, 0)
        self.assertLess(disparity, 1)

    def test_compatibility(self):
        are_compatible = compatible(self.genome1, self.genome2, threshold=0.5)
        self.assertIsInstance(are_compatible, bool)

    def test_crossover(self):
        # Perform single crossover
        child1, child2 = crossover(self.genome1, self.genome2)

        def genome_difference_score(genome1, genome2):
            score = 0
            # Compare neuron types
            score += sum(1 for a, b in zip(genome1.neuron_types, genome2.neuron_types) if a != b)
            # Compare neuron biases
            score += sum(1 for a, b in zip(genome1.neuron_biases, genome2.neuron_biases) if abs(a - b) > 1e-6)
            # Compare synapse weights
            score += sum(1 for a, b in zip(genome1.synapse_weights, genome2.synapse_weights) if abs(a - b) > 1e-6)
            return score

        # Check that children are different from parents and each other
        diff_child1_parent1 = genome_difference_score(child1, self.genome1)
        diff_child1_parent2 = genome_difference_score(child1, self.genome2)
        diff_child2_parent1 = genome_difference_score(child2, self.genome1)
        diff_child2_parent2 = genome_difference_score(child2, self.genome2)
        diff_child1_child2 = genome_difference_score(child1, child2)

        # Assert that there are some differences
        self.assertGreater(diff_child1_parent1 + diff_child1_parent2, 0, "Child1 is identical to both parents")
        self.assertGreater(diff_child2_parent1 + diff_child2_parent2, 0, "Child2 is identical to both parents")
        self.assertGreater(diff_child1_child2, 0, "Children are identical to each other")

        # Check that children have elements from both parents
        def check_inheritance(child, parent1, parent2):
            inherited_from_parent1 = False
            inherited_from_parent2 = False
            for i in range(min(len(child.neuron_types), len(parent1.neuron_types), len(parent2.neuron_types))):
                if child.neuron_types[i] == parent1.neuron_types[i]:
                    inherited_from_parent1 = True
                if child.neuron_types[i] == parent2.neuron_types[i]:
                    inherited_from_parent2 = True
                if inherited_from_parent1 and inherited_from_parent2:
                    return True
            return False

        self.assertTrue(check_inheritance(child1, self.genome1, self.genome2),
                        "Child1 did not inherit from both parents")
        self.assertTrue(check_inheritance(child2, self.genome1, self.genome2),
                        "Child2 did not inherit from both parents")

        # Check that the number of non-NULL neurons is preserved (approximately)
        non_null_parent1 = sum(1 for nt in self.genome1.neuron_types if nt != NULL_SENTINEL)
        non_null_parent2 = sum(1 for nt in self.genome2.neuron_types if nt != NULL_SENTINEL)
        non_null_child1 = sum(1 for nt in child1.neuron_types if nt != NULL_SENTINEL)
        non_null_child2 = sum(1 for nt in child2.neuron_types if nt != NULL_SENTINEL)

        self.assertLess(abs(non_null_child1 - non_null_parent1), 5,
                        "Child1 has significantly different number of neurons")
        self.assertLess(abs(non_null_child2 - non_null_parent2), 5,
                        "Child2 has significantly different number of neurons")

        print(f"Difference scores: "
              f"Child1-Parent1: {diff_child1_parent1}, "
              f"Child1-Parent2: {diff_child1_parent2}, "
              f"Child2-Parent1: {diff_child2_parent1}, "
              f"Child2-Parent2: {diff_child2_parent2}, "
              f"Child1-Child2: {diff_child1_child2}")


if __name__ == '__main__':
    unittest.main()