# coding=utf-8
import datetime
import math
import numpy as np


class Hawkes(object):
    """
    the function of this class is as same as HawkesOrigin while this class optimize the performance by reconstructing
    some code, vectorization and adding cache
    """

    def __init__(self, training_data, test_data, kernel, init_strategy, event_count, time_slot=100, omega=1,
                 init_time=100, max_day=10000):
        """
        Construct a new Hawkes Model
        :param training_data:
        :param test_data:
        Data Structure:
        {id_index: [(event_index, event_time), (event_index, event_time),...]}
        brace means a dictionary, square brackets means lists, brackets means tuples.
        Other notations follow the convention of document.

        event_index: integer, from 1 to n_j, n_j is the length of a sample sequence, smaller the index, earlier the
        event time of a event. Several events can happen simultaneously, thus the event time of two adjacent events
        can be same
        event_time: integer. Define the time of event happens. The time of first event of each sample is assigned as
        100 (day). Other event time is the time interval between current event and first event plus 100.
        event_id: string, each id indicates a independent sample sequence

        :param event_count: the number of unique event type
        :param kernel: kernel function ,'exp' or 'Fourier'
        :param init_strategy: we don't pay attention to it
        :param omega: if kernel is exp, we can set omega by dictionary, e.g. {'omega': 2}, the 'default'
        means omega will be set as 1 automatically
        automatically after initialization procedure accomplished
        :param time_slot: time slot count
        :param init_time: the time of first event
        """
        self.training_data = training_data
        self.test_data = test_data
        self.excite_kernel = kernel
        self.event_count = event_count
        self.init_strategy = init_strategy
        self.time_slot = time_slot
        self.omega = omega
        self.train_log_likelihood_tendency = []
        self.test_log_likelihood_tendency = []
        self.base_intensity = self.initialize_base_intensity()
        self.mutual_intensity = self.initialize_mutual_intensity()
        self.auxiliary_variable = self.initialize_auxiliary_variable()
        self.initial_time = init_time
        self.max_day = max_day
        self.auxiliary_variable_denominator = None
        self.mu_nominator_vector = None
        self.mu_denominator_vector = None
        self.alpha_denominator_matrix = None
        self.alpha_nominator_matrix = None
        self.discrete_time_decay = None
        self.discrete_time_integral = None
        if self.excite_kernel == 'fourier' or self.excite_kernel == 'Fourier':
            self.count_of_each_slot = self.event_count_of_each_slot_function()
            self.count_of_each_event = self.event_count_of_each_event_function()
            self.y_omega = self.y_omega_calculate()
            self.k_omega_cache = self.k_omega_cache_calculate()
            self.k_omega = self.k_omega_update()
        print("Hawkes Process Model Initialize Accomplished")

    def initialize_base_intensity(self):
        base_intensity = None
        if self.init_strategy == 'default':
            base_intensity = np.random.uniform(0.1, 1, [self.event_count, 1])
        else:
            pass

        if base_intensity is None:
            raise RuntimeError('illegal initial strategy')
        return base_intensity

    def initialize_mutual_intensity(self):
        mutual_excite_intensity = None
        if self.init_strategy == 'default':
            mutual_excite_intensity = np.random.uniform(0.1, 1, [self.event_count, self.event_count])
        else:
            pass

        if mutual_excite_intensity is None:
            raise RuntimeError('illegal initial strategy')
        return mutual_excite_intensity

    def initialize_auxiliary_variable(self):
        """
        consult eq. 8, 9, 10
        for the i-th event in a sample list, the length of corresponding auxiliary variable list is i too.
        The j-th entry of auxiliary variable list (j<i) means the probability that the i-th event of the sample sequence
        is 'triggered' by the j-th event. The i-th entry of auxiliary variable list indicates the probability that
        the i-th event are triggered by base intensity. Obviously, the sum of one list should be one

        when initialize auxiliary variable, we assume the value of all entries in a list are same

        :return: auxiliary_map, data structure { id : { event_no: [auxiliary_map_i]}}
        auxiliary_map_i [1-st triggered, ..., base triggered]
        """
        auxiliary_map = {}
        for sequence_id in self.training_data:
            auxiliary_event_map = {}
            list_length = len(self.training_data[sequence_id])
            for i in range(0, list_length):
                single_event_auxiliary_list = []
                for j in range(-1, i):
                    single_event_auxiliary_list.append(1 / (i + 1))
                auxiliary_event_map[i] = single_event_auxiliary_list
            auxiliary_map[sequence_id] = auxiliary_event_map
        return auxiliary_map

    def event_count_of_each_event_function(self):
        count_vector = np.zeros([self.event_count, 1])
        for j in self.training_data:
            for event in self.training_data[j]:
                event_index = event[0]
                count_vector[event_index][0] += 1
        return count_vector

    def event_count_of_each_slot_function(self):
        event_list_full = []
        for j in self.training_data:
            event_list = self.training_data[j]
            for event in event_list:
                event_list_full.append(event)
        event_list_full = sorted(event_list_full, key=lambda event_time: event_time[1])
        # time unit is a float number
        time_unit = (event_list_full[-1][1] - event_list_full[0][1]) / self.time_slot

        count_list = np.zeros([self.time_slot, 1])
        for item in event_list_full:
            slot_index = int((item[1] - self.initial_time) / time_unit)
            if slot_index == len(count_list):
                count_list[slot_index - 1] += 1
            else:
                count_list[slot_index] += 1
        return count_list

    # update y, k_omega
    def k_omega_cache_calculate(self):
        cache = np.zeros([self.event_count, self.time_slot], dtype=np.complex64)
        omega = np.arange(0, 2 * np.pi, 2 * np.pi / self.time_slot)
        for j in self.training_data:
            for item in self.training_data[j]:
                event_index = item[0]
                event_time = item[1]
                cache[event_index] += np.exp(-1 * complex(0, 1) * omega * event_time)
        return cache

    def k_omega_update(self):
        # calculate denominator
        k_denominator = np.zeros([self.time_slot, 1], dtype=np.complex64)
        for k in range(0, self.time_slot):
            if k == 0:
                mutual_in = self.mutual_intensity
                count_event = self.count_of_each_event
                denominator = np.matmul(mutual_in, count_event).sum()
                k_denominator[k][0] = denominator
            else:
                cache = self.k_omega_cache[:, k]
                mutual = self.mutual_intensity
                k_denominator[k][0] = np.dot(cache, mutual).sum()
        k_nominator = np.zeros([self.time_slot, 1], dtype=np.complex64)
        for k in range(0, self.time_slot):
            if k == 0:
                k_nominator[k][0] = self.y_omega[k][0] - np.pi * 2 * self.base_intensity.sum()
            else:
                k_nominator[k][0] = self.y_omega[k][0]

        self.k_omega = k_nominator / k_denominator
        return k_nominator / k_denominator

    def y_omega_calculate(self):
        y_omega = np.zeros([self.time_slot, 1], dtype=np.complex128)
        for k in range(0, self.time_slot):
            omega_k = 2 * math.pi / self.time_slot * k
            y_omega_k = 0
            for i in range(0, self.time_slot):
                y_omega_k += np.exp(-1 * omega_k * i * complex(0, 1)) * self.count_of_each_slot[i]
            y_omega[k][0] = y_omega_k
        return y_omega

    # EM Algorithm
    def maximization_step(self):
        self.alpha_nominator_update()
        self.alpha_denominator_update()
        self.mu_nominator_update()
        self.mu_denominator_update()
        self.mutual_intensity = self.alpha_nominator_matrix / self.alpha_denominator_matrix
        self.base_intensity = self.mu_nominator_vector / self.mu_denominator_vector

    def alpha_nominator_update(self):
        alpha_matrix = np.zeros([self.event_count, self.event_count])
        data_source = self.training_data
        for j in data_source:
            list_length = len(data_source[j])
            for i in range(1, list_length):
                for k in range(0, i):
                    i_event_index = data_source[j][i][0]
                    k_event_index = data_source[j][k][0]
                    alpha_matrix[i_event_index][k_event_index] += self.auxiliary_variable[j][i][k]
        self.alpha_nominator_matrix = alpha_matrix

    def alpha_denominator_update(self):
        alpha_matrix = np.zeros([self.event_count, self.event_count])
        for j in self.training_data:
            event_list = self.training_data[j]
            # for numerical stability, we add 1
            last_event_time = event_list[-1][1] + 1
            for l in range(0, self.event_count):
                for k in range(0, len(event_list)):
                    k_event_index = event_list[k][0]
                    k_event_time = event_list[k][1]
                    integral = self.discrete_time_integral[last_event_time - k_event_time]
                    alpha_matrix[l][k_event_index] += integral
        self.alpha_denominator_matrix = alpha_matrix

    def mu_nominator_update(self):
        nominator = np.zeros([self.event_count, 1])
        for j in self.training_data:
            event_list = self.training_data[j]
            for i in range(0, len(event_list)):
                i_event_index = event_list[i][0]
                nominator[i_event_index][0] += self.auxiliary_variable[j][i][i]
        self.mu_nominator_vector = nominator

    def mu_denominator_update(self):
        denominator = 0
        for j in self.training_data:
            event_list = self.training_data[j]
            first_time = event_list[0][1]
            last_time = event_list[-1][1]
            denominator += last_time - first_time
        self.mu_denominator_vector = denominator

    # E Step
    def expectation_step(self):
        self.auxiliary_variable_denominator_update()
        for j in self.training_data:
            list_length = len(self.training_data[j])
            for i in range(0, list_length):
                for l in range(0, i):
                    self.auxiliary_variable[j][i][l] = self.calculate_q_il(j=j, i=i, _l=l)
                self.auxiliary_variable[j][i][i] = self.calculate_q_ii(j=j, i=i)

    def calculate_q_il(self, j, i, _l):
        """
        according to eq. 10
        :param j:
        :param i:
        :param _l: the underline is added to eliminate the ambiguous
        :return:
        """

        event_list = self.training_data[j]
        i_event_index = event_list[i][0]
        i_event_time = event_list[i][1]
        l_event_index = event_list[_l][0]
        l_event_time = event_list[_l][1]
        alpha = self.mutual_intensity[i_event_index][l_event_index]
        kernel = self.discrete_time_decay[i_event_time - l_event_time]

        nominator = alpha * kernel
        denominator = self.auxiliary_variable_denominator[j][i]
        return nominator / denominator

    def calculate_q_ii(self, j, i):
        """
        according to eq. 9
        :param j:
        :param i:
        :return:
        """
        event_list = self.training_data[j]
        i_event_index = event_list[i][0]
        nominator = self.base_intensity[i_event_index][0]
        denominator = self.auxiliary_variable_denominator[j][i]
        q_ii = nominator / denominator
        return q_ii

    def auxiliary_variable_denominator_update(self):
        denominator_map = {}
        for j in self.training_data:
            event_list = self.training_data[j]
            single_denominator_map = {}
            for i in range(0, len(event_list)):
                i_event_index = event_list[i][0]
                i_event_time = event_list[i][1]

                denominator = 0
                denominator += self.base_intensity[i_event_index][0]
                for l in range(0, i):
                    l_event_index = event_list[l][0]
                    l_event_time = event_list[l][1]
                    alpha = self.mutual_intensity[i_event_index][l_event_index]
                    kernel = self.discrete_time_decay[i_event_time - l_event_time]
                    denominator += alpha * kernel
                single_denominator_map[i] = denominator
            denominator_map[j] = single_denominator_map

        self.auxiliary_variable_denominator = denominator_map

    def kernel_calculate(self, early_event_time, late_event_time):
        kernel_type = self.excite_kernel
        if kernel_type == 'default' or kernel_type == 'exp':
            if self.omega is None:
                raise RuntimeError('omega lost')
            omega = self.omega
            kernel_value = np.exp(-1 * omega * (late_event_time - early_event_time))
            return kernel_value
        elif kernel_type == 'fourier' or kernel_type == 'Fourier':
            exp = np.exp(complex(0, 1) * (late_event_time - early_event_time) * np.arange(0, 2 * np.pi,
                                                                                          2 * np.pi / self.time_slot))
            kappa = (exp * self.k_omega).sum()
            return abs(kappa)
        else:
            raise RuntimeError('illegal kernel name')

    def kernel_integral(self, upper_bound, lower_bound):
        if upper_bound < lower_bound:
            raise RuntimeError("upper bound smaller than lower bound")

        kernel_type = self.excite_kernel
        if kernel_type == 'default' or kernel_type == 'exp':
            if self.omega is None:
                raise RuntimeError('illegal hyper_parameter, omega lost')
            omega = self.omega
            kernel_integral = (np.exp(-1 * omega * lower_bound) - math.exp(-1 * omega * upper_bound)) / omega
            return kernel_integral
        elif kernel_type == 'fourier' or kernel_type == 'Fourier':
            # the calculate equations are different when k=0

            # for k>0
            omega = np.arange(2 * np.pi / self.time_slot, 2 * np.pi, 2 * np.pi / self.time_slot)
            first = self.k_omega[1:, ]
            middle = complex(0, 1) / omega
            last = 1 - np.exp(complex(0, 1) * omega * (upper_bound - lower_bound))
            kernel_integral = (first * middle * last).sum()

            # for k=0
            kernel_integral += self.k_omega[0][0] * (upper_bound - lower_bound)
            return abs(kernel_integral / self.time_slot)
        else:
            raise RuntimeError('illegal kernel name')

    def optimization(self, iteration):

        # initialize likelihood
        update_time_decay_start = datetime.datetime.now()
        self.update_discrete_time_decay_function()
        self.update_discrete_integral_function()
        update_time_decay_end = datetime.datetime.now()
        likelihood_star_time = datetime.datetime.now()
        train_log_likelihood = self.log_likelihood_calculate(self.training_data)
        test_log_likelihood = self.log_likelihood_calculate(self.test_data)
        likelihood_end_time = datetime.datetime.now()
        likelihood_time = str((likelihood_end_time - likelihood_star_time).seconds)
        update_time_decay = str((update_time_decay_end - update_time_decay_start).seconds)

        self.train_log_likelihood_tendency.append(train_log_likelihood)
        self.test_log_likelihood_tendency.append(test_log_likelihood)
        print(self.excite_kernel + "_" + 'iteration: ' + str(0) + ',test likelihood = ' +
              str(test_log_likelihood) + ',train likelihood = ' + str(train_log_likelihood) + " optimize time " +
              "None" + " seconds. likelihood time " + likelihood_time + " seconds. update time: " +
              update_time_decay + "seconds")

        for i in range(1, iteration + 1):
            # EM Algorithm
            if self.excite_kernel == 'fourier' or self.excite_kernel == 'Fourier':
                self.k_omega_update()
            update_time_decay_start = datetime.datetime.now()
            self.update_discrete_time_decay_function()
            self.update_discrete_integral_function()
            update_time_decay_end = datetime.datetime.now()

            optimize_start_time = datetime.datetime.now()
            self.expectation_step()
            self.maximization_step()
            optimize_end_time = datetime.datetime.now()

            likelihood_star_time = datetime.datetime.now()
            train_log_likelihood = self.log_likelihood_calculate(self.training_data)
            test_log_likelihood = self.log_likelihood_calculate(self.test_data)
            likelihood_end_time = datetime.datetime.now()

            optimize_time = str((optimize_end_time - optimize_start_time).seconds)
            likelihood_time = str((likelihood_end_time - likelihood_star_time).seconds)
            update_time_decay = str((update_time_decay_end - update_time_decay_start).seconds)
            self.train_log_likelihood_tendency.append(train_log_likelihood)
            self.test_log_likelihood_tendency.append(test_log_likelihood)
            print(self.excite_kernel + "_" + 'iteration: ' + str(i) + ',test likelihood = ' +
                  str(test_log_likelihood) + ',train likelihood = ' + str(train_log_likelihood) + " optimize time " +
                  optimize_time + " seconds. likelihood time " + likelihood_time + " seconds. update time: " +
                  update_time_decay + "seconds")

        print("optimization accomplished")

    # calculate log-likelihood
    def log_likelihood_calculate(self, data_source):
        """
        according to eq. 6
        calculate the log likelihood based on current parameter
        :return:
        """
        log_likelihood = 0

        # calculate part one of log-likelihood
        # according to equation 6
        for j in data_source:
            list_length = len(data_source[j])

            # part 1
            for i in range(0, list_length):
                part_one = self.part_one_calculate(j=j, i=i, data_source=data_source)
                log_likelihood += part_one

            # part 2
            for u in range(0, self.event_count):
                part_two = self.part_two_calculate(j=j, u=u, data_source=data_source)
                log_likelihood -= part_two
        return log_likelihood

    def part_one_calculate(self, j, i, data_source):
        """
        according to of eq. 7
        :param j:
        :param i:
        :param data_source:
        :return:
        """
        part_one = 0

        i_event_index = data_source[j][i][0]
        i_event_time = data_source[j][i][1]
        mu = self.base_intensity[i_event_index][0]
        part_one += mu

        for l in range(0, i):
            l_event_index = data_source[j][l][0]
            l_event_time = data_source[j][l][1]

            alpha = self.mutual_intensity[i_event_index][l_event_index]
            kernel = self.discrete_time_decay[i_event_time - l_event_time]
            part_one += alpha * kernel

        part_one = math.log(part_one)
        return part_one

    def part_two_calculate(self, u, j, data_source):
        """
        according to eq. 12
        :param u:
        :param j:
        :param data_source:
        :return:
        """
        part_two = 0
        last_event_time = data_source[j][-1][1]
        first_event_time = data_source[j][0][1]
        part_two += self.base_intensity[u][0] * (last_event_time - first_event_time)

        for k in range(0, len(data_source[j])):
            k_event_index = data_source[j][k][0]
            k_event_time = data_source[j][k][1]

            lower_bound = 0
            upper_bound = last_event_time - k_event_time
            alpha = self.mutual_intensity[u][k_event_index]

            part_two += alpha * self.discrete_time_integral[upper_bound - lower_bound]

        return part_two

    def update_discrete_time_decay_function(self):
        discrete_function = []
        for i in range(0, self.max_day):
            intensity = self.kernel_calculate(0, i)
            discrete_function.append(intensity)
        discrete_function = np.array(discrete_function)
        self.discrete_time_decay = discrete_function

    def update_discrete_integral_function(self):
        discrete_integral_function = []
        for i in range(0, self.max_day):
            integral = self.kernel_integral(lower_bound=0, upper_bound=i)
            discrete_integral_function.append(integral)
        discrete_integral_function = np.array(discrete_integral_function)
        self.discrete_time_integral = discrete_integral_function


def unit_test():
    pass


if __name__ == "__main__":
    unit_test()
