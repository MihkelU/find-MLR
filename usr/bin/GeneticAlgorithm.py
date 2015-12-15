#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import xlsxwriter as xls
import time


from copy import deepcopy
from functools import reduce

class Evolution():

    def __init__(self, matrix, reduce, mutation_rate, crossover_rate, population_size,
                 N_of_generations, N_of_features, N_of_models, orthogonality, queue):

        self.DP_Matrix = np.array(matrix)
        self.reduce = reduce

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.N_of_generations = N_of_generations
        self.population_size = population_size
        self.N_of_features = N_of_features
        self.N_of_models = N_of_models
        self.orthogonality = orthogonality

        self.queue = queue


        # Time measurements of different functions
        self.t_fitness, self.t_mutation, self.t_crossover, self.t_ortdes = 0, 0, 0, 0
        self.t_initial = time.time()

        self.initData()
        self.queue.put("Initial number of descriptors: %s\n"%(len(self.D_names)))

        if reduce == True:
            self.reduceDescriptors()
            self.N_remaining_descriptors = len(list(self.D_names_values_dict.keys()))
            self.queue.put("Number of descriptors remaining after "
                           "removing integer descriptors: {}\n".format(self.N_remaining_descriptors))

        self.correlation_matrix()
        self.N_remaining_descriptors = len(list(self.D_names_values_dict.keys()))
        self.queue.put("Number of descriptors remaining after "
                       "removing equal descriptors: {}\n\n".format(self.N_remaining_descriptors))

        ### The Genetic Algorithm part ###
        self.current_population = self.initPopulation()
        self.N_of_genes = len(self.current_population[0])
        self.chromosome_mutation_rate = self.mutation_rate * self.N_of_genes

        self.fitnesses = [self.fitnessFunction(chromosome) for chromosome in self.current_population]
        self.fitnesses = [self.fitnessFunction(chrom) for chrom in self.current_population]
        self.sum_of_fitnesses = sum(self.fitnesses)
        self.max_fitness = max(self.fitnesses)

        self.elite_population = []
        self.pick_elites()


        self.Evolve()


    def initData(self):

        self.D_names = self.DP_Matrix[0,1:(len(self.DP_Matrix[0])-1)]
        self.S = self.DP_Matrix[1:,0]

        self.D_values = self.DP_Matrix[1:,1:(len(self.DP_Matrix[0])-1)].astype(np.float32, copy=False)
        self.P = self.DP_Matrix[1:,-1].astype(np.float32, copy=False)

        self.D_values_cols = self.D_values.T

        self.D_names_values_dict = dict(list(zip(self.D_names, self.D_values_cols)))
        self.idx_D_names_dict = dict(list(zip(range(0,len(self.D_names)), self.D_names)))
        self.idx_D_values_dict = dict(list(zip(range(0,len(self.D_names)), self.D_values_cols)))


    def reduceDescriptors(self):

        isinteger = lambda x: np.equal(np.mod(x,1),0)
        for key in self.D_names:
            col = self.D_names_values_dict[key]
            test_int_col = all([isinteger(col[i]) for i in range(len(col))])
            if test_int_col == True:
                del self.D_names_values_dict[key]

        self.D_names = sorted(list(self.D_names_values_dict.keys()))
        self.idx_D_names_dict = dict(list(zip(range(0,len(self.D_names)), self.D_names)))
        self.idx_D_values_dict = {}
        for key in self.idx_D_names_dict:
            self.idx_D_values_dict[key] = self.D_names_values_dict[self.idx_D_names_dict[key]]

        self.D_names_values_dict_copy = deepcopy(self.D_names_values_dict)


    def correlation_matrix(self):

        t0 = time.time()
        redundant_descriptor_indices = []
        self.orthogonality_matrix = np.empty([self.N_remaining_descriptors, self.N_remaining_descriptors], dtype=float)
        self.name_orthogonality_dict = {}
        for i in range(self.N_remaining_descriptors):
            for j in range(self.N_remaining_descriptors):
                if j > i:
                    descriptor_pair_correlation = self.r_k(self.idx_D_values_dict[i], self.idx_D_values_dict[j])**2

                    if descriptor_pair_correlation > self.orthogonality:
                        self.orthogonality_matrix[i,j] = 0
                        self.orthogonality_matrix[j,i] = 0
                    else:
                        self.orthogonality_matrix[i,j] = 1
                        self.orthogonality_matrix[j,i] = 1

                    if descriptor_pair_correlation >= 1:
                        if j not in redundant_descriptor_indices:
                            redundant_descriptor_indices.append(j)

            self.orthogonality_matrix[i,i] = 0
            self.name_orthogonality_dict[self.idx_D_names_dict[i]] = self.orthogonality_matrix[i]

        self.orthogonality_matrix = np.delete(self.orthogonality_matrix, redundant_descriptor_indices, 0)
        self.orthogonality_matrix = np.delete(self.orthogonality_matrix, redundant_descriptor_indices, 1)

        for idx in redundant_descriptor_indices:
            name = self.idx_D_names_dict[idx]
            del self.D_names_values_dict[name]
            del self.name_orthogonality_dict[name]

        self.D_names = sorted(list(self.D_names_values_dict.keys()))
        self.idx_D_names_dict = dict(list(zip(range(0,len(self.D_names)), self.D_names)))

        self.idx_D_values_dict = {}
        for key in self.idx_D_names_dict:
            self.idx_D_values_dict[key] = self.D_names_values_dict[self.idx_D_names_dict[key]]

        t1 = time.time()
        self.t_ortdes += (t1-t0)


    def orthogonality_test(self, chromosome):

        idxs = np.flatnonzero(chromosome)
        for idx1 in idxs:
            for idx2 in idxs:
                if idx2 > idx1:
                    if self.orthogonality_matrix.item(idx1,idx2) == 1:
                        continue
                    else:
                        return False
        else:
            return True


    def initPopulation(self):

        population = []
        for i in range(self.population_size):
            population.append(self.initChromosome())

        return np.array(population)


    # def initChromosome(self):
    #
    #     N_ones = self.N_of_features
    #     N_zeros = len(self.D_names) - N_ones
    #     chromosome = np.array([1]*N_ones + [0]*N_zeros)
    #     while True:
    #         np.random.shuffle(chromosome)
    #         if self.orthogonality_test(chromosome) == True:
    #             return chromosome


    def initChromosome(self):

        flip = lambda x: 1-x
        chromosome = np.zeros(self.N_remaining_descriptors)
        first_descriptor_idx = np.random.randint(self.N_remaining_descriptors)
        chromosome[first_descriptor_idx] = flip(chromosome[first_descriptor_idx])

        idxs = np.flatnonzero(chromosome)
        for i in range(self.N_of_features-1): # First descriptor already chosen

            orthogonality_rows = [np.flatnonzero(self.orthogonality_matrix[idx]) for idx in idxs]
            pickable_descriptors = reduce(np.intersect1d, orthogonality_rows )
            new_descriptor_idx = np.random.choice(pickable_descriptors)
            chromosome[new_descriptor_idx] = flip(chromosome[new_descriptor_idx])
            idxs = np.append(idxs, new_descriptor_idx)

        return chromosome

    def fitnessFunction(self, chromosome):

        t0 = time.time()
        indexes = np.flatnonzero(chromosome)
        columns = [self.idx_D_values_dict[idx] for idx in indexes]

        bias = np.ones(len(columns[0]))
        X = np.vstack((bias,columns)).transpose()
        b = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()), self.P)

        Y_pred = np.dot(X,b.transpose())
        rk = self.r_k(self.P, Y_pred)

        t1 = time.time()
        self.t_fitness += (t1-t0)
        return rk**2


    def r_k(self, Y_obs, Y_pred):

        Y_pred_mean = np.average(Y_pred)
        Y_obs_mean = np.average(Y_obs)

        rk = np.dot((Y_pred - Y_pred_mean).transpose(),(Y_obs - Y_obs_mean))
        rk /= np.sqrt(np.dot((Y_pred - Y_pred_mean).transpose(),(Y_pred - Y_pred_mean)))
        rk /= np.sqrt(np.dot((Y_obs - Y_obs_mean).transpose(),(Y_obs - Y_obs_mean)))
        return rk.item(0)

    # def r_c(self, Y_obs, X):
    #
    #     Y_predicted = []
    #     for i in range(len(Y_obs)):
    #         X_training = np.delete(X, i, 0)
    #         Y_training = np.delete(Y_obs, i, 0)
    #         b = np.dot(np.dot(np.linalg.inv(np.dot(X_training.transpose(),X_training)),X_training.transpose()),Y_training)
    #         Y_prediction = np.dot(X[i],b.transpose())
    #         Y_predicted.append(Y_prediction)
    #
    #     Y_predicted = np.array(Y_predicted)
    #     Y_pred_mean = np.average(Y_predicted)
    #     Y_obs_mean = np.average(Y_obs)
    #
    #     rc = np.dot((Y_predicted - Y_pred_mean).transpose(),(Y_obs - Y_obs_mean))
    #     rc /= np.sqrt(np.dot((Y_predicted - Y_pred_mean).transpose(),(Y_predicted - Y_pred_mean)))
    #     rc /= np.sqrt(np.dot((Y_obs - Y_obs_mean).transpose(),(Y_obs - Y_obs_mean)))
    #     return rc.item(0)

    ### Implementing roulette-wheel selection via stochastic acceptance
    def roulette(self):

        while True:
            idx = np.random.randint(0,len(self.fitnesses))
            probability = self.fitnesses[idx]/self.max_fitness
            if probability > np.random.uniform(0,1):
                return idx

    ### Using two-point crossover
    def crossover(self, population):

        ch1 = population[self.roulette()]
        ch2 = population[self.roulette()]

        if np.random.uniform(0,1) < self.crossover_rate:
            while True:
                idxs = np.random.randint(0,len(ch1),(1,2))
                k = np.amin(idxs)
                l = np.amax(idxs)
                new_ch1 = np.concatenate((ch1[:k],ch2[k:l],ch1[l:]))
                new_ch2 = np.concatenate((ch2[:k],ch1[k:l],ch2[l:]))
                if (np.count_nonzero(new_ch1) == self.N_of_features and np.count_nonzero(new_ch2) == self.N_of_features)\
                    and (self.orthogonality_test(new_ch1) and self.orthogonality_test(new_ch2)):
                    return new_ch1, new_ch2
        else:
            return ch1, ch2

    ### Mutate each chromosome only once
    # Using completely random method for selecting orthogonal descriptors

    # def mutate(self, chromosome):
    #
    #     flip = lambda x: 1-x
    #
    #     if np.random.uniform(0,1) < self.chromosome_mutation_rate:
    #         while True:
    #             idx1 = np.random.randint(self.N_of_genes)
    #             if chromosome[idx1] == 1:
    #                 chromosome[idx1] = flip(chromosome[idx1])
    #                 idx2 = np.random.choice(np.where(chromosome==0)[0])
    #                 chromosome[idx2] = flip(chromosome[idx2])
    #                 if self.orthogonality_test(chromosome):
    #                     break
    #             elif chromosome[idx1] == 0:
    #                 chromosome[idx1] = flip(chromosome[idx1])
    #                 idx2 = np.random.choice(np.where(chromosome==1)[0])
    #                 chromosome[idx2] = flip(chromosome[idx2])
    #                 if self.orthogonality_test(chromosome):
    #                     break

    # Use semi-deterministic method for selecting orthogonal descriptors.
    def mutate(self, chromosome):

        flip = lambda x: 1-x

        if np.random.uniform(0,1) < self.chromosome_mutation_rate:

            # Remove one descriptor randomly
            des_idxs = np.flatnonzero(chromosome)
            removed_descriptor_idx = np.random.choice(np.arange(len(des_idxs)))
            removed_descriptor = des_idxs[removed_descriptor_idx]
            chromosome[removed_descriptor] = flip(chromosome[removed_descriptor])
            des_idxs = np.delete(des_idxs, removed_descriptor_idx)

            orthogonal_rows = [np.flatnonzero(self.orthogonality_matrix[idx]) for idx in des_idxs]
            orthogonal_descriptors = reduce(np.intersect1d, orthogonal_rows)
            pickable_descriptors = np.delete(orthogonal_descriptors, des_idxs)
            new_descriptor = np.random.choice(pickable_descriptors)
            chromosome[new_descriptor] = flip(chromosome[new_descriptor])

    def breed(self, population):

        t0 = time.time()
        ch1, ch2 = self.crossover(population)
        t1 = time.time()
        self.t_crossover += (t1-t0)

        t0 = time.time()
        self.mutate(ch1)
        self.mutate(ch2)
        t1 = time.time()
        self.t_mutation += (t1-t0)

        return ch1, ch2


    def pick_elites(self):

        fitness_chromosome_dict = dict(list(zip(self.fitnesses,self.current_population)))
        fitnesses_set = set(self.fitnesses)
        elite_fitnesses_list = sorted(list(fitnesses_set))[-self.N_of_models:]
        self.elite_population = [fitness_chromosome_dict[fitness] for fitness in elite_fitnesses_list]


    def Evolve(self):

        self.queue.put("Iterations:\n")
        N_iterations = 0
        next_population = deepcopy(self.elite_population)

        while N_iterations < self.N_of_generations:
            while len(next_population) < self.population_size:
                next_population.extend(self.breed(self.current_population))

            else:
                self.current_population = deepcopy(next_population)
                self.fitnesses = [self.fitnessFunction(chrom) for chrom in self.current_population]
                self.max_fitness = max(self.fitnesses)
                self.sum_of_fitnesses = sum(self.fitnesses)
                self.pick_elites()
                next_population = deepcopy(self.elite_population)

                self.queue.put("Number of generation - %s; Best fitness - %s\n"%(N_iterations+1,
                                                                                 round(self.max_fitness,5)))
                N_iterations += 1

        else:

            self.t_final = time.time()
            self.time_elapsed = self.t_final-self.t_initial

            self.write_outputfiles()
            self.queue.put(" Genetic Algorithm finished! ".center(50,"#"))
            self.queue.put("\n\n")
            self.queue.put(" Results ".center(50,"="))
            self.queue.put("\nTotal time elapsed %s\n" % (self.time_elapsed))
            self.queue.put("Orthogonality matrix creation time: %s\n" % (self.t_ortdes))
            self.queue.put("Fitness evaluation time: %s\n" % (self.t_fitness))
            self.queue.put("Mutation time: %s\n" % (self.t_mutation))
            self.queue.put("Crossover time: %s\n\n" % (self.t_crossover))

            self.queue.put("%i best models found:\n"% (self.N_of_models))
            for chromosome in self.elite_population:
                idxs = np.flatnonzero(chromosome)
                descriptors = [self.idx_D_names_dict[idx].strip() for idx in idxs]
                line = "Descriptors: "
                for descriptor in descriptors:
                    line += descriptor + " "
                line += "; R2="
                line += str(round(self.fitnessFunction(chromosome),5))
                line += "\n"
                self.queue.put(line)



    def write_outputfiles(self):

        workbook = xls.Workbook("Output.xlsx")

        number_format = workbook.add_format({"num_format":"#.######"})
        bold = workbook.add_format({'bold': True})
        for i in range(self.N_of_models):
            worksheet = workbook.add_worksheet("model%s"%(i+1))

            current_model = self.elite_population[-i-1]
            idxs = np.flatnonzero(current_model)
            columns = [self.idx_D_values_dict[idx] for idx in idxs]

            bias = np.ones(len(columns[0]))
            X = np.vstack((bias,columns)).transpose()
            coefficients = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()), self.P)
            for j in range(len(idxs)):

                current_descriptor_values = columns[j]
                worksheet.write(0, j, self.idx_D_names_dict[idxs[j]])
                for k in range(len(current_descriptor_values)):
                    worksheet.write(1+k, j, current_descriptor_values[k], number_format)

            for i, b in enumerate(coefficients):
                worksheet.write(i, len(idxs)+2, "b%s"%(i), bold)
                worksheet.write(i, len(idxs)+3, b, number_format)

        workbook.close()

        with open("outputfile.txt","w") as f:
            f.write("Programm lõpetas")
            pass
        pass

