#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import time

from copy import deepcopy

class Evolution():

    def __init__(self, matrix, reduce, mutation_rate, crossover_rate, population_size,
                 N_of_generations, N_of_features, N_of_models):

        self.DP_Matrix = np.array(matrix)
        self.reduce = reduce
        self.lower_bound_collinearity = 0.999

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.N_of_generations = N_of_generations
        self.population_size = population_size
        self.N_of_features = N_of_features
        self.N_of_models = N_of_models

        self.initData()
        if reduce == True:
            self.reduceDescriptors()

        ### The Genetic Algorithm part ###
        t1 = time.clock()
        self.current_population = self.initPopulation()
        self.fitnesses = [self.fitnessFunction(chromosome) for chromosome in self.current_population]

        self.sum_of_fitnesses = sum(self.fitnesses)

        self.max_fitness = max(self.fitnesses)
        self.elite_population = []
        self.pick_elites()

        for elite in self.elite_population:
            print(self.fitnessFunction(elite))
        t2 = time.clock()
        print("initPoptime: ",t2-t1)


        self.best_models = set()
        time1 = time.clock()
        self.Evolve()
        time2 = time.clock()

        print("Time elapsed:", time2-time1)
        print(len(self.best_models))
        for model in sorted(list(self.best_models))[-10:]:
            print(model)

    def initData(self):

        self.D_names = self.DP_Matrix[0,1:(len(self.DP_Matrix[0])-1)]
        self.S = self.DP_Matrix[1:,0]

        self.D_values = self.DP_Matrix[1:,1:(len(self.DP_Matrix[0])-1)].astype(np.float32, copy=False)
        self.P = self.DP_Matrix[1:,-1].astype(np.float32, copy=False)

        self.D_values_cols = self.D_values.T

        self.D_names_values_dict = dict(list(zip(self.D_names, self.D_values_cols)))
        self.idx_D_names_dict = dict(list(zip(range(0,len(self.D_names)), self.D_names)))


    def reduceDescriptors(self):

        isinteger = lambda x: np.equal(np.mod(x,1),0)
        for key in self.D_names:
            col = self.D_names_values_dict[key]
            test_int_col = all([isinteger(col[i]) for i in range(len(col))])
            if test_int_col == True:
                del self.D_names_values_dict[key]
        self.D_names = sorted(list(self.D_names_values_dict.keys()))
        self.idx_D_names_dict = dict(list(zip(range(0,len(self.D_names)), self.D_names)))

        self.D_names_values_dict_copy = deepcopy(self.D_names_values_dict)

        for idx1 in range(len(self.D_names)):
            name1 = self.idx_D_names_dict[idx1]
            des_val1 = self.D_names_values_dict[name1]
            names, des_vals = [name1], [des_val1]
            for idx2 in range(idx1+1,len(self.D_names)):
                name2 = self.idx_D_names_dict[idx2]
                des_val2 = self.D_names_values_dict[name2]
                if self.r_k(des_val2,des_val1)**2 > self.lower_bound_collinearity:
                    des_vals.append(des_val2)
                    names.append(name2)

            corrs = []
            for des in des_vals:
                corrs.append(self.r_k(self.P, des)**2)

            names_corrs = dict(list(zip(names, corrs)))
            max_corr = max(corrs)
            for name in names:
                if names_corrs[name] < max_corr:
                    try:
                        del self.D_names_values_dict_copy[name]
                    except:
                        pass
                elif names_corrs[name] == max_corr and len(corrs) > 1:
                    try:
                        del self.D_names_values_dict_copy[name]
                    except:
                        pass


        self.D_names_values_dict = deepcopy(self.D_names_values_dict_copy)
        self.D_names = sorted(list(self.D_names_values_dict.keys()))
        self.idx_D_names_dict = dict(list(zip(range(0,len(self.D_names)), self.D_names)))


    # def collinearity(self, chromosome):
    #
    #     idxs = [i for i in range(len(chromosome)) if chromosome[i] == 1 ]
    #     D_names = [self.idx_D_names_dict[idx] for idx in idxs]
    #     cols = [self.D_names_values_dict[name] for name in D_names]
    #     corrs = []
    #     for i in range(len(cols)):
    #         for j in range(len(cols)):
    #             if j > i:
    #                 corr = self.r_k(np.array(cols[i]),np.array(cols[j]))**2
    #                 corrs.append(corr)
    #
    #     test_coll = lambda x: x > self.lower_bound_collinearity
    #     if any([test_coll(cor) for cor in corrs]):
    #         return True
    #     else:
    #         return False


    def initPopulation(self):

        population = []
        for i in range(self.population_size):
            population.append(self.initChromosome())

        return np.array(population)


    def initChromosome(self):

        N_ones = self.N_of_features
        N_zeros = len(self.D_names) - N_ones
        chromosome = np.array([1]*N_ones + [0]*N_zeros)
        np.random.shuffle(chromosome)
        return chromosome


    def fitnessFunction(self, chromosome):

        indexes = np.flatnonzero(chromosome)
        names = [self.idx_D_names_dict[idx] for idx in indexes]
        columns = [self.D_names_values_dict[name] for name in names]

        bias = np.ones(len(columns[0]))
        X = np.vstack((bias,columns)).transpose()
        b = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()), self.P)

        Y_pred = np.dot(X,b.transpose())
        rk = self.r_k(self.P, Y_pred)

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


    def crossover(self, population):

        ch1 = population[self.roulette()]
        ch2 = population[self.roulette()]

        # Using two-point crossover
        if np.random.uniform(0,1) < self.crossover_rate:
            while True:
                idxs = np.random.randint(0,len(ch1),(1,2))
                k = np.amin(idxs)
                l = np.amax(idxs)
                new_ch1 = np.concatenate((ch1[:k],ch2[k:l],ch1[l:]))
                new_ch2 = np.concatenate((ch2[:k],ch1[k:l],ch2[l:]))
                if (np.count_nonzero(new_ch1) == self.N_of_features and np.count_nonzero(new_ch2) == self.N_of_features):
                    #and (not self.collinearity(new_ch1) and not self.collinearity(new_ch2)):
                    return new_ch1, new_ch2
        else:
            return ch1, ch2


    def mutate(self, chromosome):

        flip = lambda x: 1-x
        for idx in range(len(chromosome)):
            if np.random.uniform(0,1) < self.mutation_rate:
                while True:
                    idx1 = np.random.randint(len(chromosome))
                    if chromosome[idx1] == 1:
                        chromosome[idx1] = flip(chromosome[idx1])
                        idx2 = np.random.choice(np.where(chromosome==0)[0])
                        chromosome[idx2] = flip(chromosome[idx2])
                        #if not self.collinearity(chromosome):
                        break
                    elif chromosome[idx1] == 0:
                        chromosome[idx1] = flip(chromosome[idx1])
                        idx2 = np.random.choice(np.where(chromosome==1)[0])
                        chromosome[idx2] = flip(chromosome[idx2])
                        #if not self.collinearity(chromosome):
                        break


    def breed(self, population):

        ch1, ch2 = self.crossover(population)
        self.mutate(ch1)
        self.mutate(ch2)
        return ch1, ch2


    def pick_elites(self):

        fitness_chromosome_dict = dict(list(zip(self.fitnesses,self.current_population)))
        # elite_fitnesses = sorted(self.fitnesses)[-self.N_of_models:]
        #
        # if self.elite_population == []:
        #     for fitness in elite_fitnesses:
        #         self.elite_population.append(fitness_chromosome_dict[fitness])
        # else:
        #     self.elite_population = [fitness_chromosome_dict[fitness] for fitness in elite_fitnesses]

        # return elite population with unique chromosomes.
        elite_fitnesses_set = set(self.fitnesses)
        elite_fitnesses_list = sorted(list(elite_fitnesses_set))[-self.N_of_models:]
        if self.elite_population == []:
            for fitness in elite_fitnesses_list:
                self.elite_population.append(fitness_chromosome_dict[fitness])
        else:
            self.elite_population = [fitness_chromosome_dict[fitness] for fitness in elite_fitnesses_list]

    def Evolve(self):

        N_iterations = 0
        next_population = deepcopy(self.elite_population)

        while N_iterations < self.N_of_generations:
            print("N_iterations - ", N_iterations)
            while len(next_population) < self.population_size:
                next_population.extend(self.breed(self.current_population))

            else:
                self.current_population = deepcopy(next_population)
                self.fitnesses = [self.fitnessFunction(chromosome) for chromosome in self.current_population]
                self.sum_of_fitnesses = sum(self.fitnesses)

                self.pick_elites()
                next_population = deepcopy(self.elite_population)

                N_iterations += 1
                elite_fitnesses = [self.fitnessFunction(chrom) for chrom in self.elite_population]
                self.best_models.add(max(elite_fitnesses))
        else:
            for elite in self.elite_population:
                print(self.fitnessFunction(elite))
            print(self.D_names)
            print(self.D_names_values_dict['D0316001000     '])
            idxs = [i for i in range(len(self.elite_population[-1])) if self.elite_population[-1][i] == 1]
            for idx in idxs:
                print(self.idx_D_names_dict[idx])

