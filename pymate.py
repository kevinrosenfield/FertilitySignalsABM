#!/usr/bin/env python
# coding: utf-8

# ## Questions:
# 
# ### thinking about optimal model parameters vs. making the model realistic:
# 
# Should generations overlap?
# Should parent pairs produce 2 offspring or one?
# 
# We are doing **single-point** crossover.
# We could also do **double-point** (two  segmentations) or **uniform** (each gene is independent) crossover.
# 
# We are using **probibalistic tournament selection** with 3 random competitors vying to be parents (based on fitness)
# We could also use **deterministic5 tournament selection**
# We could also use **fitness proportionate selection**, which is tournament selection among a whole group
# We could choose fathers (or mothers) using one of these methods, and then choose from among that agent's mates

# ## Socioecological variables to review
# 
# Degree of synchrony
# Degree of skew

# Coding genes as parameters of a (e.g. gamma) dist

# # Import packages
# 
# 

# In[40]:


import random as rd
import os
os.environ['QT_API'] = 'pyside6'
import math
import statistics
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from numpy.random import choice
from numpy.random import uniform
from numpy.random import normal
from numpy.random import randint
from numpy.random import permutation
from numpy import corrcoef
from numpy import flip
from numpy import around
from numpy import array as nparray
from numpy import arange as arange
from random import choices as rdchoices
from random import uniform as rduniform
import time
import scipy.stats
from scipy.stats import multivariate_normal as mvn
import statistics
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
from collections import Counter
from scipy.stats import skewnorm
#import pandas as pd
from itertools import chain
import cProfile, pstats


# ## Male object
# 
# Calling the class "male" instantiates a "male" object with the following class variables representing its physical and biological traits:
# 
# **"rank"** represents the agent's position in mating competition; the **makeMatingPairs** method of class "group" matches males of lower numerical **"rank"** (higher dominance) with females of higher **"swelling"**  for mating.
# 
# **"fitness"** is correlated with **"rank"** with a correlation coefficient of approximately **rankFitnessCorrelation ** and influences the chances that mating will result in surviving offspring when it is multiplied by **"conceptionRisk"** to determine **"reproductiveSuccess"**.
# 
# **"reproductiveSuccess"** results from mating with fertile females and through the **setupNextGen** method of class "group" and determines which males wil produce offspring to populate the next generation.
# 
# **"mateTiming"** indicates the **"cycleDay"** on which mating occurred (for visualiztion purposes only)
# 
# **"startingSwelling"** indicates how the fertility swelling **"genes"** will be distributed at the start of the model run. Because males do not swelling, this does not affect their behavior during th model, but can influence that of their female offspring.
# 
# **"genes"** determine swelling strength and are passed on to offspring. They only influence female behavior.
# 
# **"cost"** of swellinging only influences female reproduction.
# 
# Key: \
# "Quotation marks" indicates a class \
# **bold** indicates global variables \
# ***bold, italicized*** indicates methods \
# **"bold, in quotation marks"** indicates class variables

# In[107]:


class Male:
    
    __slots__ = 'rank', 'fitness', 'reproductiveSuccess',  'genes', 'cost', 'mateTiming', '__dict__'
    
    def __init__(self, m, fitness, global_vars):
        self.__dict__.update(global_vars)
        self.rank = m
        self.fitness = fitness
        self.reproductiveSuccess = 1e-10
        self.mateTiming = []
        if self.startingSwelling == "noSwelling":
            self.genes = [randint(1,self.cycleLength+1)] + [0] * 4
        elif self.startingSwelling == "randomUniform":
            self.genes = [randint(1,self.cycleLength+1)]
            self.genes += [randint(0,self.cycleLength+1 - sum(self.genes))]
            self.genes += [randint(0,self.cycleLength+1 - sum(self.genes))]
            self.genes += [randint(0,self.cycleLength+1 - sum(self.genes))] + [uniform(0,1)]
        self.cost = 0
        


# ## Female object
# 
# Calling the class "female" instantiates a "female" object with the following class variables representing its behavioral and biological traits:
# 
# **"ID"** helps locate the agent's index in lists
# 
# **"cycleDay"** determines the day of the cycle on which females start the model. High cycle **synchrony** constrains  it to fewer days
# 
# **"reproductiveSuccess"** results from mating with fit males and through the **setupNextGen** method of class "group" and determines which females will produce offspring to populate the next generation.
# 
# **"startingSwelling"** indicates how the fertility swelling **"genes"** will be distributed at the start of the model run.
# 
# **"genes"** determine swelling strength and are passed on to offspring
# 
# **"cost"** of swellinging is calculated by adding absolute daily swelling strengths and daily increases in swelling strength
# 
# Key: \
# "Quotation marks" indicates a class \
# **bold** indicates global variables \
# ***bold, italicized*** indicates methods \
# **"bold, in quotation marks"** indicates class variables

# In[ ]:


class Female:

    __slots__ ='ID','cycleDay','mateList','reproductiveSuccess','genes','cost','swellingList','swelling','conceptionProbability', '__dict__'
    
    def __init__(self, f, cycleDay, global_vars):
        self.__dict__.update(global_vars)
        self.ID = f
        self.cycleDay = cycleDay
        self.mateList = []
        self.reproductiveSuccess = 1e-10
        if self.startingSwelling == "noSwelling":
            self.genes = [randint(1,self.cycleLength+1)] + [0] * 4
        elif self.startingSwelling == "randomUniform":
            self.genes = [randint(1,self.cycleLength+1)]
            self.genes += [randint(0,self.cycleLength+1 - sum(self.genes))]
            self.genes += [randint(0,self.cycleLength+1 - sum(self.genes))]
            self.genes += [randint(0,self.cycleLength+1 - sum(self.genes))] + [uniform(0,1)]
           
        self.swellingList = self.setSwelling()
        self.swelling = self.swellingList[self.cycleDay - 1]
        self.conceptionProbability = self.conceptionProbabilityList[self.cycleDay - 1]

    def setupCycleDay(self):  
        self.cycleDay = self.cycleDay + 1 if self.cycleDay < self.cycleLength else 1  
        self.swelling = self.swellingList[self.cycleDay - 1]
        self.conceptionProbability = self.conceptionProbabilityList[self.cycleDay - 1]
        
    def setSwelling(self):
        inceraseStartDay, increaseDuration, peakDuration, decreaseDuration, peakSwelling = self.genes
        peakStartDay = inceraseStartDay + increaseDuration
        decreaseStartDay = peakStartDay + peakDuration
        decreaseEndDay = decreaseStartDay + decreaseDuration
        increaseCoefficient =  peakSwelling / (increaseDuration + 1)
        decreaseCoefficient =  peakSwelling / (decreaseDuration + 1)

        x =  np.arange(1,self.cycleLength + 1,1)
        y = [0] * (inceraseStartDay - 1) + [0 + increaseCoefficient * i for i in range(1,increaseDuration)]
        y = y + [peakSwelling] * peakDuration + [peakSwelling - decreaseCoefficient * i for i in range(1,decreaseDuration+1)]
        y = y + [0] * (self.cycleLength - len(y))

        self.cost = sum(y)
        for g in range(1, len(y)): # to add cost of growth
            self.cost = self.cost + (y[g] - y[g - 1]) * 2 if y[g] > y[g-1] else self.cost
        
        self.cost = self.cost + (y[0] - y[-1]) * 2 if y[0] > y[-1] else self.cost
        
        self.cost /= 10
        
        return(y)


# # Group object
# 
# The class "group" generates and simulates the behavior of a single population over the course of a mating season. At initialization, the **"run"** boolean variable is set to "True" and **"day"**, which keeps count of timedays (days) of the simulation, is set to 0. The ***setFitness*** method then sets up a list of male fitness values (**"fitnessList"**) that is correlated to male model.ranks with a correlation coefficient of approximately **rankFitnessCorrelation **. Finally, **nFemales** objects of class "female" and **nMales** objects of class "male" are instantiated in lists (**"males"** and **"females"**) contained in the "group" object.
# 
# The ***runModel*** method simulates agent behavior for **nDays** timedays (days). It first calls the **makeMatingPairs** method, which orders "male" and "female" objects by **"rank"** and **"swelling"**, respctively. Mating pairs are created by pairing males and females with the same index in their respective ordered lists. Males then receive an increase to their **"reproductiveSuccess"** variable in the amount of the current **"conceptionProbability"** of their mate, and females receive an increase to their **"reproductiveSuccess"** variable in the amount of their current **"conceptionProbability"** multipled by the **"fitness"** of their mate.
# 
# For each "female" object, the ***setupCycleDay*** method of class "female" is run to 1) increase **"cycleDay"** by one, and set **"swelling"** and **"conceptionProbability"** based on the unique **"swellingList"** associating that "female's" **"cycleDay"** and **"swelling"** strength variables, and the global **conceptionProbabilityList**, which associates **"cycleDay"** with **"conceptionProbability"**.
# 
# Key: \
# "Quotation marks" indicates a class \
# **bold** indicates global variables \
# ***bold, italicized*** indicates methods \
# **"bold, in quotation marks"** indicates class variables

# In[172]:


class group:
    
    def __init__(self, g, ranks, global_vars):
        
        self.__dict__.update(global_vars)
        
        self.ID = g
        self.day = 0
        self.ranks = ranks
        self.genesInGroup = self.nAgents * self.numberGenes
        self.mutations = [round(uniform(0,self.genesInGroup * self.mutationRate * 2)) for i in range(self.nGenerations)]
        self.potentialMoms = [rdchoices(range(self.nFemales), k = 3) for i in arange(self.nAgents * self.nGenerations)]
        self.potentialDads = [rdchoices(range(self.nMales), k = 3) for i in arange(self.nAgents * self.nGenerations)]
        self.global_vars=global_vars
        
        self.setFitness()
        self.cycleDayList = randint(1, round((self.cycleLength - 1) * (1 - self.synchrony)) + 2, size = self.nFemales)
        
        self.males, self.females = [], []
        
        self.tieBreaker = uniform(0,0.00000000001, self.nFemales * self.nDays)
        
        for m in range(self.nMales):
            self.males.append(Male(m, self.fitnessList[m], global_vars=global_vars))

        for f in range(self.nFemales):
            self.females.append(Female(f, self.cycleDayList[f], global_vars=global_vars))
        
    def runModel(self):

        while self.day < self.nDays:
                
            
            self.females = sorted(self.females, key=self.sortSwelling)
            self.makeMatingPairs()
            self.setupCycleDay()

            self.day += 1
            
            '''
            elif self.day == nDays / 2:
                print(self.day)
            ''' 
            
    def setupCycleDay(self):
        self.cycleDayList += 1
        self.cycleDayList[self.cycleDayList == 31] = 1
        for f in self.females:
            f.cycleDay = self.cycleDayList[f.ID]
            f.swelling = f.swellingList[f.cycleDay - 1]
            f.swelling += self.tieBreaker[self.nFemales * (self.day):self.nFemales * (self.day + 1)][f.ID]
            f.conceptionProbability = self.conceptionProbabilityList[f.cycleDay - 1]
    
    def sortSwelling(self, f):
        return f.swelling

    def makeMatingPairs(self):
        i = 0
        while i < self.nPairs:
            f = self.nFemales - 1 - i
            self.males[i].reproductiveSuccess += self.females[f].conceptionProbability * self.males[i].fitness
            #self.males[i].mateTiming.append(self.females[f].cycleDay)
            self.females[f].reproductiveSuccess += self.females[f].conceptionProbability * self.males[i].fitness
            i += 1
        
    def setFitness(self):

        fitnessList = self.ranks if self.rankFitnessCorrelation > 0.15 else uniform(0,1,self.nMales)
        i = 0.5
        while abs(0 - self.rankFitnessCorrelation  + corrcoef(self.ranks, fitnessList)[1,0]) > 0.05:
            fitnessList = [fitnessList[f] + rduniform(-i,i) for f in range(len(fitnessList))]
            i += 0.05
            if i >= 30.5:
                i = 0.5
                fitnessList = self.ranks
        
        self.fitnessList = flip((fitnessList - np.min(fitnessList))/np.ptp(fitnessList))
            
    def setupNextGen(self):
        
        self.nextGenMotherGenes = []
        self.motherProbabilities = [f.reproductiveSuccess - f.cost for f in self.females]
        # lack of ability to choose becomes a cost as rankFitnessCorrelation  goes down

        self.nextGenFatherGenes = []
        self.fatherProbabilities = [m.reproductiveSuccess for m in self.males] # does male fitness matter?

        # parentsStartingPoint = self.generation * self.nAgents + self.ID * self.nAgents    
        
        self.moms = []
        for i in np.arange(0, self.nAgents):
            # print(self.potentialMoms[0])
            self.moms.append([rdchoices(self.potentialMoms[0], weights=[self.motherProbabilities[p] for p in self.potentialMoms[0]],k = 1)[0]][0])
            del self.potentialMoms[0]

        self.dads = []
        for i in np.arange(0, self.nAgents):
            self.dads.append([rdchoices(self.potentialDads[0], weights=[self.fatherProbabilities[p] for p in self.potentialDads[0]],k = 1)[0]][0])
            del self.potentialDads[0]
        
            
        # moms = [rdchoices(self.potentialMoms[0],
        #                   weights=[motherProbabilities[p] for p in self.potentialMoms[0]],k = 1)[0] for i in np.arange(0, (self.nAgents * 3), 3)]
        # dads = [rdchoices(self.potentialDads[0],
        #                   weights=[fatherProbabilities[p] for p in self.potentialDads[0]], k = 1)[0] for i in np.arange(0, (self.nAgents * 3), 3)]
            
        self.nextGenMotherGenes = [self.females[m].genes for m in self.moms]
        self.nextGenFatherGenes = [self.males[d].genes for d in self.dads]

        self.recombination()
        self.mutation() if self.mutations[(self.generation - 1)] > 0 else 0
        self.reset()
            
    def recombination(self):
        
        self.offspringGenes = []
        recombinationPoints = choice(range(self.numberGenes), self.nAgents)
        splitTypes = randint(0,2, self.nAgents)
        i = 0
        while i < self.nAgents:
            recombinationPoint = recombinationPoints[i]
            if splitTypes[i] == 1:
                self.offspringGenes.append([m for m in self.nextGenMotherGenes[i][:recombinationPoint]] + 
                                           [f for f in self.nextGenFatherGenes[i][recombinationPoint:]])
            else:
                self.offspringGenes.append([f for f in self.nextGenFatherGenes[i][:recombinationPoint]] + 
                                           [m for m in self.nextGenMotherGenes[i][recombinationPoint:]])
            i += 1
                
    def mutation(self):
        
        mutations = self.mutations[(self.generation - 1)]
       
        dayMutations = randint(mutations)
        peakMutations = mutations - dayMutations
      
        dayGenesMutating = choice(range(4), dayMutations)
        peakGenesMutating = [4] * (peakMutations)
        
        dayOffspringsMutating = choice(range(len(self.offspringGenes)), dayMutations, replace=False)
        peakOffspringsMutating = choice(range(len(self.offspringGenes)), peakMutations, replace=False)

        dayPertubations = choice([-1,1], dayMutations)
        peakPertubations = uniform(-0.02,0.02, peakMutations)
        
        newDayGenes = nparray([self.offspringGenes[dayOffspringsMutating[m]][dayGenesMutating[m]] + dayPertubations[m] for m in range(dayMutations)])
        newPeakGenes = nparray([self.offspringGenes[peakOffspringsMutating[m]][peakGenesMutating[m]] + peakPertubations[m] for m in range(peakMutations)]) 
                
        newDayGenes[newDayGenes > 30] = 30
        newDayGenes[newDayGenes < 0] = 0
        for i in range(dayMutations):
            newDayGenes[i] = 1 if dayGenesMutating[i] == 0 and newDayGenes[i] <= 0 else newDayGenes[i]
        newPeakGenes[newPeakGenes < 0] = 0
        newPeakGenes[newPeakGenes > 1] = 1

        for m in range(dayMutations):
            self.offspringGenes[dayOffspringsMutating[m]][dayGenesMutating[m]] = newDayGenes[m]
            self.offspringGenes[m][self.offspringGenes[m].index(max(self.offspringGenes[m]))] = max(
                self.offspringGenes[m]) - (sum(
                self.offspringGenes[m][:4]) - self.cycleLength) if self.offspringGenes[m][sum(
                self.offspringGenes[m][:4]) > self.cycleLength] == 2 else max(self.offspringGenes[m])
            
        for m in range(peakMutations):
            self.offspringGenes[peakOffspringsMutating[m]][peakGenesMutating[m]] = newPeakGenes[m]
     

    def setGenotypes(self):
        
        for f in self.females:
            f.genes = self.offspringGenes[f.ID]
            f.swellingList = f.setSwelling()
            #f.cost = sum(self.offspringGenes[f.ID])
            #f.swellingList = f.genes
            f.setupCycleDay()

        for m in self.males:
            m.genes = self.offspringGenes[m.rank + self.nFemales]
            
    def reset(self):
        self.day = 0
        self.setFitness()
        self.cycleDayList = randint(1, round((self.cycleLength - 1) * (1 - self.synchrony)) + 2, size = self.nFemales)
        self.males, self.females = [], []
        self.males = [Male(m, self.fitnessList[m], global_vars=self.global_vars) for m in range(self.nMales)]
        self.females = [Female(f, self.cycleDayList[f], global_vars=self.global_vars) for f in range(self.nFemales)]
        
        


# # Evolving Model object
# 
# The class "evolvingModel" contains class variables and methods to initialize a simulated world with multiple groups and simulate biological evolution using a genetic algorithm. Initialization sets the model to generation 0 and instantiates **nGroups** social groups of class "group."
# 
# The ***evolve*** method of class "evolvingModel" loops through **nGenerations** generations. ***Evolve*** method process: First, the generation number is increased by 1. Then, the behavior of each group is simulated for a single generation using the ***runModel*** method of class "group." The ***setupNextGen*** and ***setGenotypes*** methods of class "group" then decides on a cohort of **nAgents** mothers and **nAgents** fathers for the next generation, perform  genetic **recombination** using mothers' and fathers' genes, performs probabilistic **mutation** of offspring genes, and finally initializes a new generation of **nFemales** and **nMales** with the those genes. Finally, the ***migration*** method of class "evolvingModel" probabilistically selects some number of the next gerenations' agents to migrate to new groups, based on parameter **migrationRate**, with a probability of **maleDispersalBias** that the agent selected for migration with be male. Migration switches the following class variables of the agent (a member of either class "female" or "male") : **"genes", "swellingList"** (females only), and **"cost"**.
# 
# If **realTimePlots** is set to True, a dot plot of average swelling strengh (Y-axis) across group 0 against cycle day (X-axis) appears in a quartz window following the first generation and updates with probability **realTimePlotsRate** for future generations. Following a model run of **nGenerations**, a line plot showing changes across generations in the intra-generation variability of cycle days on which the alpha male of group 0 mates, with lines for first quartile, median, and 3rd quartile of mating days. The Y-axis is mating days and the X-axis is generations. This concludes the model run.
# 
# Key: \
# "Quotation marks" indicates a class \
# **bold** indicates global variables \
# ***bold, italicized*** indicates methods \
# **"bold, in quotation marks"** indicates class variables
# 

# In[138]:


class evolvingModel:
    
    def __init__(self, dispersal = True, nMales = 10, nFemales = 10, nGroups = 2, cycleLength = 30, rankFitnessCorrelation  = 0.0,
                synchrony = 0.0, mutationRate = 0.01, migrationRate = 0.01, maleDispersalBias = 0.5, realTimePlots = True,
                whichPlot = "swelling", #whichPlot = "Pairs", swellingFunction = "eachDay", #startingSwelling = "randomUniform"
                realTimePlotsRate = 0.25, nDays = 60, nGenerations = 1000, swellingFunction = "slopes",
                startingSwelling = "noSwelling",
                # cycle parameters
                ovulation = 16):

        prePOPLength = ovulation - 6
        postPOPLength = cycleLength - prePOPLength - 6
        conceptionProbabilityList = [0]*prePOPLength+[.05784435,.16082819,.19820558,.25408223,.24362408,.10373275]+[0]*postPOPLength

        self.generation = 0
        self.ranks = range(nMales)
        self.nPairs = min(nMales, nFemales)
        self.nAgents = nMales + nFemales
        self.numberGenes = cycleLength if swellingFunction == "eachDay" else 5
        self.genesInGroup = self.numberGenes * self.nAgents  
        
        # basic model parameters
        global_vars = {"nMales": nMales,
        "nFemales": nFemales,
        "nAgents": self.nAgents,
        "nPairs": self.nPairs,
        "numberGenes": self.numberGenes,
        "nGroups": nGroups,
        "dispersal": False if nGroups < 2 else dispersal,
        "cycleLength": cycleLength,
        "rankFitnessCorrelation": rankFitnessCorrelation,
        "synchrony": synchrony, # seasonality vs. group-size influences
        "mutationRate": mutationRate,
        "migrationRate": migrationRate,
        "maleDispersalBias": maleDispersalBias,
        "realTimePlots": realTimePlots,
        #"whichPlot": "Pairs",
        "whichPlot": whichPlot,
        "realTimePlotsRate": realTimePlotsRate,
        "nDays": nDays,
        "nGenerations": nGenerations,
        #"swellingFunction": "eachDay",
        "swellingFunction": swellingFunction,
        #"startingSwelling": "randomUniform",
        "startingSwelling": startingSwelling,
        "genesInGroup": self.genesInGroup,
        "generation": 0,

        # cycle parameters
        "ovulation": ovulation,
        "prePOPLength": prePOPLength,
        "postPOPLength": postPOPLength,
        "conceptionProbabilityList": conceptionProbabilityList}

        self.__dict__.update(global_vars)

        self.groups = []

        self.alphaMates, self.alphaMatingSpread = [], [[],[],[]]

        self.totalAgents = self.nAgents * self.nGroups
        self.ranks = range(self.nMales)
        self.nPairs = min(self.nMales, self.nFemales)
        self.numberGenes = self.cycleLength if self.swellingFunction == "eachDay" else 5
        
        self.potentialMoms = rdchoices(range(self.nFemales), k = 3 * self.nAgents * self.nGroups * self.nGenerations)
        self.potentialDads = rdchoices(range(self.nMales), k = 3 * self.nAgents * self.nGroups * self.nGenerations)
        
        for g in range(self.nGroups):
            self.groups.append(group(g, self.ranks, global_vars=global_vars))

        
        
    def evolve(self):
        
        # if realTimePlots == True:
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        
        for self.generation in range(1, self.nGenerations):
            self.generation += 1
            
            g = 0
            while g < self.nGroups:
                self.groups[g].runModel() 
                g += 1
        
            # if self.realTimePlots == True and (rd.uniform(0,1) < self.realTimePlotsRate or self.generation == 1):
            #     self.plotSwelling() if self.whichPlot == "swelling" else self.plotPairs()
            # elif rd.uniform(0,1) > 0.99:
            #     print(self.generation)
            
            #self.updateAlphaMatingDays()
            
            # if self.generation == nGenerations - 1:
                # get_ipython().run_line_magic('matplotlib', 'inline')
                # self.plotRS()
            
            for g in self.groups:
                g.setupNextGen()
                g.setGenotypes()
                g.generation += 1
               
            if self.dispersal == True:
                self.migration()
        
        lst = []
        lstLower = []
        lstUpper =[]
        for j in range(self.cycleLength):
            lst.append(statistics.mean([f.swellingList[j] for f in sum([g.females for g in self.groups], [])]))
            SEM = scipy.stats.tstd([f.swellingList[j] for f in sum([g.females for g in self.groups], [])])
            lstLower.append(lst[j] - SEM) if SEM < lst[j] else lstLower.append(0)
            lstUpper.append(lst[j] + SEM)

        
        # if realTimePlots == False:
        #     self.plotSwelling()
        #     plt.figure()
        #     self.plotPairs()
            
        #self.plotMatingDays()
            
    def migration(self):
                
        migrations = round(uniform(self.totalAgents * self.migrationRate * 2))
        groupsLeavingFrom = choice([g for g in self.groups], migrations)
        agentsLeaving = choice(range(self.nMales), migrations)
        
        for m in range(migrations):
            
            groupLeavingFrom = groupsLeavingFrom[m]
            
            if rd.uniform(0,1) > self.maleDispersalBias:
                agentLeaving = groupLeavingFrom.males[agentsLeaving[m]]
                agentComing = rd.choice(rd.choice([g for g in self.groups if g != groupLeavingFrom]).males)
            else:
                agentLeaving = groupLeavingFrom.females[agentsLeaving[m]]
                agentComing = rd.choice(rd.choice([g for g in self.groups if g != groupLeavingFrom]).females)
            
            tempGenes, tempCost = agentLeaving.genes, agentLeaving.cost
            agentLeaving.genes, agentLeaving.cost = agentComing.genes, agentComing.cost
            agentComing.genes, agentComing.cost = tempGenes, tempCost
        
    def plotSwelling(self):

        lst = []
        lstLower = []
        lstUpper =[]
        for j in range(self.cycleLength):
            lst.append(statistics.mean([f.swellingList[j] for f in sum([g.females for g in self.groups], [])]))
            SEM = scipy.stats.tstd([f.swellingList[j] for f in sum([g.females for g in self.groups], [])])
            lstLower.append(lst[j] - SEM) if SEM < lst[j] else lstLower.append(0)
            lstUpper.append(lst[j] + SEM)

        plt.clf()
        plt.xlabel('Day of the cycle')
        plt.ylabel('Swelling size')
        plt.plot(lst, "bo")
        #[plt.plot(lstLower, "r")
        [plt.plot([i,i],[l,u], "r") for i,l,u in zip(range(self.nFemales),lstLower,lstUpper)]
        #plt.hist([i.genes[0] for i in model.groups[1].females])
        plt.ylim = [0,max(lst) * 1.1]
        #plt.text(0.1, max(lst) * 0.9, str(self.generation))
        plt.pause(0.000001)
        plt.show()
            
    def plotRS(self):
        
        femaleRS = [f.reproductiveSuccess for f in self.groups[0].females]
        maleRS = [m.reproductiveSuccess for m in self.groups[0].males]
        
        print("SD^2 in female RS:"+str(statistics.variance(femaleRS)))
        print("SD^2 in male RS:"+str(statistics.variance(maleRS)))

        plt.figure()
        plt.hist(maleRS, alpha=0.5, label = "male RS")
        plt.hist(femaleRS, alpha=0.5, label = "female RS")
        plt.title('Male and female Reproductive Success')
        plt.legend(loc='upper right')
        plt.show()
        
        plt.figure()
        plt.plot([m.rank for m in self.groups[0].males],
                 [m.reproductiveSuccess for m in self.groups[0].males], 'bo')
        plt.title('male RS ~ rank')
        plt.show()
        
    def plotMatingDays(self):
        
        plt.figure()
        
        for i in range(3):
            plt.plot(range(self.generation), self.alphaMatingSpread[i])
        
        plt.ylim = [0, self.cycleLength]
        plt.xlim = [0, 1.0]
        plt.title("Synchrony: " + str(round(self.synchrony, 2)) + "; Rank/Fitness Correlation: " + str((self.rankFitnessCorrelation , 2)))
        
    def updateAlphaMatingDays(self):
    
        self.alphaMates.extend(self.groups[0].males[0].mateTiming)
        a,b,c = np.percentile(self.alphaMates,[25, 5, 75])

        self.alphaMatingSpread[0].append(a)
        self.alphaMatingSpread[1].append(b)
        self.alphaMatingSpread[2].append(c)
        
    def plotPairs(self):
        plt.clf()
        plotGroup = self.groups[0]
        plotGroup.females = sorted(choice(plotGroup.females, size=self.nFemales, replace=False), key=plotGroup.sortSwelling)
        swellings = [f.swelling * 1000 for f in plotGroup.females]
        IDs = [f.ID for f in plotGroup.females]
        plt.scatter([1] * self.nFemales, [f.ID for f in plotGroup.females], s = swellings)
        plt.scatter([2] * self.nMales, [m.rank for m in plotGroup.males], s = [m.rank for m in plotGroup.males])
        plt.scatter(0,0, s = 0)
        plt.scatter(3,0, s = 0)
        mates = [[plotGroup.females[i].ID] + [plotGroup.males[i].rank] for i in range(self.nPairs)]
        mates = [agent for pair in mates for agent in pair]
        for i in range(self.nPairs):
            plt.plot([1,2],[plotGroup.females[i].ID, plotGroup.males[i].rank],linewidth=0.5)
        plt.xlim = [0,3]
        plt.text(0.1, self.nMales * 0.9, str(self.generation))
        plt.title('mating pairs based on female swelling size (left)\nand male rank (right)')
        plt.pause(0.00001)
        plt.show()



# # In[170]:


# plt.plot(conceptionProbabilityList, 'bo')
# plt.xlabel('Day of the cycle')
# plt.ylabel('Conception probability')


# # In[163]:


# model.groups[0].nextGenMotherGenes = []
# motherProbabilities = [f.reproductiveSuccess - f.cost for f in model.groups[0].females]
# # lack of ability to choose becomes a cost as rankFitnessCorrelation  goes down

# model.groups[0].nextGenFatherGenes = []

# fatherProbabilities = [m.reproductiveSuccess for m in model.groups[0].males] # does male fitness matter?
            
# parentsStartingPoint = model.generation * nGroups * model.nAgents + model.groups[0].ID * model.nAgents    

# fatherProbabilities

# fatherArray = np.arange(0, (model.nAgents * 3), 3)

# [[fatherProbabilities[p] for p in model.potentialDads[
#                       parentsStartingPoint + i: parentsStartingPoint + i+2]] for i in fatherArray]

# #dads = [rdchoices(model.potentialDads[parentsStartingPoint + i:parentsStartingPoint + i + 2],
# #                  weights=[fatherProbabilities[p] for p in model.potentialDads[
# #                      parentsStartingPoint + i: parentsStartingPoint + i+2]], k = 1)[0] for i in np.arange(0, (model.nAgents * 3), 3)]

# motherProbabilities


# # In[169]:


# model.potentialMoms[parentsStartingPoint + 3: parentsStartingPoint + 3 + 2]


# # In[160]:


# nDays = 60
# nGenerations = 1
# modelRuns = 0
# nMales = 3
# nFemales = 2
# rankFitnessCorrelation = 0.2
# #synchrony = 0.0
# model = evolvingModel()
# model.evolve()
# modelRuns += 1
# print(modelRuns)


# # In[158]:


# #print([m for m in [g.males for g in model.groups]])

# [male.fitness for males in [g.males for g in model.groups] for male in males]

# #[m.fitness for m in model.groups[1].males]


# # # modelData4 = pd.read_csv('slopesModelData1000.csv')
# # modelData4 = modelData4.iloc[:,1:]

# # In[ ]:


# modelData4[modelData4.columns[modelData4.iloc[0]==1]]


# # In[9]:


# a = [plt.plot(modelData4.iloc[2,i], max(modelData4.iloc[3:33,i]), 'yo') for i in range(48)]
# plt.figure()
# b = [plt.plot(modelData4.iloc[0,i], max(modelData4.iloc[3:33,i]), 'bo') for i in range(48)]
# plt.figure()
# c = [plt.plot(modelData4.iloc[1,i], max(modelData4.iloc[3:33,i]), 'ro') for i in range(48)]
# plt.figure()
# d = [plt.plot(modelData4.iloc[0,i] + modelData4.iloc[1,i], sum(modelData4.iloc[3:33,i]), 'go')
#  for i in range(len(modelData4.T))]


# # In[ ]:


# e= [[sns.lineplot(range(30), modelData4.iloc[3:33,i]) for i in range(48)]]


# # ## Everything below this point is for testing code

# # In[171]:


# nMales = 25
# nFemales = 25
# nAgents = nFemales + nMales
# model.nPairs = min(nMales, nFemales)
# model.ranks = range(nMales)
# nDays = 100
# rankFitnessCorrelation = 0.66
# nGenerations = 1
# model = evolvingModel()
# profiler = cProfile.Profile()
# profiler.enable()
# #model.groups[0].runModel()
# model.evolve()
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.print_stats()


# # In[ ]:





# # In[ ]:


# nDays = 60
# nMales = 10
# nFemales = 10
# nAgents = nFemales + nMales
# model.nPairs = min(nMales, nFemales)
# model.ranks = range(nMales)
# rankFitnessCorrelation = 1.0
# cycleLength = 30
# synchrony = 0.0
# model = evolvingModel()

# model.groups = [group(0)]

# profiler = cProfile.Profile()
# profiler.enable()

# cycles = math.ceil(nDays / cycleLength)

# IDs = [d for d in range(nFemales)] * nDays
# cycleDays = [f.cycleDay-1
#              for f
#              in model.groups[0].females]

# swellings = []
# conceptionRisks = []

# swellDictList = []

# for i in range(nDays):
#   swellings += [(f.swellingList  * nDays)[cycleDays[f.ID] + i] 
#                 for f 
#                 in model.groups[0].females]
#   conceptionRisks += [(conceptionProbabilityList * nDays)[cycleDays[f.ID] + i]
#                       for f
#                       in model.groups[0].females]

# fitnesses = model.groups[0].fitnessList


# for day in range(nDays):
#  swellDictList += [[[ID, swellings[day*nFemales + ID]
#                      + model.groups[0].tieBreaker[day*nFemales + ID],conceptionRisks[day*nFemales + ID]]
#                     for ID
#                     in IDs[day*nFemales:day*nFemales+nFemales]]]


# swellDictList = [sorted(swellDictList[i], key=lambda swelling: swelling[1])
#                  for i
#                  in range(nDays)]

# swellDictList = [swellDictList[day][ID] + [ID] + [fitnesses[ID]] + [fitnesses[ID] * swellDictList[day][ID][2]]
#                  for ID
#                  in range(nFemales)
#                  for day
#                  in range(nDays)]

# newSwellDictList = sorted(swellDictList, key=lambda swelling: swelling[0])

# for lst in newSwellDictList:
#     model.groups[0].females[lst[0]].reproductiveSuccess += lst[5]
    
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.print_stats()


# # In[ ]:


# model.


# # In[ ]:


# model = evolvingModel()

# profiler = cProfile.Profile()
# profiler.enable()

# model.groups[0].runModel()

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.print_stats()


# # In[ ]:




