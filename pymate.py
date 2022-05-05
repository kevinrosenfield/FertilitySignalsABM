import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class male:

    def __init__(self, m, g, genes=[0.0,0.5]):

        self.id = m
        self.group_id = g
        self.total_quality = genes[0]
        self.contest = genes[0] * genes[1]
        self.scramble = (genes[0] - self.contest) + 1
        self.genes = genes
        self.rank = "N/A"
        self.energy = 1 - self.contest

class female:

    def __init__(self,
                 f,
                 days_until_cycling,
                 conception_probability_list,
                 mean_days_to_conception,
                 sd_days_to_conception,
                 g,
                 genes=[0.0,0.5]):

        self.id = f
        self.group_id = g
        #self.days_until_cycling = days_until_cycling
        self.conception_probability_list = conception_probability_list
        self.conception_probability = 0
        self.days_gestation_plus_lactation = days_until_cycling
        self.status = "gestating or lactating"
        self.cycle_day = "N/A"
        self.genes = genes
        self.days_until_conception = "N/A"
        self.pseudo_cycling_days = "N/A"
        self.menses_onset_days = []

        self.conception_probability_master_list = [[],[]]

        self.days_until_conception_future = abs(
            round(
                np.random.normal(mean_days_to_conception,
                                 sd_days_to_conception)))

        self.days_gestation_plus_lactation_future = days_gestation_plus_lactation
        
    def switch_to_cycling(self, gestating_or_lactating, females_cycling, day):

        self.menses_onset_days.append(day)
        self.days_until_conception = self.days_until_conception_future
        
        females_cycling.append(self)
        gestating_or_lactating.remove(self)
        self.status = "cycling"
        self.days_gestation_plus_lactation = "N/A"

        self.cycle_day = 0
        self.conception_probability = self.conception_probability_list[
            0]  # starts day 0

    def switch_to_gestating_or_lactating(self, females_cycling, gestating_or_lactating, end_of_season):

        gestating_or_lactating.append(self)
        females_cycling.remove(self)
        self.status = "gestating or lactating"
        self.cycle_day = "N/A"
        self.days_gestation_plus_lactation = 365 if end_of_season == True else random.randint(0,round(365 - (365 * seasonality)))
        self.conception_probability = 0


class group:

    # when a 'group' object is initialized, 'male' and 'female' objects are automatically instantiated in lists
    # ('males' and 'females') contained in the 'group' object
    # the male dominance hierarchy is set when the 'setRanks' method runs during 'group' initialization

    def __init__(self, g, conception_probability_list,
                 mean_days_to_conception, sd_days_to_conception):

        self.array_of_latencies_to_cycling = np.array([random.randint(0,round(365 - (365 * seasonality))) for i in range(number_females)])
        self.array_of_latencies_to_cycling -= (min(self.array_of_latencies_to_cycling) + 1)
        
        self.id = g
        self.females_gestating_or_lactating = [
            female(f,
                   self.array_of_latencies_to_cycling[f],
                   conception_probability_list,
                   mean_days_to_conception,
                   sd_days_to_conception,
                   g=self.id,genes=[random.uniform(0,1),0.5]) for f in range(number_females)
        ]

        self.females_cycling = []

        self.all_females = sorted(np.concatenate([
            np.array(self.females_cycling),
            np.array(self.females_gestating_or_lactating)]),key=self.sort_by_id)

        self.males = [male(m, g=self.id,genes=[random.uniform(0,1),0.5]) for m in range(number_males)]

        self.mothers = []
        self.fathers = []

        self.mating_matrix = np.array(
            [np.array([1e-40] * number_males) for f in range(number_females)])

        self.conception_probability_master_list = []
        self.conception_probability_master_plot_list = []
        self.model_day_per_female_list = [item for sublist in [[i] * number_females for i in range(4000)] for item in sublist]

        self.set_ranks()
        self.day = 0
        self.year = 0
        
    def set_ranks(self):

        self.rank_entries = [m.contest + 1e-50 for m in self.males]

        self.rank_entries_scaled = [
            e / sum(self.rank_entries) for e in self.rank_entries
        ]

        self.ranks = np.random.choice(range(number_males),
                                      p=self.rank_entries_scaled,
                                      size=number_males,
                                      replace=False)

        for i, m in enumerate(self.males):
            m.id = np.where(self.ranks == i)[0][0]
            m.rank = m.id

        self.males = sorted(self.males, key=self.sort_by_id)
        
    def start_cycling(self, day):

        switch_to_cycling_list = []
        
        for f in self.females_gestating_or_lactating:
            f.days_gestation_plus_lactation -= 1
            if f.days_gestation_plus_lactation < 0:
                switch_to_cycling_list.append(f)

        [f.switch_to_cycling(self.females_gestating_or_lactating, self.females_cycling, day) for f in switch_to_cycling_list]


    def continue_cycling(self, day):

        switch_to_gestating_or_lactating_list = []

        if self.day >= round((365 * (1-seasonality)) + ((1 - seasonality) * cycle_length)):
            for f in self.females_cycling:
                switch_to_gestating_or_lactating_list.append(f)
                f.pseudo_cycling_days = -1
                f.conception_probability = 0.0

            [
                f.switch_to_gestating_or_lactating(self.females_cycling,
                                                   self.females_gestating_or_lactating, end_of_season = True)
                for f in switch_to_gestating_or_lactating_list
            ]
                
        else:

            if fixed_number_of_cycles == True:
                for f in self.females_cycling:
                    if f.status == "pseudo_cycling":
                        switch_to_gestating_or_lactating_list.append(f) if f.pseudo_cycling_days <= 0 else 0
                        f.pseudo_cycling_days -= 1
                        f.conception_probability = 0.0
                    else:
                        f.days_until_conception -= 1
                        f.cycle_day = f.cycle_day + 1 if f.cycle_day < cycle_length - 1 else 0
                        f.conception_probability = f.conception_probability_list[
                            f.cycle_day]
                        if f.days_until_conception < 0:
                            f.status = "pseudo_cycling"
                            f.pseudo_cycling_days = cycle_length - f.cycle_day

            else:
                for f in self.females_cycling:
                    if f.status == "pseudo_cycling":
                        switch_to_gestating_or_lactating_list.append(f) if f.pseudo_cycling_days <= 0 else 0
                        f.pseudo_cycling_days -= 1
                        f.conception_probability = 0.0
                    elif f.cycle_day < cycle_length - 1:
                        f.cycle_day = f.cycle_day + 1
                        f.conception_probability = f.conception_probability_list[f.cycle_day]
                    else:
                        f.cycle_day = 0
                        f.menses_onset_days.append(day)
                        f.conception_probability = f.conception_probability_list[f.cycle_day]
                
            [
                f.switch_to_gestating_or_lactating(self.females_cycling,
                                             self.females_gestating_or_lactating, end_of_season = False)
                for f in switch_to_gestating_or_lactating_list
            ]
        
    def make_mating_pairs(self):

        males_mating = self.males #[m for m in self.males if random.uniform(0,0.0000001) < m.energy]
        number_pairs = min(len(males_mating), len(self.females_cycling))

        if fixed_number_of_cycles == True:
            for f, m in zip(np.random.permutation( #randomize cycling females
                    self.females_cycling)[:number_pairs], males_mating[:number_pairs]):
                if f.status == "cycling":
                    self.mating_matrix[
                        f.id][m.id] += f.conception_probability * m.scramble
        
        else:
            for f, m in zip(np.random.permutation( #randomize cycling females
                    self.females_cycling)[:number_pairs], males_mating[:number_pairs]):
                if np.random.uniform(0,1) < (f.conception_probability) * m.scramble and f.status == "cycling":
                    f.pseudo_cycling_days = cycle_length - f.cycle_day
                    f.status = "pseudo_cycling"

                    self.fathers.append(m)
                    self.mothers.append(f)

    def go_one_day(self):
        
        for f in self.all_females:
            f.conception_probability_master_list[0].append(self.day + (self.year * 365))
            f.conception_probability_master_list[1].append(f.conception_probability + f.id)
            
        self.continue_cycling(self.day)
        
        self.start_cycling(self.day)

        self.make_mating_pairs() if any(
            [f.conception_probability for f in self.females_cycling]
        ) else 0  # only run function to make mating pairs if conception is possible

        if self.day == 365:
            self.day = 0
            self.year += 1
            for f in self.females_gestating_or_lactating:
                f.days_gestation_plus_lactation = random.randint(0,round(365 - (365 * seasonality)))
        else:
            self.day += 1
        
    def go_one_mating_season(self):

        self.set_ranks()
            
        if fixed_number_of_cycles == True:
            while len(self.females_cycling) < number_females:
                self.go_one_day()
            while len(self.females_gestating_or_lactating) < number_females:
                self.go_one_day()
        else:
            while len(self.mothers) < number_females * 2:
                self.go_one_day()

        [
            f.switch_to_gestating_or_lactating(self.females_cycling,
                                         self.females_gestating_or_lactating, end_of_season = False)
            for f in self.females_cycling
        ]

        #plt.hist([f.id for f in self.mothers])
        #plt.show()
        
        l = len(self.conception_probability_master_plot_list)
        l2 = [i * number_females for i in range(int(l / number_females))]
        self.daily_conception_probability_sums = [sum(_) - sum(range(number_females)) for _ in zip(*[
            f.conception_probability_master_list[1] for f in self.all_females])]
        
        self.daily_conception_probability_counts = [len([i for i in _ if i % 1 != 0]) for _ in zip(*[f.conception_probability_master_list[1] for f in self.all_females])]
            
    def determine_next_gen_parents(self):

        self.females_gestating_or_lactating = sorted(self.females_gestating_or_lactating,
                                               key=self.sort_by_id)

        total_conception_probabilities = []
        self.parents = []

        for f in range(number_females):
            self.mating_matrix[f] = [
                fms / sum(self.mating_matrix[f])
                for fms in self.mating_matrix[f]
            ]

        for _ in [0, 1]:
            for mother in self.females_gestating_or_lactating:

                #potential_fathers = random.choices(self.males, weights=[1 - m.cost for m in self.males] k=3)
                potential_fathers = random.choices(self.males, k=3)

                father = random.choices(potential_fathers,
                                        weights=self.mating_matrix[mother.id][[
                                            p.id for p in potential_fathers
                                        ]],
                                        k=1)[0]

                self.parents.append([mother, father])

    def generate_offspring(self, max_non_cycling_days,
                           conception_probability_list,
                           mean_days_to_conception, sd_days_to_conception):

        self.females_gestating_or_lactating = []
        self.males = []

        if fixed_number_of_cycles == True:
            self.parents = np.random.permutation(
                self.parents)  # randomize order to avoid biasing offspring sex
        else:
            self.parents = [_ for _ in zip(self.mothers[0:number_agents],self.fathers[0:number_agents])]
        
        for i, p in enumerate(
                self.parents[:number_females]
        ):  # loop through parents until reaching number females
            new_genes = [random.choice([p[0].genes[0], p[1].genes[0]]), random.choice([p[0].genes[1], p[1].genes[1]])]
            self.females_gestating_or_lactating.append(
                female(i,
                       max_non_cycling_days,
                       conception_probability_list,
                       mean_days_to_conception,
                       sd_days_to_conception,
                       g=self.id,
                       genes=new_genes))

        for i, p in enumerate(
                self.parents[number_males:]
        ):  # loop through remaining parents until reaching number males
            new_genes = [random.choice([p[0].genes[0], p[1].genes[0]]), random.choice([p[0].genes[1], p[1].genes[1]])]
            self.males.append(male(i, g=self.id, genes=new_genes))
            
        self.all_females = sorted(np.concatenate([
            np.array(self.females_cycling),
            np.array(self.females_gestating_or_lactating)]),key=self.sort_by_id)

    def mutate(self):

        mutation_lottery = np.random.uniform(0, 1,
                                             (number_males + number_females) * 2)

        number_mutations = sum(
            [1 for m in mutation_lottery if m < mutation_rate])

        for m in range(number_mutations):
            agent_mutating = np.random.choice(self.males +
                                           self.females_gestating_or_lactating)
            gene_mutating = np.random.choice([0,1])
            agent_mutating.genes[gene_mutating] += np.random.uniform(-0.01, 0.01)
            if agent_mutating.genes[gene_mutating] < 0:# or agent_mutating.gene > 20:
                agent_mutating.genes[gene_mutating] = 0
            elif agent_mutating.genes[gene_mutating] > 1:# or agent_mutating.gene > 20:
                agent_mutating.genes[gene_mutating] = 1

    def recombination(self):

        pass

    def make_agent_data_dfs(self):
        all_females = np.concatenate([
            np.array(self.females_cycling),
            np.array(self.females_gestating_or_lactating)
        ])

        self.female_data = pd.DataFrame({
            'id': [f.id for f in all_females],
            'status': [f.status for f in all_females],
            'days until cycling': [f.days_until_cycling for f in all_females],
            'days until conception':
            [f.days_until_conception for f in all_females],
            #'conception probability':
            #[f.conception_probability for f in all_females],
            #'total conception probability':
            #[round(np.sum(i), 2) for i in self.mating_matrix]
        })

        self.male_data = pd.DataFrame({
            'rank': [m.rank for m in self.males],
            'contest': [m.contest for m in self.males],
            'total conception probability':
            [round(np.sum(i), 2) for i in self.mating_matrix.T]
        })

    def make_mating_df(self):
        self.mating_df = pd.DataFrame(self.mating_matrix).round(2).set_axis(
            ['m{}'.format(m) for m in range(number_males)],
            axis=1,
            inplace=False).set_axis(
                ['f{}'.format(f) for f in range(number_females)],
                axis=0,
                inplace=False)

    def plot_fertile_mating_success(self):
        plt.close()
        self.make_mating_df()
        plt.figure(figsize=(14, 5))
        plt.xlabel('Male ID')
        plt.ylabel('Female ID')
        plt.title(f'Seasonality: {seasonality}, Day: {self.day}')
        fig = sns.heatmap(self.mating_df, cmap='RdYlGn_r')
        plt.pause(0.000001)
        plt.show()

    def plot_fertile_mating_success_aggregated(self, size = (14, 5)):
        plt.close()
        means = np.array([round(np.mean(i - 1e-40),4) for i in self.mating_matrix.T])
        plt.rc('axes', labelsize=11.5)
        fig2 = plt.figure(figsize=size)
        myPlot = fig2.add_subplot(111)
        hm = myPlot.imshow(means[np.newaxis, :], cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(hm)
        plt.yticks([])
        plt.xticks([])
        #plt.ylim([min(means),max(means)])
        plt.xlabel('Male ID')
        plt.ylabel('Mean male conception probability\n across females')
        plt.pause(0.000001)
        plt.show()
        return fig2

    def plot_conception_probabilities(self, size = (14, 5)):
        #l = len(self.conception_probability_master_plot_list)
        #l2 = [i * number_females for i in range(int(l / number_females))]
        plt.close()
        fig2 = plt.figure(figsize=size)
        #plt.plot(self.model_day_per_female_list[:l], self.conception_probability_master_plot_list, 'bo')
        #plt.plot([sum(self.conception_probability_master_list[i:i+number_females]) - 1 for i in l2], 'r')
        [plt.plot(f.conception_probability_master_list[0], f.conception_probability_master_list[1]) for f  in self.all_females]
        plt.xlabel("Day of mating season")
        plt.ylabel("Conception probability\n(one line per female)")
        plt.title("Female conception probabilities")

        plt.pause(0.000001)
        plt.show
        return fig2

    def sort_by_id(self, agent):
        return agent.id


class population:

    def __init__(self):

        pre = ovulation - 6
        post = cycle_length - pre - 6

        self.max_non_cycling_days = round(365 - (365 * seasonality)) - cycle_length

        self.conception_probability_list = [0] * pre + [
            .05784435, .16082819, .19820558, .25408223, .24362408, .10373275
        ] + [0] * post

        self.mean_days_to_conception = 50
        self.sd_days_to_conception = 0  # * (1.0 - seasonality)

        self.groups = [
            group(g,
                  self.conception_probability_list,
                  self.mean_days_to_conception, self.sd_days_to_conception)
            for g in range(number_groups)
        ]

    def migrate(self):

        migration_lottery = np.random.uniform(
            0, 1, (number_males + number_females) * number_groups)

        number_migrations = sum(
            [1 for m in migration_lottery if m < migration_rate])

        number_females_migrating = random.randint(0, number_migrations)
        number_males_migrating = number_migrations - number_females_migrating

        self.groups_leaving = np.random.choice(self.groups,
                                               size=number_migrations)
        self.groups_coming = [
            random.choice([i for i in self.groups if i != l])
            for l in self.groups_leaving
        ]

        for gl, gc in zip(self.groups_leaving[:number_females_migrating],
                          self.groups_coming[:number_females_migrating]):

            fl = np.random.choice(gl.females_gestating_or_lactating)
            fc = np.random.choice(gc.females_gestating_or_lactating)

            gl.females_gestating_or_lactating.remove(fl)
            gc.females_gestating_or_lactating.remove(fc)

            gl.females_gestating_or_lactating.append(fc)
            gc.females_gestating_or_lactating.append(fl)

            fc.group_id = gl.id
            fl.group_id = gc.id

        for gl, gc in zip(self.groups_leaving[number_females_migrating:],
                          self.groups_coming[number_females_migrating:]):

            ml = np.random.choice(gl.males)
            mc = np.random.choice(gc.males)

            gl.males.remove(ml)
            gc.males.remove(mc)

            gl.males.append(mc)
            gc.males.append(ml)

            mc.group_id = gl.id
            ml.group_id = gc.id

    def evolve(self):
        for _ in range(number_generations):
            for g in self.groups:
                g.go_one_mating_season()
                g.generate_offspring(self.max_non_cycling_days,
                                     self.conception_probability_list,
                                     self.mean_days_to_conception,
                                     self.sd_days_to_conception)
                g.mating_matrix = np.array(
                    [np.array([1e-40] * number_males) for f in range(number_females)])
                g.mutate()
                g.mothers = []
                g.fathers = []

            
            self.migrate() if number_groups > 1 else 0
            print(_) if np.random.uniform(0, 1) > 0.999 else 0

        for g in self.groups:
            g.set_ranks()
            g.males = sorted(g.males, key=g.sort_by_id)


def set_parameters(number_generations_set = 100, number_groups_set = 3,
                   number_females_set = 10, number_males_set = 10, seasonality_set = 0.0, days_gestation_plus_lactation_set = 365,
                   fixed_number_of_cycles_set = False, mutation_rate_set = 0.01, migration_rate_set = 0.01,
                   cycle_length_set = 28, ovulation_set = 16, real_time_plots_set = False):
    

    model = []
    
    global number_generations
    global number_groups
    global number_females
    global number_males
    global number_agents
    global seasonality
    global days_gestation_plus_lactation
    
    global fixed_number_of_cycles
    global mutation_rate
    global migration_rate
    global cycle_length
    global ovulation
    global pre
    global post

    global real_time_plots
    
    number_generations = number_generations_set
    number_groups = number_groups_set
    number_females = number_females_set
    number_males = number_males_set
    number_agents = number_females + number_males
    seasonality = seasonality_set
    days_gestation_plus_lactation = days_gestation_plus_lactation_set

    fixed_number_of_cycles = fixed_number_of_cycles_set
    mutation_rate = mutation_rate_set
    migration_rate = migration_rate_set
    cycle_length = cycle_length_set
    ovulation = ovulation_set
    pre = ovulation - 6
    post = cycle_length - pre - 6

    real_time_plots = real_time_plots_set

    i = 0.04

    # finding the number for seasonality where it would produce a mating season longer than 365 days
    # and scaling minimum seasonality from that number to 1.0
    # setting seasonality to it's new point on that scale
    
    while abs(365 * (1 - i) + (cycle_length * (1 - i)) - 365) > 0.005:
        i += 0.00001

    seasonality = 1 - ((1 - i) * seasonality)

set_parameters()


                                                                                                   

