import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class male:

    def __init__(self, m, g, gene=0.0):

        self.id = m
        self.group_id = g
        self.quality = np.random.uniform(0.1, 0.9)
        self.competitive_effort = gene
        self.gene = gene


class female:

    def __init__(self,
                 f,
                 max_non_cycling_days,
                 conception_probability_list,
                 mean_days_to_conception,
                 sd_days_to_conception,
                 g,
                 gene=0.0):

        self.id = f
        self.group_id = g
        self.days_until_cycling = random.randint(0, max_non_cycling_days + 1)
        self.conception_probability_list = conception_probability_list
        self.conception_probability = "N/A"
        self.status = "not yet cycling"
        self.cycle_day = "N/A"
        self.gene = gene

        self.days_until_conception = abs(
            round(
                np.random.normal(mean_days_to_conception,
                                 sd_days_to_conception)))

    def switch_to_cycling(self, not_yet_cycling, females_cycling):

        np.append(females_cycling, self)
        np.remove(not_yet_cycling, self)
        self.status = "cycling"
        self.days_until_cycling = "N/A"

        self.cycle_day = 0
        self.conception_probability = self.conception_probability_list[
            0]  # starts day 0

    def switch_to_finished_cycling(self, females_cycling, finished_cycling):

        np.append(finished_cycling, self)
        np.remove(females_cycling, self)
        self.status = "finished cycling"
        self.cycle_day = "N/A"

        self.conception_probability = "N/A"


class group:

    # when a 'group' object is initialized, 'male' and 'female' objects are automatically instantiated in lists
    # ('males' and 'females') contained in the 'group' object
    # the male dominance hierarchy is set when the 'setRanks' method runs during 'group' initialization

    def __init__(self, g, max_non_cycling_days, conception_probability_list,
                 mean_days_to_conception, sd_days_to_conception):

        self.id = g
        self.females_not_yet_cycling = np.array([
            female(f,
                   max_non_cycling_days,
                   conception_probability_list,
                   mean_days_to_conception,
                   sd_days_to_conception,
                   g=self.id) for f in range(number_females)
        ])
        self.females_cycling = np.array([])
        self.females_finished_cycling = np.array([])

        self.males = np.array([male(m, g=self.id) for m in range(number_males)])

        self.mating_matrix = np.array(
            [np.array([1e-40] * number_males) for f in range(number_females)])

        self.list_of_rank_quality_corrlations = np.array([])

    def set_ranks(self):
        #         self.rank_entries = [(m.quality * m.competitive_effort) +
        #                              random.uniform(0, 1 - m.competitive_effort)
        #                              for m in self.males]

        #         self.rank_entries_scaled = [
        #             e / sum(self.rank_entries) for e in self.rank_entries
        #         ] if sum(self.rank_entries) > 0 else [
        #             1 / number_males for r in self.rank_entries
        #         ]

        self.rank_entries = [m.competitive_effort + 1e-50 for m in self.males]

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

        np.append(self.list_of_rank_quality_corrlations,
            np.corrcoef([m.id for m in self.males],
                        [m.quality for m in self.males])[1, 0])

    def start_cycling(self):

        switch_to_cycling_list = np.array([])

        for f in self.females_not_yet_cycling:
            f.days_until_cycling -= 1
            if f.days_until_cycling < 0:
                np.append(switch_to_cycling_list, f)

        [
            f.switch_to_cycling(self.females_not_yet_cycling,
                                self.females_cycling)
            for f in switch_to_cycling_list
        ]

    def end_cycling(self, switch_to_finished_cycling_list):

        [
            f.switch_to_finished_cycling(self.females_cycling,
                                         self.females_finished_cycling)
            for f in switch_to_finished_cycling_list
        ]

    def make_mating_pairs(self):

        for m, f in enumerate(np.random.permutation(
                self.females_cycling)):  #randomize cycling females
            self.mating_matrix[
                f.id][m] += f.conception_probability * self.males[m].quality

    def go_one_day(self):

        self.start_cycling()

        switch_to_finished_cycling_list = np.array([])
        for f in self.females_cycling:
            f.days_until_conception -= 1
            if f.days_until_conception < 0:
                np.append(switch_to_finished_cycling_list, f)
            else:
                f.cycle_day = f.cycle_day + 1 if f.cycle_day < cycle_length - 1 else 0
                f.conception_probability = f.conception_probability_list[
                    f.cycle_day]

        self.end_cycling(switch_to_finished_cycling_list)

        self.make_mating_pairs() if any(
            [f.conception_probability for f in self.females_cycling]
        ) else 0  # only run function to make mating pairs if conception is possible

    def go_one_mating_season(self):

        self.set_ranks()
        self.males = sorted(self.males, key=self.sort_by_id)

        while len(self.females_finished_cycling) < number_females:
            self.go_one_day()

    def determine_next_gen_parents(self):

        self.females_finished_cycling = sorted(self.females_finished_cycling,
                                               key=self.sort_by_id)

        total_conception_probabilities = np.array([])
        self.parents = np.array([])

        for f in range(number_females):
            self.mating_matrix[f] = [
                fms / sum(self.mating_matrix[f])
                for fms in self.mating_matrix[f]
            ]

        for _ in [0, 1]:
            for mother in self.females_finished_cycling:

                potential_fathers = np.random.choice(self.males, size=3)

                father = random.choices(potential_fathers,
                                        weights=self.mating_matrix[mother.id][[
                                            p.id for p in potential_fathers
                                        ]],
                                        k=1)[0]

                np.append(self.parents, [mother, father])

    def generate_offspring(self, max_non_cycling_days,
                           conception_probability_list,
                           mean_days_to_conception, sd_days_to_conception):

        self.females_not_yet_cycling = []
        self.males = []

        self.parents = np.random.permutation(
            self.parents)  # randomize order to avoid biasing offspring sex

        for i, p in enumerate(
                self.parents[:number_females]
        ):  # loop through parents until reaching number females
            new_gene = np.mean([p[0].gene, p[1].gene])
            self.females_not_yet_cycling.append(
                female(i,
                       max_non_cycling_days,
                       conception_probability_list,
                       mean_days_to_conception,
                       sd_days_to_conception,
                       g=self.id,
                       gene=new_gene))

        for i, p in enumerate(
                self.parents[number_males:]
        ):  # loop through remaining parents until reaching number males
            new_gene = np.mean([p[0].gene, p[1].gene])
            self.males.append(male(i, g=self.id, gene=new_gene))

    def reset(self):

        self.females_finished_cycling = []

        self.mating_matrix = np.array(
            [np.array([1e-40] * number_males) for f in range(number_females)])

    def mutate(self):

        mutation_lottery = np.random.uniform(0, 1,
                                             number_males + number_females)

        number_mutations = sum(
            [1 for m in mutation_lottery if m < mutation_rate])

        for m in range(number_mutations):
            agent_mutating = random.choice(self.males +
                                           self.females_not_yet_cycling)
            agent_mutating.gene += np.random.uniform(-0.01, 0.01)
            if agent_mutating.gene > 1 or agent_mutating.gene < 0:
                agent_mutating.gene = round(agent_mutating.gene)

    def recombination(self):

        pass

    def sort_by_id(self, agent):
        return agent.id


class population:

    def __init__(self):

        pre = ovulation - 6
        post = cycle_length - pre - 6

        self.max_non_cycling_days = round(365 - (365 * seasonality))

        self.conception_probability_list = [0] * pre + [
            .05784435, .16082819, .19820558, .25408223, .24362408, .10373275
        ] + [0] * post

        self.mean_days_to_conception = 50
        self.sd_days_to_conception = 0  # * (1.0 - seasonality)

        self.groups = np.array([
            group(g, self.max_non_cycling_days,
                  self.conception_probability_list,
                  self.mean_days_to_conception, self.sd_days_to_conception)
            for g in range(number_groups)
        ])

    def migrate(self):

        migration_lottery = np.random.uniform(
            0, 1, (number_males + number_females) * number_groups)

        number_migrations = sum(
            [1 for m in migration_lottery if m < migration_rate])

        number_females_migrating = random.randint(0, number_migrations)
        number_males_migrating = number_migrations - number_females_migrating

        self.groups_leaving = np.random.choice(self.groups,
                                               size=number_migrations)
        self.groups_coming = np.array([
            random.choice(self.groups[self.groups != l])
            for l in self.groups_leaving
        ])

        for gl, gc in zip(self.groups_leaving[:number_females_migrating],
                          self.groups_coming[:number_females_migrating]):

            fl = np.random.choice(gl.females_not_yet_cycling)
            fc = np.random.choice(gc.females_not_yet_cycling)

            gl.females_not_yet_cycling.remove(fl)
            gc.females_not_yet_cycling.remove(fc)

            gl.females_not_yet_cycling.append(fc)
            gc.females_not_yet_cycling.append(fl)

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
                g.determine_next_gen_parents()
                g.generate_offspring(self.max_non_cycling_days,
                                     self.conception_probability_list,
                                     self.mean_days_to_conception,
                                     self.sd_days_to_conception)
                g.reset()
                g.mutate()

            self.migrate()
            print(_) if np.random.uniform(0, 1) > 0.1 else 0

        for g in self.groups:
            g.set_ranks()
            g.males = sorted(g.males, key=g.sort_by_id)


number_generations = 100
number_groups = 3
number_females = 20
number_males = 20
seasonality = 0.1

mutation_rate = 0.01
migration_rate = 0.01
cycle_length = 28
ovulation = 16
pre = ovulation - 6
post = cycle_length - pre - 6

real_time_plots = False
