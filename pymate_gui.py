import tkinter as tk
import pymate
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from pandastable import Table
import numpy as np

class gui():
    
    def __init__(self):
        
        self.root = tk.Tk()
        self.root.title('pymate_gui')
        
        self.create_tk_variables()
        self.hm_plot = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.hm_plot,master = self.root)
        self.make_buttons()
        self.make_pop_size_user_inputs()
        self.make_other_user_inputs()
        self.model_label = tk.Label(text = "Model: No model initiated yet")
        self.model_label.grid(row=0,column=1, columnspan = 3)
        
        self.root.mainloop()

    def setup_simulation(self):

        self.update_variables()
        self.model = pymate.population()
        self.update_labels_and_buttons("sim")
        
    def setup_demo(self):

        self.update_variables()
        self.model = pymate.population()
        self.update_labels_and_buttons("demo")

    def create_tk_variables(self):
        
        self.number_groups = tk.IntVar()
        self.number_groups.set(pymate.number_groups)

        self.number_females = tk.IntVar()
        self.number_females.set(pymate.number_females)
        
        self.number_males = tk.IntVar()
        self.number_males.set(pymate.number_males)

        self.seasonality = tk.DoubleVar()
        self.seasonality.set(pymate.seasonality)

        self.number_generations = tk.IntVar()
        self.number_generations.set(pymate.number_generations)

        self.mutation_rate = tk.DoubleVar()
        self.mutation_rate.set(pymate.mutation_rate)

        self.migration_rate = tk.DoubleVar()
        self.migration_rate.set(pymate.migration_rate)

        self.which_data = tk.StringVar()
        
    def update_variables(self):
        
        pymate.number_groups = int(self.number_groups_entry.get())
        self.number_groups.set(pymate.number_groups)

        pymate.number_females = int(self.number_females_entry.get())
        self.number_females.set(pymate.number_females)

        pymate.number_males = int(self.number_males_entry.get())
        self.number_males.set(pymate.number_males)

        pymate.seasonality = float(self.seasonality_entry.get())
        self.seasonality.set(pymate.seasonality)

        pymate.number_generations = int(self.number_generations_entry.get())
        self.number_generations.set(pymate.number_generations)

        pymate.mutation_rate = float(self.mutation_rate_entry.get())
        self.mutation_rate.set(pymate.mutation_rate)

        pymate.migration_rate = float(self.migration_rate_entry.get())
        self.migration_rate.set(pymate.migration_rate)

    def update_labels_and_buttons(self, demo_or_sim):

        self.model_label["text"] = f"""This model contains {pymate.number_groups} groups, each with {pymate.number_females} females and {pymate.number_males} males,
        and will run for {pymate.number_generations} generations."""

        self.which_data_radio_button_1['state'] = "normal"
        self.which_data_radio_button_2['state'] = "normal"
        
        if demo_or_sim == "sim":
            self.simulate_evolution_button['command'] = lambda: self.go_evolution()
            self.simulate_evolution_button['state'] = "normal"
            self.setup_demonstration_button['state'] = "disabled"
            self.one_day_button['state'] = "disabled"
            self.one_mating_season_button['state'] = "disabled"
        else:
            self.one_day_button['state'] = "normal"

            if self.which_data.get() == "mating_data":
                self.one_day_button['command'] = lambda: self.show_mating_df()
            else:
                self.one_day_button['command'] = lambda: self.show_agent_data_dfs()
            
                
            self.one_mating_season_button['command'] = lambda: self.demonstrate_mating_season_heatmap(self.model.groups[0])
            self.one_mating_season_button['state'] = "normal"

            self.simulate_evolution_button['state'] = "disabled"

    def go_evolution(self):

        self.simulate_evolution_button['state'] = "disabled"
        self.model.evolve()
        self.setup_demonstration_button['state'] = "normal"

        
    def demonstrate_mating_season_heatmap(self, group):

        try:
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError: 
            pass
        
        group.go_one_mating_season()
        self.hm_plot = group.plot_fertile_mating_success()
        self.canvas = FigureCanvasTkAgg(self.hm_plot,master = self.root)  
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9,column=0, columnspan = 8)
        self.setup_demo()
        
    def show_mating_df(self):

        self.model.groups[0].go_one_day()
        
        self.frame_mating = tk.Frame(self.root, width=150, height=230)
        self.frame_mating.grid(row=10,column=0, columnspan = 5)

        self.pt_mating = Table(self.frame_mating,height = 230)
        self.pt_mating.show()
        
        self.model.groups[0].make_mating_df()
        self.pt_mating.model.df = self.model.groups[0].mating_df
        
    def show_agent_data_dfs(self):

        self.model.groups[0].go_one_day()
        
        self.frame_m = tk.Frame(self.root, width=100, height=230)
        self.frame_f = tk.Frame(self.root, width=100,  height=230)
        self.frame_m.grid(row=10,column=0, columnspan = 5)
        self.frame_f.grid(row=11,column=0, columnspan = 5)

        self.pt_m = Table(self.frame_m,height = 230)
        self.pt_f = Table(self.frame_f,height = 230)
        self.pt_m.show()
        self.pt_f.show()

        self.model.groups[0].make_agent_data_dfs()
        self.pt_m.model.df = self.model.groups[0].male_data
        self.pt_f.model.df = self.model.groups[0].female_data
        
    def make_pop_size_user_inputs(self):
        
        self.number_groups_label = tk.Label(text = "Number of Groups", width = 18)
        self.number_groups_label.grid(row=5,column=0)
        self.number_groups_entry = tk.Entry(textvariable = self.number_groups)
        self.number_groups_entry.grid(row=5,column=1)
        
        self.number_females_label = tk.Label(text = "Number of Females", width = 18)
        self.number_females_label.grid(row=6,column=0)
        self.number_females_entry = tk.Entry(textvariable = self.number_females)
        self.number_females_entry.grid(row=6,column=1)
        
        self.number_males_label = tk.Label(text = "Number of Males", width = 18)
        self.number_males_label.grid(row=7,column=0)
        self.number_males_entry = tk.Entry(textvariable = self.number_males)
        self.number_males_entry.grid(row=7,column=1)
        
    def make_other_user_inputs(self):

        self.seasonality_label = tk.Label(text = "Breeding Seasonilty", width = 18)
        self.seasonality_label.grid(row=5,column=2)
        self.seasonality_entry = tk.Entry(textvariable = self.seasonality)
        self.seasonality_entry.grid(row=5,column=3)
        
        self.number_generations_label = tk.Label(text = "Number of Generations", width = 18)
        self.number_generations_label.grid(row=6,column=2)
        self.number_generations_entry = tk.Entry(textvariable = self.number_generations)
        self.number_generations_entry.grid(row=6,column=3)
        
        self.mutation_rate_label = tk.Label(text = "Mutation Rate", width = 18)
        self.mutation_rate_label.grid(row=7,column=2)
        self.mutation_rate_entry = tk.Entry(textvariable = self.mutation_rate)
        self.mutation_rate_entry.grid(row=7,column=3)
        
        self.migration_rate_label = tk.Label(text = "Migration Rate", width = 18)
        self.migration_rate_label.grid(row=8,column=2)
        self.migration_rate_entry = tk.Entry(textvariable = self.migration_rate)
        self.migration_rate_entry.grid(row=8,column=3)
        
    def make_buttons(self):
        
        self.setup_simulation_button = tk.Button(self.root, text = "Setup Simulation", width = 16, command = lambda: self.setup_simulation())
        self.setup_demonstration_button = tk.Button(self.root, text = "Setup Demonstration", width = 16,  command = lambda: self.setup_demo())
        self.simulate_evolution_button = tk.Button(self.root, text = "Simulate Evolution", width = 16,  state = "disabled")
        self.one_day_button = tk.Button(self.root, text = "Go One Day", width = 15,  state = "disabled")
        self.one_mating_season_button = tk.Button(self.root, text = "Go One Mating Season", width = 16,  state = "disabled")
        self.which_data_radio_button_1 = tk.Radiobutton(self.root, text="Show Agent Data", width = 16,  variable=self.which_data, value="agent_data", state = "normal")
        self.which_data_radio_button_2 = tk.Radiobutton(self.root, text="Shaw Mating Matrix", width = 16,  variable=self.which_data, value="mating_data", state = "normal")
        #self.which_data_radio_button_3 = tk. Radiobutton(self.root, text="Show Agent Data", variable=self.which_data, value="agent_data", state = "disabled")
        
        self.setup_simulation_button.grid(row=0,column=0)
        self.setup_demonstration_button.grid(row=1,column=0)
        self.simulate_evolution_button.grid(row=2,column=0)
        self.one_day_button.grid(row=3,column=0)
        self.one_mating_season_button.grid(row=4,column=0)
        self.which_data_radio_button_1.grid(row=2, column = 2)
        self.which_data_radio_button_2.grid(row=3, column = 2)
        
GUI = gui()       
