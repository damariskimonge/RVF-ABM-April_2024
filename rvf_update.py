"""
Define rvf model.
"""

import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir) # Set the current directory to the working directory


import numpy as np 
import pylab as pl
import sciris as sc
import starsim as ss
from data_processing import unique_districts, district_probabilities #Assigning cattle to the various districts
from data_processing import total_pop # Getting the total population of cattle in Uganda
from data_processing import movement_prob_array # Getting the probability of movement of cattle
from data_processing import env_arrays # Get the environment data
from data_processing import date_list # Get the month and year
from data_processing import prev_by_district # Get the prevalence by district

# Parameters:
n_contacts = 10
n_agents = 5000
pars = sc.objdict(
    start = 0,
    end = 1,  # Simulate for 365 days (1 year) for seasonality
    dt = 1/365, # The default is a year so I'm making it a day
    dt_demog = 1/365, # also making this a day
    birth_rate = (32.6/365), #National Animal Census of 2021
    death_rate = (30/365) # National Animal Census of 2021
    #total_pop = total_pop # This is so that the values are scaled to the actual population.
    )

# The disease class
class RVF(ss.SIS): # RVF class inherits from ss.SIS

    def __init__(self, env_arrays, district_probabilities, prev_by_district=None, default_prevalence=0.0, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """
        self.env_arrays = env_arrays # assigning the environmental data
        self.district_probabilities = district_probabilities  # assigning district probabilities
        self.prev_by_district = prev_by_district or {}
        self.default_prevalence = default_prevalence

        pars = ss.dictmergeleft(pars,
            # Natural history parameters, duration specified in days
            dur_inf = 7,       # Musa Sekamatte et al.
            dur_inc = 3,      # Musa Sekamatte et al.
            p_symp = 0.07,  # Abel Walekhwa
            p_death = 0.05,     #  Wright et al.

            # Initial conditions and beta
            init_prev = 0.11, # Tumusiime et al.
            beta = 0.04,     # Abel Walekhwa
            waning = 0.05,   # Assumption
            imm_boost = 1.0, # Assumption 
        )

        par_dists = ss.dictmergeleft(par_dists, # Distributions
            dur_inf   = ss.poisson,
            dur_inc = ss.poisson,
            init_prev = ss.bernoulli,
            p_symp = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs) # this intialises the sim calling the base classes

        # Susceptible and Infected are added automatically
        self.add_states( 
            ss.BoolArr('symptomatic'), # Creates state symptomatic with initial value false
            ss.FloatArr('ti_symptomatic'), # Create time symptomatic with initial value missing; holds a number
            ss.BoolArr('recovered'), # Creates state recovered with initial value false
            ss.FloatArr('ti_recovered'), # Create time recovered with inital value missing; holds a number
            ss.FloatArr('ti_dead') # Creates time dead; initial is missing; holds a number
        )

        return

    def init_vals(self, sim):
        """ Set initial values for states based on district-specific prevalence """
        district_vals = sim.people.states['district'].values
        init_prev = np.zeros(len(sim.people), dtype=bool)
        
        for district in np.unique(district_vals):
            district_mask = (district_vals == district)
            prevalence = self.prev_by_district.get(district, self.default_prevalence)
            initial_cases = ss.bernoulli(prevalence).filter(uids=district_mask)
            init_prev[initial_cases] = True
        
        initial_cases = np.where(init_prev)[0]
        self.set_prognoses(sim, initial_cases)
        return

    def initialize(self, sim):
        super().initialize(sim)
        
        # Calculate the baseline susceptibility
        all_rain = []
        all_veg = []
        all_temp = []

        for district in self.env_arrays:
            rain_arr = self.env_arrays[district]['rain_arr']
            veg_arr = self.env_arrays[district]['veg_arr']
            temp_arr = self.env_arrays[district]['temp_arr']
            all_rain.extend(rain_arr)
            all_veg.extend(veg_arr)
            all_temp.extend(temp_arr)

        mean_rain = np.mean(all_rain)
        mean_veg = np.mean(all_veg)
        mean_temp = np.mean(all_temp)

        self.baseline_sus = mean_rain * mean_veg * mean_temp
        
    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
            ss.Result(self.name, 'cum_deaths', sim.npts, dtype=int),
        ]
        return
    @property
    def infectious(self): # Here infectious are the infected
        return self.infected 

    @property
    def asymptomatic(self):
        return self.infected & ~self.symptomatic

    def update_pre(self, sim):
        """ Update pre-disease progression steps """
        # Update relative susceptibility for each timestep
        num_days = 365
        
        for i in range(len(sim.people)):
            district = int(sim.people.states['district'].values[i])
            rain_arr = self.env_arrays[district]['rain_arr']
            veg_arr = self.env_arrays[district]['veg_arr']
            temp_arr = self.env_arrays[district]['temp_arr']
            current_sus =  ((veg_arr[sim.ti % num_days]) * (temp_arr[sim.ti % num_days])) * (rain_arr[sim.ti % num_days])  # The % operator ensures that the array indices remain within bounds by wrapping around after reaching the end of the array.
            self.rel_sus[i] = (current_sus / self.baseline_sus) * (1 - self.immunity[i])   # Calculate relative susceptibility based on environment and immunity


        # Identify new births (age == 0)
        new_births = (sim.people.states['age'] == 0).uids # For some reason there are no new births
        if len(new_births) > 0:
            self.assign_new_births_to_districts(sim, new_births)


        # Progress susceptible -> infected
        infected = (self.ti_infected <= sim.ti).uids
        self.infected[infected] = True
        self.susceptible[infected] = False  

        # Progress infected -> symptomatic
        symptomatic = ((self.ti_infected <= sim.ti) & (self.ti_symptomatic <= sim.ti)).uids
        self.symptomatic[symptomatic] = True # Become symptomatic   

        # Progress symptomatic -> recovered
        recovered = ((self.ti_infected <= sim.ti) & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.symptomatic[recovered] = False
        self.recovered[recovered] = True    
        self.susceptible[recovered]= True # become susceptible again

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids
        if len(deaths):
            sim.people.request_death(deaths)
        return
     
    def assign_new_births_to_districts(self, sim, new_uids):
        # Assign each new individual to a district based on the predefined probabilities
        districts = np.arange(len(self.district_probabilities))
        new_districts = np.random.choice(districts, size=len(new_uids), p=self.district_probabilities)

        for i, uid in enumerate(new_uids):
            sim.people.states['district'][uid] = new_districts[i]

        return

    
    def update_immunity(self, sim, recovered_uids=None):
        """ Update immunity levels """

        if recovered_uids is not None:
            self.immunity[recovered_uids] += self.pars.imm_boost  # Boost immunity after recovery
        
        immune_uids = (self.immunity > 0).uids
        self.immunity[immune_uids] = self.immunity[immune_uids] * (1 - self.pars.waning)  # Update immunity for all with some immunity
        return

    def set_prognoses(self, sim, uids, source_uids=None): # uids here is newly infected individuals
        """ Set prognoses for those who get infected ie we decide if they will be symptomatic, die or recover when they get infected """
        #super().set_prognoses(sim, uids, source_uids) # call the base classes

        # Determine when susceptible become infected
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        p = self.pars 

        # Determine who becomes symptomatic and when
        will_symp = p.p_symp.filter() # I'm guessing this initialises because it allows us to get accurate results
        symp_uids = p.p_symp.filter(uids)
    
        if len(symp_uids) > 0:
            dur_inc = p.dur_inc.rvs(len(symp_uids))
            self.ti_symptomatic[symp_uids] = self.ti_infected[symp_uids] + dur_inc

        # Sample duration of infection, being careful to only sample from the distribution once per timestep.
        dur_inf = p.dur_inf.rvs(len(uids)) 

        # Determine who dies and who recovers and when  
        will_die = p.p_death.filter() # This is also so that I get the correct number of dead and recovered
        dead_uids, rec_uids = p.p_death.filter(uids, both=True)
        if len(dead_uids) > 0:
            dead_indices = np.isin(uids, dead_uids)
            self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[dead_indices]
        
        if len(rec_uids) > 0:
            rec_indices = np.isin(uids, rec_uids)
            self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[rec_indices]

        return

    def update_death(self, sim, uids):

        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'infected', 'symptomatic', 'recovered']:
            self.statesdict[state][uids] = False # Sets all states to False because now dead
        return

    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res.new_deaths[:ti+1])
        return


# The People Class
class Cattle(ss.People):
    def __init__(self, n_agents, movement_prob_array, extra_states=None):
        super().__init__(n_agents, extra_states=extra_states)
        self.movement_prob_array = movement_prob_array
        return

    def update_post(self, sim): # After the transmission has run within the time step
        super().update_post(sim)
        
        n_districts = len(self.movement_prob_array)
        districts = self.states['district'].values.astype(int)  # Ensure districts are treated as integers

        # Loop over each cow to update its district based on movement probabilities
        for cow_id in range(len(districts)):
            current_district = districts[cow_id]
            move_prob = self.movement_prob_array[current_district]  # Get the movement probabilities for the current district

            # Determine the new district based on the movement probabilities
            new_district = np.random.choice(n_districts, p=move_prob)

            # Update the cow's district
            self.states['district'].values[cow_id] = new_district
        
        return


# Initialize the Cattle class with the number of agents, districts, and movement probabilities
unique_districts = len(district_probabilities)
district_values = np.random.choice(unique_districts, size=n_agents, p=district_probabilities)  # Assign initial districts based on population probabilities
district = ss.FloatArr("district", None, district_values)  # Create the district state for cattle using FloatArr

# The Network Class        
class Cattlenetwork(ss.DynamicNetwork): # Inherits from DynamicNetwork
    def __init__(self, pars = None, key_dict = None, **kwargs): # 
        super().__init__(pars=pars, key_dict = key_dict, **kwargs)
        
    def initialize(self, sim): # initialise 
        super().initialize(sim)
        self.update_contacts_within_district(sim)

    def add_contact(self, p1, p2, sim, beta=1.0, dur=1.0):
        """ Add a contact between two agents. """
        new_contact = {
            'p1': np.array([p1], dtype=np.int32),
            'p2': np.array([p2], dtype=np.int32),
            'beta': np.array([beta], dtype=np.float32),
            'dur': np.array([dur], dtype=np.float32),
        }
        self.append(new_contact)
        
    def update_contacts_within_district(self, sim): # set contacts
        """ Update contacts within the same district."""
        districts = sim.people.states['district'].values # Get district values
        
        # Initialize the district contacts dictionary
        unique_districts = np.unique(districts)
        district_contacts = {district: [] for district in unique_districts} # Separate contacts by district
        
        for i, district in enumerate(districts):
            potential_contacts = np.where(districts == district)[0]  # Agents in the same district
            if len(potential_contacts) > n_contacts:
                contacts = np.random.choice(potential_contacts, n_contacts, replace=False)  # Randomly choose 4 contacts
            else:
                contacts = [contact for contact in potential_contacts if contact != i]  # All contacts except self
            
            for contact in contacts:
                district_contacts[district].append((i, contact))  # Add to district contacts
                            
        # Store contacts
        for district, pairs in district_contacts.items():
            for p1, p2 in pairs:
                self.add_contact(p1, p2, sim, beta = 1.0, dur = 1/365)

        return
                
    def step(self, sim): # end existing pairs and update contacts
        self.end_pairs(sim.people)
        self.update_contacts_within_district(sim)


# The Intervention Class: Vaccination
class Vaccination(ss.Intervention):  

    def __init__(self, prob=0.12, efficacy=0.623):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of vaccination
        self.efficacy = efficacy

    def apply(self, sim):

        # Define  who is eligible for vaccination
        eligible_ids = sim.people.uid[disease.susceptible]  # People are eligible for vacc if they are susceptible
        n_eligible = len(eligible_ids) # Number of people who are eligible

        # Define who receives  vaccination
        is_vaccinated = np.random.rand(n_eligible) < self.prob  # Define which of the n_eligible people get treated by comparing np.random.rand() to self.p
        vaccinated_ids = eligible_ids[is_vaccinated]  # Pull out the IDs for the people receiving the treatment

        # vaccinating cattle will reduce susceptibility
        rvf.rel_sus[vaccinated_ids] = 1-self.efficacy 

        return



# The Intervention Class: Quarantine
#class Quarantine(ss.Intervention):  

    # If symptomatic
    # Make contacts zero
    # Stop movement of all cattle in and out of said district




# Running the simulation
base_sim = ss.Sim(pars = pars, label = "baseline", people = Cattle(n_agents=n_agents, movement_prob_array=movement_prob_array, extra_states=district), diseases = RVF(env_arrays=env_arrays, district_probabilities=district_probabilities, prev_by_district=prev_by_district), networks = Cattlenetwork())

vacc_sim = ss.Sim(pars= pars, label = "vaccination", people = Cattle(n_agents=n_agents, movement_prob_array=movement_prob_array, extra_states = district), diseases = RVF(env_arrays=env_arrays, district_probabilities = district_probabilities), interventions = Vaccination(), networks = Cattlenetwork())
#quar_sim = ss.Sim(pars = pars, label = "quarantine", people = Cattle(n_agents=n_agents, movement_prob_array=movement_prob_array, extra_states = district), diseases = RVF(env_arrays = env_arrays) interventions = Quarantine(), networks = Cattlenetwork()) 


#base_sim.run()
#vacc_sim.run()


# Make sim.results.rvf.new_deaths a numpy array
#base_new_deaths = np.array(base_sim.results.rvf.new_deaths)
#vacc_new_deaths = np.array(vacc_sim.results.rvf.new_deaths)
    
# Find the index when new_deaths becomes 1 for the first time
#base_index_death = np.where(base_new_deaths == 1)[0][0]
#vacc_index_death = np.where(vacc_new_deaths == 1)[0][0]

# Make symptomatic cases a numpy array
#base_signs = np.array((base_sim.results.rvf.n_infected)*0.07)
#vacc_signs = np.array((vacc_sim.results.rvf.n_infected)*0.07)
    
# Find the index when signs becomes 1 for the first time
#base_index_signs = np.where(base_signs >= 1)[0][0]
#vacc_index_signs = np.where(vacc_signs >= 1)[0][0]
    

def plot_n_susceptible():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.n_susceptible)[:len(date_list)], color="blue", label="Susceptible")
    pl.title('RVF Number of Susceptible', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()   

plot_n_susceptible()

def plot_n_infected():
    pl.figure()
    pl.plot(date_list, base_sim.results.rvf.n_infected[:len(date_list)], color="black", label="Number Infected") # The results are truncated to be only one year
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()

plot_n_infected()

def plot_n_symptomatic():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.n_symptomatic)[:len(date_list)], color="blue", label="Symptomatic Cases")
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()   

plot_n_symptomatic()

def plot_n_recovered():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.n_recovered)[:len(date_list)], color="blue", label="Recovered")
    pl.title('RVF Number of Recovered', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()   

plot_n_recovered()

def plot_n_dead(): # The cumulative deaths
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.cum_deaths)[:len(date_list)], color="blue", label="Cumulative Deaths")
    pl.title('RVF Cumulative Deaths', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()   

plot_n_dead()

def plot_rel_sus():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.rel_sus)[:len(date_list)], color="blue", label="Relative susceptibility")
    pl.title('RVF Relative Susceptibility', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Relative Susceptibility', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()   

plot_rel_sus()





