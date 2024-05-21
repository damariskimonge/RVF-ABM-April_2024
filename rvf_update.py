"""
Define rvf model.
"""

import os
os.chdir("C:/Users/angel/OneDrive/Documents/cema/Github/RVF-ABM-April_2024")

import numpy as np 
import pylab as pl
import sciris as sc
import starsim as ss
from data_processing import unique_districts, district_probabilities #Assigning cattle to the various districts
from data_processing import total_pop # Getting the total population of cattle in Uganda
from data_processing import movement_prob_array # Getting the probability of movement of cattle
from data_processing import env_arrays
from data_processing import date_list

# The disease class
class RVF(ss.SIS): # RVF class inherits from ss.SIS

    def __init__(self, env_arrays, pars=None, par_dists=None, *args, **kwargs):
        self.env_arrays = env_arrays
        """ Initialize with parameters """
        pars = ss.dictmergeleft(pars,
            # Natural history parameters, duration specified in days
            #dur_exp = 0.01,    #  Assumption because very short
            dur_inf = 14,       # Assumption from human duration of infection
            p_death = 0.1,     #  In adult cattle: Rift Valley Fever Factsheet: Pennsylvania Dept of Health (2013)

            # Initial conditions and beta
            init_prev = 0.001, # Consider changing so that it becomes dependent on district.
            beta = 0.04,     # From the Review of Mosquitoes associated with RFV virus in Madagascar paper (Tantely et al 2015)
            waning = 0.05,   # Assumption
            imm_boost = 1.0, # Assumption 
        )

        par_dists = ss.dictmergeleft(par_dists, # Distributions
            #dur_exp   = ss.normal,
            dur_inf   = ss.normal,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs) # this intialises the sim

        # SIR are added automatically, here we add Exposed
        self.add_states( 
            #ss.BoolArr('exposed', bool, False), # Creates state exposed with initial false; options are T or F
            ss.FloatArr('recovered', float, np.nan), # Creates state recovered; initial value is missing
            ss.FloatArr('immunity', float, 0.0), # Creates state immunity; inital value is 0.0, holds a number ie level of immunity
            #ss.FloatArr('ti_exposed', float, np.nan), # Creates time exposed; initial is missing, holds a number
            ss.FloatArr('ti_recovered', float, np.nan), # Creates time recovered; initial value is missing, holds a number
            ss.FloatArr('ti_dead', float, np.nan), # Creates time dead; initial is missing; holds a number
            ss.FloatArr('rel_sus', float, 0.0), # Creates state rel_sus with initial value 0.0
        )

        return

    def initialize(self, sim):
        super().initialize(sim) # adding initial relative susceptibility to already initialised object
        num_days = 365
        num_cattle = len(sim.people)  # Assuming sim.people is an array-like structure with cattle data

        # Initialize relative susceptibility array
        rel_sus = np.zeros(num_cattle)

        # Calculate relative susceptibility based on the district of each cow
        for i in range(num_cattle):
            district = sim.people.district[i]
            if district in self.env_arrays:
                rain_arr = self.env_arrays[district]['rain_arr']
                veg_arr = self.env_arrays[district]['veg_arr']
                temp_arr = self.env_arrays[district]['temp_arr']
                rel_sus[i] = (rain_arr[i % num_days] / 100) * (veg_arr[i % num_days] / 1000) * (temp_arr[i % num_days] / 10)
                # Make relative susceptibility a function of the first day of the first district.
                
        # Assign the calculated relative susceptibility to the state
        self.rel_sus[:] = rel_sus
        
        # Register the state with starsim
        sim.people.register_state('rel_sus', self.rel_sus)        
                
    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim) # Calls the results of the superclass
        self.results += [ # Here we are adding new results
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cum_deaths', sim.npts, dtype=int, scale=True),
        ]
        return

    @property
    def infectious(self): # Here infectious are both the infected and exposed
        return self.infected #| self.exposed

    def update_pre(self, sim):
        # I think I will also put relative susceptibility here so that updated as a loop.

        # Progress exposed -> infected
        #infected = (self.exposed & (self.ti_infected <= sim.ti)).uids # Become infected if exposed and the time infected is <= current time
        infected = (self.ti_infected <= sim.ti).uids
        #self.exposed[infected] = False # Stop being exposed
        self.infected[infected] = True # Start being infected

        # Progress infected -> recovered
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids # Become recovered if infected and time recovered is <= current time
        self.infected[recovered] = False # stop being infected
        self.susceptible[recovered] = True # start being susceptible again
        self.update_immunity(sim) # gain immunity
        #return

        # also define the immunity within here perhaps.

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids # dead if time dead is <= current time 
        self.results.new_deaths[sim.ti] = len(deaths) # store update new_death result
        if len(deaths):
            sim.people.request_death(deaths) #The simulation should process the deaths if they exist.
        return

    def update_immunity(self, sim):
        uids = (self.immunity > 0).uids
        self.immunity[uids] = (self.immunity[uids]) * (1 - self.pars.waning)  # Update immunity
        self.rel_sus[uids] = np.maximum(0, self.rel_sus[uids] - self.immunity[uids])  # Update susceptibility
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        # Do not call set_prognosis on parent
        # super().set_prognoses(sim, uids, source_uids)

        #self.susceptible[uids] = False # Move from susceptible to exposed
        #self.exposed[uids] = True
        #self.ti_exposed[uids] = sim.ti # store this time

        p = self.pars 

        # Determine when exposed become infected
        #self.ti_infected[uids] = sim.ti + p.dur_exp.rvs(uids)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        self.immunity[uids] += self.pars.imm_boost # Boosting the immunity

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids) 
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die]  # set time of death
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die] # set time of recoverey

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        #for state in ['susceptible', 'exposed', 'infected', 'recovered']:
        for state in ['susceptible', 'infected', 'recovered']:
            self.statesdict[state][uids] = False # Sets all states to False because now dead
        return
    
    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti
        res.cum_deaths[ti] = np.sum(res['new_deaths'][:ti+1])

        # ensure I have: SIRD here.
        # also perhaps prevalence
        return

# The disease
rvf = RVF(env_arrays=env_arrays)

# The People Class
class Cattle(ss.People):
    def __init__(self, n_agents, movement_prob_array, extra_states=None):
        super().__init__(n_agents, extra_states=extra_states)
        self.movement_prob_array = movement_prob_array
        return

    def update_post(self, sim):
        super().update_post(sim)
        
        n_districts = len(self.movement_prob_array)
        #print(f"Number of districts: {n_districts}")
        districts = self.states['district'].values.astype(int)  # Ensure districts are treated as integers
        #print(f"districts array size: {len(districts)}")

        # Loop over each cow to update its district based on movement probabilities
        for cow_id in range(len(districts)):
            current_district = districts[cow_id]
            move_prob = self.movement_prob_array[current_district]  # Get the movement probabilities for the current district

            # Determine the new district based on the movement probabilities
            new_district = np.random.choice(n_districts, p=move_prob)

            # Debugging: Print statements to understand the issue
            #print(f"Cow ID: {cow_id}")
            #print(f"Current District: {current_district}")
            #print(f"Movement Probabilities: {move_prob}")
            #print(f"New District: {new_district}")

            # Update the cow's district
            self.states['district'].values[cow_id] = new_district
        
        return


# Initialize the Cattle class with the number of agents, districts, and movement probabilities
n_agents = 1000
unique_districts = len(district_probabilities)
district_values = np.random.choice(unique_districts, size=n_agents, p=district_probabilities)  # Assign initial districts based on population probabilities
district = ss.FloatArr("district", None, district_values)  # Create the district state for cattle using FloatArr

cattle = Cattle(n_agents=n_agents, movement_prob_array=movement_prob_array, extra_states=district)

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
            if len(potential_contacts) > 4:
                contacts = np.random.choice(potential_contacts, 4, replace=False)  # Randomly choose 4 contacts
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

# The network
cattle_network = Cattlenetwork()

# The Intervention Class: Vaccination
class Vaccination(ss.Intervention):  

    def __init__(self, prob=0.12, efficacy=0.623):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of vaccination
        self.efficacy = efficacy

    def apply(self, sim):

        # Define  who is eligible for vaccination
        eligible_ids = sim.people.uid[rvf.susceptible]  # People are eligible for vacc if they are susceptible
        n_eligible = len(eligible_ids) # Number of people who are eligible

        # Define who receives  vaccination
        is_vaccinated = np.random.rand(n_eligible) < self.prob  # Define which of the n_eligible people get treated by comparing np.random.rand() to self.p
        vaccinated_ids = eligible_ids[is_vaccinated]  # Pull out the IDs for the people receiving the treatment

        # vaccinating cattle will reduce susceptibility
        rvf.rel_sus[vaccinated_ids] = 1-self.efficacy 

# The intervention
vaccination = Vaccination()

# The Intervention Class: Quarantine
class Quarantine(ss.Intervention):  

    def __init__(self, prob=0.12, efficacy=0.623):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of vaccination
        self.efficacy = efficacy

    def apply(self, sim):

        # Define  who is eligible for vaccination
        eligible_ids = sim.people.uid[rvf.susceptible]  # People are eligible for vacc if they are susceptible
        n_eligible = len(eligible_ids) # Number of people who are eligible

        # Define who receives  vaccination
        is_vaccinated = np.random.rand(n_eligible) < self.prob  # Define which of the n_eligible people get treated by comparing np.random.rand() to self.p
        vaccinated_ids = eligible_ids[is_vaccinated]  # Pull out the IDs for the people receiving the treatment

        # vaccinating cattle will reduce susceptibility
        rvf.rel_sus[vaccinated_ids] = 1-self.efficacy 

# The intervention
quarantine = Quarantine()

# Adding the parameters to the model
pars = sc.objdict(
  start = 0,
  end = 1,  # Simulate for 365 days (1 year) for seasonality
  dt = 1/365, # The default is a year so I'm making it a day
  birth_rate = (32.6/365)*1e3, #National Animal Census of 2021
  death_rate = (30/365)*1e3, # National Animal Census of 2021
  total_pop = total_pop # This is so that the values are scaled to the actual population.
   )

base_sim = ss.Sim(pars = pars, people = cattle, diseases = rvf, networks = cattle_network)
#vacc_sim = ss.Sim(pars=pars, people=cattle, diseases=RVF(), interventions=vaccination(), networks = network)
    
base_sim.run()
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
    
# Make plots:
# First is susceptible
# Second is infected
# 2b is symptomatic
# Third is recovered
# Fourth is dead
# Others are first sign
# Others are first death
# Compare the three scenarios here
def plot_n_infected():
    pl.figure()
    pl.plot(date_list, base_sim.results.rvf.n_infected[:len(date_list)], color="black", label="Number Infected") # The results are truncated to be only one year
    #pl.axvline(10, color="black", linestyle='--')  # First sign
    #pl.axvline(12, color="black")  # First death
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()
    

#plot_n_infected()

def plot_n_symptomatic():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.n_infected * 0.07)[:len(date_list)], color="blue", label="Symptomatic Cases")
    #pl.axvline(10, color="black", linestyle='--')  # First sign
    #pl.axvline(12, color="black")  # First death
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()
    

#plot_n_symptomatic()

def plot_new_deaths():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.new_deaths[:len(date_list)]), color="red", label="New Deaths")
    #pl.axvline(10, color="black", linestyle='--')  # First sign
    #pl.axvline(12, color="black")  # First death
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()

#plot_new_deaths()
    
def plot_cum_deaths():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.cum_deaths[:len(date_list)]), color="green", label="Cumulative Deaths")
    #pl.axvline(10, color="black", linestyle='--')  # First sign
    #pl.axvline(12, color="black")  # First death
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()
    
#plot_cum_deaths()

def plot_new_infections():
    pl.figure()
    pl.plot(date_list, (base_sim.results.rvf.new_infections[:len(date_list)]), color="orange", label="New Infections")
    #pl.axvline(10, color="black", linestyle='--')  # First sign
    #pl.axvline(12, color="black")  # First death
    pl.title('RVF Number of Infected', fontsize=20)  # Title of the plot
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()
    
#plot_new_infections()



