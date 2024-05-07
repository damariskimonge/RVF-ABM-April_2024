"""
Define rvf model.
"""

import numpy as np 
import pylab as pl
import sciris as sc
import starsim as ss
from data_processing import p_kampala, prob_kampala_kiruhura, prob_kiruhura_kampala


class RVF(ss.SIS): # RVF class inherits from ss.SIS

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """

        pars = ss.dictmergeleft(pars,
            # Natural history parameters, duration specified in days
            dur_exp = 0.01,    #  Assumption because very short
            dur_inf = 14,       # Assumption from human duration of infection
            p_death = 0.1,     #  In adult cattle: Rift Valley Fever Factsheet: Pennsylvania Dept of Health (2013)

            # Initial conditions and beta
            init_prev = 0.001, # This is so that we simulate the spread from one infected cow.
            beta = 0.04,     # From the Review of Mosquitoes associated with RFV virus in Madagascar paper (Tantely et al 2015)
            waning = 0.05,
            imm_boost = 1.0,   
            rel_sus_0 = 1,  # Baseline level of relative susceptibility
            rel_sus_1 = 1.2,  # Relative to in district 0
        )

        par_dists = ss.dictmergeleft(par_dists,
            dur_exp   = ss.normal,
            dur_inf   = ss.normal,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs) #?? what's the idea of this line

        # SIR are added automatically, here we add Exposed
        self.add_states( 
            ss.BoolArr('exposed', bool, False), # Creates state exposed with initial false; options are T or F
            ss.FloatArr('recovered', float, np.nan), # Creates state recovered; initial value is missing
            ss.FloatArr('immunity', float, 0.0), # Creates state immunity; inital value is 0.0, holds a number ie level of immunity
            ss.FloatArr('ti_exposed', float, np.nan), # Creates time exposed; initial is missing, holds a number
            ss.FloatArr('ti_recovered', float, np.nan), # Creates time recovered; initial value is missing, holds a number
            ss.FloatArr('ti_dead', float, np.nan), # Creates time dead; initial is missing; holds a number
        )

        return

    def initialize(self, sim):
        super().initialize(sim) # adding initial relative susceptibility to already initialised object
        self.rel_sus[sim.people.district==0] = self.pars.rel_sus_0
        self.rel_sus[sim.people.district==1] = self.pars.rel_sus_1

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
        return self.infected | self.exposed

    def update_pre(self, sim):
        # Progress exposed -> infected
        infected = (self.exposed & (self.ti_infected <= sim.ti)).uids # Become infected if exposed and the time infected is <= current time
        self.exposed[infected] = False # Stop being exposed
        self.infected[infected] = True # Start being infected

        # Progress infected -> recovered
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids # Become recovered if infected and time recovered is <= current time
        self.infected[recovered] = False # stop being infected
        self.susceptible[recovered] = True # start being susceptible again
        self.update_immunity(sim) # gain immunity
        #return

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids # dead if time dead is <= current time 
        self.results.new_deaths[sim.ti] = len(deaths) # store update new_death result
        if len(deaths):
            sim.people.request_death(deaths) #The simulation should process the deaths if they exist.
        return

    def update_immunity(self, sim):
        uids_0 = ((self.immunity > 0) & (sim.people.district==0)).uids
        uids_1 = ((self.immunity > 0) & (sim.people.district==1)).uids
        self.immunity[uids_0] = (self.immunity[uids_0])*(1 - self.pars.waning*sim.dt) # update immunity
        self.rel_sus[uids_0] = np.maximum(0, self.pars.rel_sus_0 - self.immunity[uids_0]) # update susceptibility
        self.immunity[uids_1] = (self.immunity[uids_1])*(1 - self.pars.waning*sim.dt)
        self.rel_sus[uids_1] = np.maximum(0, self.pars.rel_sus_1 - self.immunity[uids_1])
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        # Do not call set_prognosis on parent
        # super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False # Move from susceptible to exposed
        self.exposed[uids] = True
        self.ti_exposed[uids] = sim.ti # store this time

        p = self.pars 

        # Determine when exposed become infected
        self.ti_infected[uids] = sim.ti + p.dur_exp.rvs(uids) / sim.dt
        self.immunity[uids] += self.pars.imm_boost # Boosting the immunity

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids) 
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die] / sim.dt # set time of death
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die] / sim.dt # set time of recoverey

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'recovered']:
            self.statesdict[state][uids] = False # Sets all states to False because now dead
        return
    
    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti
        res.cum_deaths[ti] = np.sum(res['new_deaths'][:ti+1])
        return


class Cattle(ss.People):
    def __init__(self, n_agents, extra_states=None):
        super().__init__(n_agents, extra_states=extra_states)
        self.pars = sc.objdict(
            p_move_01 = prob_kiruhura_kampala, # Could update this later to vary over time
            p_move_10 = prob_kampala_kiruhura
        )
        return

    def update_post(self, sim): #called after each simulation step to update the state of the agents
        super().update_post(sim)
        district0_uids = (self.district==0).uids # get uids of those in each district
        district1_uids = (self.district==1).uids
        will_move_01 = self.pars.p_move_01 > np.random.rand(len(district0_uids))  # get who will move
        will_move_10 = self.pars.p_move_10 > np.random.rand(len(district1_uids))
        moving_01_uids = district0_uids[will_move_01] # get the uids of those who will move
        moving_10_uids = district1_uids[will_move_10]
        self.district[moving_01_uids] = 1 # actual movement
        self.district[moving_10_uids] = 0


class vaccination(ss.Intervention):  

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


if __name__ == '__main__':

    # Make the cattle with the extra state for district
    district = ss.FloatArr("district", None, p_kampala)
    cattle = Cattle(n_agents=1000, extra_states=district)

    # The disease
    rvf = RVF()

    # The intervention
    vaccination = vaccination()

    # Adding the parameters to the model
    pars = sc.objdict(
        start = 0,
        end = 365,  # Simulate for 365 days (1 year) for seasonality
        dt = 1,
        birth_rate = 32.6, #National Animal Census of 2021
        death_rate = 30, # National Animal Census of 2021
        networks = "random",    
    )

    base_sim = ss.Sim(pars = pars, people = cattle, diseases = rvf)
    #vacc_sim = ss.Sim(pars=pars, people=cattle, diseases=RVF(), interventions=vaccination())
    
    base_sim.run()
    #vacc_sim.run()

    # Make sim.results.rvf.new_deaths a numpy array
    base_new_deaths = np.array(base_sim.results.rvf.new_deaths)
    #vacc_new_deaths = np.array(vacc_sim.results.rvf.new_deaths)
    
    # Find the index when new_deaths becomes 1 for the first time
    base_index_death = np.where(base_new_deaths == 1)[0][0]
    #vacc_index_death = np.where(vacc_new_deaths == 1)[0][0]

    # Make symptomatic cases a numpy array
    base_signs = np.array((base_sim.results.rvf.n_infected)*0.07)
    #vacc_signs = np.array((vacc_sim.results.rvf.n_infected)*0.07)
    
    # Find the index when signs becomes 1 for the first time
    base_index_signs = np.where(base_signs >= 1)[0][0]
    #vacc_index_signs = np.where(vacc_signs >= 1)[0][0]
    
    # Make plots
    pl.figure()
    pl.plot(base_sim.yearvec, base_sim.results.rvf.n_infected, color = "black", label = "Baseline Infected")
    #pl.plot(base_sim.yearvec, base_signs, label = "Symptomatic Cases", color = "black")
    #pl.plot(base_sim.yearvec, base_sim.results.rvf.new_deaths, label = "New Deaths", color = "red")
    #pl.plot(base_sim.yearvec, base_sim.results.rvf.cum_deaths, label = "Cumulative Deaths", color = "green")
    #pl.plot(base_sim.yearvec, base_sim.results.rvf.new_infections, label = "New Infections", color = "orange")
    pl.axvline(base_index_signs, color = "black", linestyle='--')
    pl.axvline(base_index_death, color = "black")
    pl.title('RVF Number of Infected', fontsize = 20)
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()


