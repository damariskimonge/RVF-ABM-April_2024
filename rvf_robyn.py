"""
Define rvf model.
"""

import numpy as np
import pylab as pl
import sciris as sc
import starsim as ss


class RVF(ss.SIS):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """

        pars = ss.omergeleft(pars,
            # Natural history parameters, duration specified in days
            dur_exp = 0.01,    #  Assumption because very short
            dur_inf = 14,       # Assumption from human duration of infection
            p_death = 0.1,     #  In adult cattle: Rift Valley Fever Factsheet: Pennsylvania Dept of Health (2013)

            # Initial conditions and beta
            init_prev = 0.001, # Assumption
            beta = 0.04,     # From the Review of Mosquitoes associated with RFV virus in Madagascar paper (Tantely et al 2015)
            waning = 0.05,
            imm_boost = 1.0,   
            rel_sus_0 = 1,  # Baseline level of relative susceptibility
            rel_sus_1 = 1.2,  # Relative to in district 0
        )

        par_dists = ss.omergeleft(par_dists,
            dur_exp   = ss.normal,
            dur_inf   = ss.normal,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs) #?? what's the idea of this line

        # SIR are added automatically, here we add Exposed
        self.add_states(
            ss.State('exposed', bool, False),
            ss.State('recovered', float, np.nan),
            ss.State('immunity', float, 0.0),
            ss.State('ti_exposed', float, np.nan),
            ss.State('ti_recovered', float, np.nan),
            ss.State('ti_dead', float, np.nan),
        )

        return

    def initialize(self, sim):
        super().initialize(sim)
        self.rel_sus[sim.people.district==0] = self.pars.rel_sus_0
        self.rel_sus[sim.people.district==1] = self.pars.rel_sus_1

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cum_deaths', sim.npts, dtype=int, scale=True),
        ]
        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    def update_pre(self, sim):
        # Progress exposed -> infected
        infected = ss.true(self.exposed & (self.ti_infected <= sim.ti))
        self.exposed[infected] = False
        self.infected[infected] = True

        # Progress infected -> recovered
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity(sim)
        #return

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.year)
        self.results.new_deaths[sim.ti] = len(deaths)
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def update_immunity(self, sim):
        uids_0 = ss.true((self.immunity > 0) & (sim.people.district==0))
        uids_1 = ss.true((self.immunity > 0) & (sim.people.district==1))
        self.immunity[uids_0] = (self.immunity[uids_0])*(1 - self.pars.waning*sim.dt)
        self.rel_sus[uids_0] = np.maximum(0, self.pars.rel_sus_0 - self.immunity[uids_0])
        self.immunity[uids_1] = (self.immunity[uids_1])*(1 - self.pars.waning*sim.dt)
        self.rel_sus[uids_1] = np.maximum(0, self.pars.rel_sus_1 - self.immunity[uids_1])
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        # Do not call set_prognosis on parent
        # super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = sim.ti

        p = self.pars

        # Determine when exposed become infected
        self.ti_infected[uids] = sim.ti + p.dur_exp.rvs(uids) / sim.dt
        self.immunity[uids] += self.pars.imm_boost

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die] / sim.dt
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die] / sim.dt

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'recovered']:
            self.statesdict[state][uids] = False
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
            p_move_01 = 0.0006, # Could update this later to vary over time
            p_move_10 = 0.000000001
        )
        return

    def update_post(self, sim):
        super().update_post(sim)
        district0_uids = ss.true(self.district==0)
        district1_uids = ss.true(self.district==1)
        will_move_01 = self.pars.p_move_01 > np.random.rand(len(district0_uids))
        will_move_10 = self.pars.p_move_10 > np.random.rand(len(district1_uids))
        self.district[will_move_01] = 1
        self.district[will_move_10] = 0


class vaccination(ss.Intervention):  # Create a new, generic treatment intervention

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
    district = ss.State("district", int, ss.bernoulli(name="district", p=0.5))
    cattle = Cattle(n_agents=1000, extra_states=district)

    # The disease
    rvf = RVF()

    # Adding the parameters to the model
    pars = sc.objdict(
        start = 0,
        end = 100,  # Simulate for 100 days to see when the number of infected cows exceeds some threshold
        dt = 1,
        birth_rate = 32.6, #National Animal Census of 2021
        death_rate = 30, # National Animal Census of 2021
        networks = "random",    
    )

    #base_sim = ss.Sim(pars = pars, people = cattle, diseases = rvf)
    vacc_sim = ss.Sim(pars=pars, people=cattle, diseases=rvf, interventions=vaccination)
    
    #base_sim.run()
    vacc_sim.run()

    # Make sim.results.rvf.new_deaths a numpy array
    #base_new_deaths = np.array(base_sim.results.rvf.new_deaths)
    vacc_new_deaths = np.array(vacc_sim.results.rvf.new_deaths)
    # Find the index when new_deaths becomes 1 for the first time
    #base_index_death = np.where(base_new_deaths == 1)[0][0]
    vacc_index_death = np.where(vacc_new_deaths == 1)[0][0]

    # Make symptomatic cases a numpy array
    #base_signs = np.array((base_sim.results.rvf.n_infected)*0.07)
    vacc_signs = np.array((vacc_sim.results.rvf.n_infected)*0.07)
    # Find the index when signs becomes 1 for the first time
    #base_index_signs = np.where(base_signs >= 1)[0][0]
    vacc_index_signs = np.where(vacc_signs >= 1)[0][0]
    
    # Make plots
    pl.figure()
    #pl.plot(base_sim.yearvec, base_sim.results.rvf.n_infected, color = "black", label = "Baseline")
    pl.plot(vacc_sim.yearvec, vacc_sim.results.rvf.n_infected, color = "red", label = "Vaccinated")
    #pl.plot(sim.yearvec, signs, label = "Symptomatic Cases", color = "black")
    #pl.plot(sim.yearvec, sim.results.rvf.new_deaths, label = "New Deaths", color = "red")
    #pl.plot(sim.yearvec, sim.results.rvf.cum_deaths, label = "Cumulative Deaths", color = "green")
    #pl.plot(sim.yearvec, sim.results.rvf.new_infections, label = "New Infections", color = "orange")
    #pl.axvline(base_index_signs, color = "black", linestyle='--')
    pl.axvline(vacc_index_signs, color = "red", linestyle='--')
    #pl.axvline(base_index_death, color = "black")
    pl.axvline(vacc_index_death, color = "red")
    pl.title('RVF Number of Infected', fontsize = 20)
    pl.xlabel('Time in days', fontsize=12)  # X-axis title
    pl.ylabel('Number of Cattle', fontsize=12)  # Y-axis title
    pl.legend()
    pl.show()


    
