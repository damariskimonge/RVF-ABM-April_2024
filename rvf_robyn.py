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
            beta = 0.33,     # From the Review of Mosquitoes associated with RFV virus in Madagascar paper (Tantely et al 2015)
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


class Cattle(ss.People):
    def __init__(self, n_agents, extra_states=None):
        super().__init__(n_agents, extra_states=extra_states)
        self.pars = sc.objdict(
            p_move_01 = 0.25, # Could update this later to vary over time
            p_move_10 = 0.25
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

    sim = ss.Sim(pars=pars, people=cattle, diseases=rvf)
    sim.run()
    
    # Make plots
    pl.figure()
    pl.plot(sim.yearvec, sim.results.rvf.n_infected)
    pl.title('Number infected')
    pl.show()

