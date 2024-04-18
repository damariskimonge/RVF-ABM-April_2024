"""
Define rvf model.
Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import numpy as np
import starsim as ss
from starsim.diseases.sir import SIS
import pylab as pl

__all__ = ['rvf']


#def death_by_age(module, sim, uids, p_old=0.1, p_young=0.5):
 #   age = age[uids]
 #   death_probs = np.zeros(len(uids))
 #   old = age == "Adult"
 #   young = age == "Calf"
 #   death_probs[old] = p_old
 #   death_probs[young] = p_young
 #   return death_probs


class rvf(SIS):

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

if __name__ == '__main__':
    import sciris as sc
    import starsim as ss

    rvf_disease = rvf()

    pars = sc.objdict(
        birth_rate = 32.6, #National Animal Census of 2021
        death_rate = 30, # National Animal Census of 2021
        n_agents = 1000,
        networks = 'random',
    )

    sim = ss.Sim(pars=pars, diseases=rvf_disease)
    sim.run()
    sim.plot()