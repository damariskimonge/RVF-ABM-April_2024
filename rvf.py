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


class rvf(SIS):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """

        pars = ss.omergeleft(pars,
            # Natural history parameters, all specified in days
            dur_exp = 8,       #>> we need model how the vector transmits the disease to humans 
            dur_inf = 6,       # time from exposure to the virus to the onset of symptoms, typically ranges from 2 to 6 days
            p_death = 0.21,     # mortality rate for RVF in cows can range from 10% to 30% during outbreaks, but it can be higher in severe cases or in naive herds 

            # Initial conditions and beta
            init_prev = 0.38, # Prevalence rates in animals can range from a few percentage points to more than 50% in some areas during outbreaks
            beta = 0.33,     # From the Review of Mosquitoes associated with RFV virus in Madagascar paper (Tantely et al 2015)
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
            ss.State('ti_exposed', float, np.nan),
            ss.State('ti_recovered', float, np.nan),
            ss.State('ti_dead', float, np.nan),
        )

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

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.year)
        if len(deaths):
            sim.people.request_death(deaths)
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
        n_agents = 1000,
        networks = 'random',
    )
    sim = ss.Sim(pars=pars, diseases=rvf_disease)
    sim.run()
    sim.plot()