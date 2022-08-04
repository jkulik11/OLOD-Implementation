from astropy import units as u
import numpy as np

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

import util
import olod

def test1():
	ts = 60.*np.arange(0., 10., 1.) << u.s
	Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	a = 7000. << u.km
	ecc = 0.1 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	print("Test Orbit r and v:")
	print(orb.r)
	print(orb.v)
	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		r = orbt.r
		rho = r-Rss[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	print("Solution Orbit:")
	print(soln)
	return soln

def testSTM():
	a = 7000. << u.km
	ecc = 0.01 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	dt = 100.*u.s
	orbt = orb.propagate(dt)
	dr0= np.array([3.,1.2,0.])
	dv0 = np.array([0.,0.,0.])
	orb1 = Orbit.from_vectors(Earth, orb.r + (dr0*orb.r.unit), orb.v + (dv0*orb.v.unit))
	orbt1 = orb1.propagate(dt)
	drt = orbt1.r - orbt.r
	dvt = orbt1.v - orbt.v
	stm = util.findSTM(orb.r, orb.v, orbt.r, orbt.v, dt)
	dxt = np.matmul(stm, np.hstack((dr0, dv0)))
	print(dxt)
	print(drt)
	print(dvt)
	
test1()

