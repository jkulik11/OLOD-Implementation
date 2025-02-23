from astropy import units as u
import numpy as np

import util
import olod

def test1():
	ts = 60.*np.arange(0., 10., 1.) << u.s
	# Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	Rss = np.zeros((len(ts),3)) << u.km
	print("Rss: ", Rss)

	"""
	Lines 19 - 24 are for creating orbit using orbit library
	Remove these lines for the 3 body case"""
	a = 7000. << u.km
	ecc = 0.1 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	"""need orbit to work with here. put info here when you get it and remove until the ########
	Then continue like normal
	"""
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	print("Test Orbit r and v:")
	print(orb.r)
	print(orb.v)
	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		r = orbt.r
		####################################################
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

test1()

