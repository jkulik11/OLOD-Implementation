from astropy import units as u
import numpy as np

from STMint.STMint import STMint
import util
import olod

def test1():
	# Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	"""
	Lines 19 - 24 are for creating orbit using orbit library
	Remove these lines for the 3 body case"""
	# a = 7000. << u.km
	# ecc = 0.1 << u.one
	# inc = 30. << u.deg
	# raan = 0. << u.deg
	# argp = 0. << u.deg
	# nu = 0. << u.deg
	# """need orbit to work with here. put info here when you get it and remove until the ########
	# Then continue like normal
	# """
	# orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	# print("Test Orbit r and v:")
	# print(orb.r)
	# print(orb.v)
	x0 = 1.02202151273581740824714855590570360
	z0 = 0.182096761524240501132977765539282777 
	yd0 = -0.103256341062793815791764364248006121

	t_max = 1.5111111111111111111111111111111111111111/4

	x_0 = np.array([x0, 0, z0, 0, yd0, 0])
	Orbit = STMint(preset = "threeBody", preset_mult = (1.0 / (81.30059 + 1.0)), variational_order=2)
	states, stms, stts, ts = Orbit.dynVar_int2([0, 1.5111111111111111111111111111111111111111/4], x_0, output="all")
	states = np.array(states)
	print(states)
	r = states[:,:3]
	Rss = np.zeros((len(r),3))
	for i in range(len(ts)):
		rho = r[i] - Rss[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
	#even with a GEO guess, we find a LEO orbit
	print(ls)
	r0Guess = x_0[:3]
	r0Guess = r0Guess * 1.001
	v0Guess = x_0[3:]
	# v0Guess[0] = 0.1
	print(f"Orbit Guess:\n r_0 = {r0Guess}\n v_0 = {v0Guess}")
	print("Running OLOD")
	soln = olod.olod(ls, t_max, Rss, r0Guess, v0Guess, .001, 1., 20)
	print("Solution Orbit:")
	print(soln)
	return soln

test1()

