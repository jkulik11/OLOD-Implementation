from astropy import units as u
import numpy as np

from STMint.STMint import STMint
import util
import olod

def test1():
	# Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)

	# Initial Conditions
	x0 = 1.02202151273581740824714855590570360
	y0 = 0.0
	z0 = 0.182096761524240501132977765539282777 
	xd0 = 0.0
	yd0 = -0.103256341062793815791764364248006121
	zd0 = 0.0
	x_0 = np.array([x0, y0, z0, xd0, yd0, zd0])

	# Initial guess
	delta = 1.001 # Amount to perturb the initial conditions for guess
	r0Guess = x_0[:3] * delta
	v0Guess = x_0[3:] * delta

	# Time information
	t_max = 1.5111111111111111111111111111111111111111/4 # time of simulation
	t_steps = 10 # Number of steps taken
	ts = np.linspace(0, t_max, num=t_steps) # time steps

	# Create True orbit
	Orbit = STMint(preset = "threeBody", preset_mult = (1.0 / (81.30059 + 1.0)), variational_order=2)
	states, stms, stts, ts = Orbit.dynVar_int2([0, ts[-1]], x_0, t_eval = ts, output="all")
	states = np.array(states)

	# Gather line of sight vectors
	r = states[:,:3]
	Rss = np.zeros((len(r),3))
	for i in range(len(ts)):
		rho = r[i] - Rss[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
	#even with a GEO guess, we find a LEO orbit
	print(f"Orbit Guess:\n r_0 = {r0Guess}\n v_0 = {v0Guess}")
	print("Running OLOD")
	soln = olod.olod(ls, ts, Rss, r0Guess, v0Guess, .001, 1., 20)
	print("Solution Orbit:")
	print(soln)
	return soln

test1()

