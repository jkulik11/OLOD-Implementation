from util import skew
from astropy import units as u
import numpy as np
from STMint.STMint import STMint

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

def olod_iteration(ls, t_max, t_steps, Rss, r0Guess, v0Guess):
	"""
	Perform one iteration of optimal linear orbit determination

	Args: 
		ls (numpy array (nx3)):
			n line of sight unit vector observations
		Rss (numpy array (nx3)):
			n locations of the sensor
		ts (numpy array (n)):
			n floats representing the times of the observations
		r0Guess (numpy array (3)):
			A guess for the initial state of the orbit at t=0
		v0Guess (numpy array (3)):
			A guess for the initial state of the orbit at t=0
	Returns:
		x0Update (numpy array (2x3)):
			An updated guess at the initial orbit state
	"""
	#find states and STMs predicted at each time
	x_initial = np.array([r0Guess[0], r0Guess[1], r0Guess[2], v0Guess[0], v0Guess[1], v0Guess[2]], dtype=object)
	Orbit = STMint(preset = "threeBody", preset_mult = (1.0 / (81.30059 + 1.0)), variational_order=2)
	states, stms, stts, ts = Orbit.dynVar_int2([0, t_max], x_initial, t_eval= np.linspace(0, t_max, num=t_steps), output="all")
	stms = np.array(stms)
	stms = stms[:,:3,:]
	states = np.array(states)
	rs = states[:,:3]
	vs = states[:,3:]
	rhos = rs - Rss
	lmats = tuple(map(lambda l: skew(l), ls))
	A = np.vstack(tuple(map(lambda x, y: np.matmul(x, y), lmats, stms)))
	b = -1.*np.hstack(tuple(map(lambda x, y: np.matmul(x, y), lmats, rhos)))
	dx0 = np.array(np.linalg.lstsq(A, b, rcond=None)[0])
	#print("Deltas")
	#print(dx0)
	#print([(r0Guess.value + dx0[:3])*r0Guess.unit, (v0Guess.value + dx0[3:])*v0Guess.unit])
	return [(r0Guess + dx0[:3]), (v0Guess + dx0[3:])]
		

def olod(ls, t_max, t_steps, Rss, r0Guess, v0Guess, tolPos, tolVel, maxIter):
	"""
	Perform iterations of optimal linear orbit determination until difference between iterations has norm less than tol

	Args: 
		ls (numpy array (nx3)):
			n line of sight unit vector observations
		Rss (numpy array (nx3)):
			n locations of the sensor
		ts (numpy array (n)):
			n floats representing the times of the observations
		r0Guess (numpy array (3)):
			A guess for the initial state of the orbit at t=0
		v0Guess (numpy array (3)):
			A guess for the initial state of the orbit at t=0
		tolPos (float):
			Tolerance for ending iterations
		tolVel (float):
			Tolerance for ending iterations
		maxIter (int):
			Max number of iterations to perform
	Returns:
		x0Fit (numpy array (2x3)):
			An updated guess at the initial orbit state
	"""
	success = False
	for i in range(maxIter):
		r0GuessOld = r0Guess
		v0GuessOld = v0Guess
		x0Guess = olod_iteration(ls, t_max, t_steps, Rss, r0Guess, v0Guess)
		r0Guess = x0Guess[0]
		v0Guess = x0Guess[1]
		if np.linalg.norm(r0Guess-r0GuessOld) < tolPos and np.linalg.norm(v0Guess-v0GuessOld) < tolVel:
			success = True
			Iterations = i + 1
			print(f"Converged in {Iterations} iterations")
			break
	if not success:
		print("OLOD did not converge to tolerance")
	return x0Guess

