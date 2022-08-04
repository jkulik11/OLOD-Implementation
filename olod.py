from util import skew, findSTM
from astropy import units as u
import numpy as np

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

def olod_iteration(ls, Rss, ts, r0Guess, v0Guess):
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
	orb = Orbit.from_vectors(Earth, r0Guess, v0Guess)
	stms = []
	for i in range(len(ts)):
		orbNew = orb.propagate(ts[i])
		r = orbNew.r
		v = orbNew.v
		stm = findSTM(r0Guess, v0Guess, r, v,ts[i])
		if i == 0:
			rs = r
			vs = v
		else:
			rs = np.vstack((rs, r))
			vs = np.vstack((vs, v))
		stms.append(stm[:3,:])
	rhos = rs - Rss
	lmats = tuple(map(lambda l: skew(l), ls))
	A = np.vstack(tuple(map(lambda x, y: np.matmul(x, y), lmats, stms)))
	b = -1.*np.hstack(tuple(map(lambda x, y: np.matmul(x, y), lmats, rhos)))
	dx0 = np.array(np.linalg.lstsq(A.value, b.value)[0])
	#print("Deltas")
	#print(dx0)
	#print([(r0Guess.value + dx0[:3])*r0Guess.unit, (v0Guess.value + dx0[3:])*v0Guess.unit])
	return [(r0Guess.value + dx0[:3])*r0Guess.unit, (v0Guess.value + dx0[3:])*v0Guess.unit]
		

def olod(ls, Rss, ts, r0Guess, v0Guess, tolPos, tolVel, maxIter):
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
		x0Guess = olod_iteration(ls, Rss, ts, r0Guess, v0Guess)
		r0Guess = x0Guess[0]
		v0Guess = x0Guess[1]
		if np.linalg.norm(r0Guess-r0GuessOld) < tolPos and np.linalg.norm(v0Guess-v0GuessOld) < tolVel:
			success = True
			break
	if not success:
		print("OLOD did not converge to tolerance")
	return x0Guess

