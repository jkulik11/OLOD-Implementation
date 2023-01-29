from util import skew, findSTM
from astropy import units as u
import numpy as np

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from dataclasses import dataclass


class ObservationEntry:
	"""
	Class associated with any information pertaining to a given observation
	"""
	def __init__(self, observerIndex, observedIndex, time, lVec, rVec=None):
		self.i = observerIndex
		self.j = observedIndex
		self.t = time
		self.l = lVec
		#default of None is used when both objects are unknown
		self.r = rVec
		#if riKnown then rj is assumed unknown if one of the object locations is known
		#self.riKnown = riKnown

def findOrbData(x0, dt):
	"""
	Propagate and find STM
	
	Args:
		x0 (numpy array (6)):
			initial state
		dt (float):
			propagation time
	Returns:
		finalState (numpy array (6)):
			state propgated by time dt
		stm (numpy array (6x6))
			state transition matrix associated with the initial conditon and propagation time
	"""
	r0 = x0[:3]
	v0 = x0[3:]
	orb = Orbit.from_vectors(Earth, r0 * u.km, v0 * u.km/u.s)
	orbNew = orb.propagate(dt*u.s)
	rt = orbNew.r.value
	vt = orbNew.v.value
	stm = findSTM(r0 * u.km, v0 * u.km/u.s, rt * u.km, vt * u.km/u.s, dt*u.s)
	return [np.concatenate((rt, vt)), stm]

def assemble_row(lmat, numUndet, i, j, prop):
	"""
	Helper function to assemble a row of the A matrix for least squares
	
	Args:
		lmat (numpy array (3x3)):
			Line of sight cross product matrix from observation
		numUndet (int):
			Number of objects with undetermined state
		i (int):
			Index of the observer
		j (int):
			Index of the observed
		prop (complicated):
			Contains two outputs of findOrbData for the i and j indices. An entry is None if i or j >= numUndet
	Returns:
		row matrix (3 x numUndet)
			
	"""
	return lmat @ np.hstack(tuple(map(lambda k: prop[0][1][:3] if k == i else -1.*prop[1][1][:3] if k == j else np.zeros((3,6)), range(numUndet))))


def general_olod_iteration(numUndet, numDet, data, initGuesses):
	"""
	Perform one iteration of optimal linear orbit determination

	Args: 
		numUndet (int):
			number of objects with undetermined state
		numDet (int):
			number of objects with determined state
		data (list of  ObservationEntry):
			observation data, assuming indices of undetermined objects precede known objects
		initGuesses (numpy array (numUndet x 6)):
			A guess for the initial state of each undetermined orbit at t=0
	Returns:
		x0Update (numpy array (numUndet x 6)):
			An updated guess at the initial orbit states
	"""
	#find states and STMs predicted at each time
	props = [[findOrbData(initGuesses[dat.i], dat.t) if dat.i < numUndet else None, 
	findOrbData(initGuesses[dat.j], dat.t) if dat.j < numUndet else None] for dat in data]

	lmats = tuple(map(lambda dat: skew(dat.l), data))
	A = np.vstack(tuple(map(lambda lmat, dat, prop: assemble_row(lmat, numUndet, dat.i, dat.j, prop), lmats, data, props)))
	b = np.hstack(tuple(map(lambda lmat, dat, prop: lmat @ ((prop[1][0][:3] if dat.j < numUndet else dat.r) - (prop[0][0][:3] if dat.i < numUndet else dat.r)), lmats, data, props)))
	#print("Least square setup")	
	#print(A)
	#print(b)
	dx0 = np.array(np.linalg.lstsq(A.value, b.value)[0])
	return initGuesses + np.reshape(dx0, (numUndet, 6))
		

def general_olod(numUndet, numDet, data, initGuesses, tol, maxIter):
	"""
	Perform iterations of optimal linear orbit determination until difference between iterations has norm less than tol

	Args: 
		numUndet (int):
			number of objects with undetermined state
		numDet (int):
			number of objects with determined state
		data (list of  ObservationEntry):
			observation data
		initGuesses (numpy array (numUndet x 6)):
			A guess for the initial state of each undetermined orbit at t=0
		tol (float):
			Tolerance for ending iterations
		maxIter (int):
			Max number of iterations to perform
	Returns:
		x0Fit (numpy array (numUndet x 6)):
			An updated guess at the initial orbit states
	"""
	success = False
	x0Guess = initGuesses
	for i in range(maxIter):
		x0GuessOld = x0Guess
		x0Guess = general_olod_iteration(numUndet, numDet, data, x0Guess)
		if np.linalg.norm(x0Guess-x0GuessOld) < tol:
			success = True
			break
	if not success:
		print("OLOD did not converge to tolerance")
	return x0Guess
