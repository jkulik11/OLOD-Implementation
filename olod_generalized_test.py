from astropy import units as u
import numpy as np

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

import util
import olod_generalized

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
	obss = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(1, 0, time, lVec, rVec), ts.value, ls, Rss.value)))
	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	#soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	soln = olod_generalized.general_olod(1, 1, obss, np.array([np.concatenate((r0Guess.value, v0Guess.value))]), .001, 20)
	print("Solution Orbit:")
	print(soln)
	return soln

def test2():
	ts = 60.*np.arange(0., 10., 1.) << u.s
	Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	Rss1 = util.groundStationPos(np.pi/2.*u.rad,np.pi/3.*u.rad,ts)
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
		rho1 = r-Rss1[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(1, 0, time, lVec, rVec), ts.value, ls, Rss.value)))
	obss1 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(2, 0, time, lVec, rVec), ts.value, ls1, Rss1.value)))
	obss = np.hstack((obss,obss1))
	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	#soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	soln = olod_generalized.general_olod(1, 2, obss, np.array([np.concatenate((r0Guess.value, v0Guess.value))]), .001, 20)
	print("Solution Orbit:")
	print(soln)
	return soln

def test3():
	ts = 60.*np.arange(0., 10., 1.) << u.s
	Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	Rss1 = util.groundStationPos(np.pi/2.*u.rad,np.pi/3.*u.rad,ts)
	a = 7000. << u.km
	ecc = 0.1 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	orb1 = Orbit.from_classical(Earth, a, ecc+(0.01 << u.one), inc, raan, argp, nu)
	print("Test Orbit r and v:")
	print(orb.r)
	print(orb.v)
	print(orb1.r)
	print(orb1.v)
	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		r = orbt.r
		rho = r-Rss[i]
		rho1 = r-Rss1[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(2, 0, time, lVec, rVec), ts.value, ls, Rss.value)))
	obss1 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(3, 0, time, lVec, rVec), ts.value, ls1, Rss1.value)))
	obss = np.hstack((obss,obss1))

	for i in range(len(ts)):
		orbt = orb1.propagate(ts[i])
		r = orbt.r
		rho = r-Rss[i]
		rho1 = r-Rss1[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss2 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(2, 1, time, lVec, rVec), ts.value, ls, Rss.value)))
	obss3 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(3, 1, time, lVec, rVec), ts.value, ls1, Rss1.value)))
	obss4 = np.hstack((obss2,obss3))

	obss = np.hstack((obss,obss4))

	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	#soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	soln = olod_generalized.general_olod(2, 2, obss, np.array([np.concatenate((r0Guess.value, v0Guess.value)), np.concatenate((r0Guess.value, v0Guess.value))]), .001, 20)
	print("Solution Orbit:")
	print(soln)
	return soln

def test4():
	ts = 60.*np.arange(0., 10., 1.) << u.s
	Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	Rss1 = util.groundStationPos(np.pi/2.*u.rad,np.pi/3.*u.rad,ts)
	a = 7000. << u.km
	ecc = 0.1 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	orb1 = Orbit.from_classical(Earth, a, ecc+(0.01 << u.one), inc + (0.01 << u.deg), raan, argp, nu +(0.001 << u.deg))
	orb2 = Orbit.from_classical(Earth, a, ecc+(0.005 << u.one), inc, raan, argp, nu)
	print("Test Orbit r and v:")
	print(orb.r)
	print(orb.v)
	print(orb1.r)
	print(orb1.v)
	print(orb2.r)
	print(orb2.v)
	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		r = orbt.r
		rho = r-Rss[i]
		rho1 = r-Rss1[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(3, 0, time, lVec, rVec), ts.value, ls, Rss.value)))
	obss1 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(4, 0, time, lVec, rVec), ts.value, ls1, Rss1.value)))
	obss = np.hstack((obss,obss1))


	for i in range(len(ts)):
		orbt = orb1.propagate(ts[i])
		r = orbt.r
		rho = r-Rss[i]
		rho1 = r-Rss1[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss2 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(3, 1, time, lVec, rVec), ts.value, ls, Rss.value)))
	obss3 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(4, 1, time, lVec, rVec), ts.value, ls1, Rss1.value)))
	obss4 = np.hstack((obss2,obss3))

	obss = np.hstack((obss,obss4))


	for i in range(len(ts)):
		orbt = orb2.propagate(ts[i])
		orbt1 = orb.propagate(ts[i])
		orbt2 = orb1.propagate(ts[i])
		r = orbt.r
		rho = r-orbt1.r
		rho1 = r-orbt2.r
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss2 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(0, 2, time, lVec, rVec), ts.value, ls, orbt1.r.value)))
	obss3 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(1, 2, time, lVec, rVec), ts.value, ls1, orbt2.r.value)))
	obss4 = np.hstack((obss2,obss3))

	obss = np.hstack((obss,obss4))


	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	#soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	soln = olod_generalized.general_olod(3, 2, obss, np.array([np.concatenate((r0Guess.value, v0Guess.value)), np.concatenate((r0Guess.value, v0Guess.value)), np.concatenate((r0Guess.value, v0Guess.value))]), .001, 20)
	print("Solution Orbit:")
	print(soln)
	return soln



def test5():
	ts = 60.*np.arange(0., 10., 1.) << u.s
	Rss = util.groundStationPos(0*u.rad,np.pi/4.*u.rad,ts)
	Rss1 = util.groundStationPos(np.pi/2.*u.rad,np.pi/3.*u.rad,ts)
	a = 7000. << u.km
	ecc = 0.1 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	orb1 = Orbit.from_classical(Earth, a, ecc+(0.01 << u.one), inc + (0.01 << u.deg), raan, argp, nu +(0.001 << u.deg))
	orb2 = Orbit.from_classical(Earth, a, ecc+(0.005 << u.one), inc, raan, argp, nu)
	print("Test Orbit r and v:")
	print(orb.r)
	print(orb.v)
	print(orb1.r)
	print(orb1.v)
	print(orb2.r)
	print(orb2.v)
	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		r = orbt.r
		rho = r-Rss[i]
		rho1 = r-Rss1[i]
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
	obss = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(3, 0, time, lVec, rVec), ts.value, ls, Rss.value)))
	obss1 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(4, 0, time, lVec, rVec), ts.value, ls1, Rss1.value)))
	obss = np.hstack((obss,obss1))


	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		orbt1 = orb1.propagate(ts[i])
		orbt2 = orb2.propagate(ts[i])
		r = orbt.r
		rho = r-orbt1.r
		rho1 = r-orbt2.r
		rho2 = orbt2.r-orbt1.r
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
		if i == 0:
			ls2 = rho2/np.linalg.norm(rho2)
		else:
			ls2 = np.vstack((ls2, rho2/np.linalg.norm(rho2)))
	obss2 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(0, 1, time, lVec, rVec), ts.value, ls, orbt.r.value)))
	obss3 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(0, 2, time, lVec, rVec), ts.value, ls1, orbt.r.value)))
	obss4 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(1, 2, time, lVec, rVec), ts.value, ls2, orbt1.r.value)))
	obss5 = np.hstack((obss2,obss3, obss4))

	obss = np.hstack((obss,obss5))


	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	#soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	soln = olod_generalized.general_olod(3, 2, obss, np.array([.9*np.concatenate((orb.r.value, orb.v.value)), 1.5*np.concatenate((orb1.r.value, orb1.v.value)), .5*np.concatenate((orb2.r.value, orb2.v.value))]), .001, 20)
	print("Solution Orbit:")
	print(soln)
	return soln



def test6():
	ts = 60.*60.*np.arange(0., 10., 1.) << u.s
	a = 7000. << u.km
	ecc = 0.1 << u.one
	inc = 30. << u.deg
	raan = 0. << u.deg
	argp = 0. << u.deg
	nu = 0. << u.deg
	orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
	orb1 = Orbit.from_classical(Earth, a, ecc+(0.01 << u.one), inc + (0.01 << u.deg), raan, argp, nu +(0.001 << u.deg))
	orb2 = Orbit.from_classical(Earth, a, ecc+(0.005 << u.one), inc, raan, argp, nu)
	print("Test Orbit r and v:")
	print(orb.r)
	print(orb.v)
	print(orb1.r)
	print(orb1.v)
	print(orb2.r)
	print(orb2.v)

	for i in range(len(ts)):
		orbt = orb.propagate(ts[i])
		orbt1 = orb1.propagate(ts[i])
		orbt2 = orb2.propagate(ts[i])
		r = orbt.r
		rho = r-orbt1.r
		rho1 = r-orbt2.r
		rho2 = orbt2.r-orbt1.r
		if i == 0:
			ls = rho/np.linalg.norm(rho)
		else:
			ls = np.vstack((ls, rho/np.linalg.norm(rho)))
		if i == 0:
			ls1 = rho1/np.linalg.norm(rho1)
		else:
			ls1 = np.vstack((ls1, rho1/np.linalg.norm(rho1)))
		if i == 0:
			ls2 = rho2/np.linalg.norm(rho2)
		else:
			ls2 = np.vstack((ls2, rho2/np.linalg.norm(rho2)))
	obss2 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(0, 1, time, lVec, rVec), ts.value, ls, orbt.r.value)))
	obss3 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(0, 2, time, lVec, rVec), ts.value, ls1, orbt.r.value)))
	obss4 = np.hstack(tuple(map(lambda time, lVec, rVec: olod_generalized.ObservationEntry(1, 2, time, lVec, rVec), ts.value, ls2, orbt1.r.value)))
	obss5 = np.hstack((obss2,obss3, obss4))

	obss = obss5


	#even with a GEO guess, we find a LEO orbit
	r0Guess = ls[0] * 42164. *u.km
	v0Guess = np.array([0,-3.07467,0])*u.km/u.s
	print("Running OLOD")
	#soln = olod.olod(ls, Rss, ts, r0Guess, v0Guess, .001*u.km, 1. * u.m/u.s, 20)
	soln = olod_generalized.general_olod(3, 0, obss, np.array([np.concatenate((orb.r.value, orb.v.value)), .999*np.concatenate((orb1.r.value, orb1.v.value)), np.concatenate((orb2.r.value, orb2.v.value))]), .0000000001, 400)
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
	
test6()

