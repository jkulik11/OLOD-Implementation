import numpy as np
from astropy import units as u

def skew(vec):
	"""
	Return the skew symmetric matrix operator corresponding to the cross product

	Args: 
		vec (numpy array (3)):
			A vector
	Returns:
		mat (numpy array (3x3)):
			The cross product matrix associated with vec
	"""
	return np.array([[0., -1.*vec[2].value, vec[1].value],[vec[2].value,0.,-1.*vec[0].value],[-1.*vec[1].value, vec[0].value, 0.]])*(vec[[0]].unit)

def findSTM(r0, v0, rf, vf, dt):
	"""
	Return the state transition matrix associated with this trajectory

	Args: 
		r0 (numpy array (3)):
			Initial position vector
		v0 (numpy array (3)):
			Initial velocity vector	
		rf (numpy array (3)):
			Final position vector
		vf (numpy array (3)):
			Final velocity vector	
		dt (float):
			Time between the two states	
	Returns:
		stm (numpy array (6x6))
	"""
	#km and s units
	mu = 3.986004418E5 << u.km**3 / u.s**2
	r0Mag = np.linalg.norm(r0)
	rfMag = np.linalg.norm(rf)
	h = np.cross(r0,v0)
	sr0 = skew(r0)
	sv0 = skew(v0)
	srf = skew(rf)
	svf = skew(vf)
	sh = skew(h)
	B=np.transpose(np.vstack([r0/np.sqrt(mu*r0Mag), r0Mag*v0/mu]))
	Y0 = np.block([[sr0.value, -1.*np.matmul((np.matmul(sr0, sv0)+sh), B).value, -1.*np.transpose([r0])],[sv0.value, np.matmul(mu/r0Mag**3*np.matmul(sr0,sr0)-np.matmul(sv0,sv0), B).value, np.transpose([v0])/2.]])
	Yf = np.block([[srf.value, -1.*np.matmul((np.matmul(srf, svf)+sh), B).value, np.transpose([-1.*rf+3./2.*dt*vf])],[svf.value, np.matmul(mu/rfMag**3*np.matmul(srf,srf)-np.matmul(svf,svf), B).value, np.transpose([vf/2.-3./2.*mu/rfMag**3*dt*rf])]])
	return np.matmul(Yf, np.linalg.inv(Y0))

def groundStationPos(lon0,lat,ts):
	"""
	Return low fidelity ground station positions for a sensor

	Args:
		lon0 (float):
			Initial longitude in radians
		lat (float):
			Latitude in radians
		ts (numpy array (n)):
			Seconds for each position
	Returns:
		Rss (numpy array (n)):
			n positions of a hypothetical sensor on the ground
	"""
	Re = 6371. << u.km
	lons = lon0 + ts * 2.*np.pi/(1. * u.d)*u.rad
	for i in range(len(ts)):
		if i==0:
			Rss =Re*np.array([np.cos(lons[i])*np.cos(lat),np.sin(lons[i])*np.cos(lat),np.sin(lat)])
		else:
			Rss = np.vstack((Rss, Re*np.array([np.cos(lons[i])*np.cos(lat),np.sin(lons[i])*np.cos(lat),np.sin(lat)])))
	return Rss
	













