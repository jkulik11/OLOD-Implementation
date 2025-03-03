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
	return np.array([[0., -1.*vec[2], vec[1]],[vec[2],0.,-1.*vec[0]],[-1.*vec[1], vec[0], 0.]])

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
	













