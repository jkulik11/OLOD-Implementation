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

# Need to nondimensionalize things I believe. convert earths radius as well as the earth rotation rate to be non dimensional. get the nondimensional rotation rate of earth moon system
# Use these values to get the lons.

	Re = 6371 / 384400 # Earth radius in nondimensional form (radius of the earth divided by the distance from earth to moon)
	t_nondimensional = 1 / 27.32 # Nondimensional period for one earth rotation (1 earth day / 1 lunar month)
	lons = lon0 + ts * 2.*np.pi/(t_nondimensional)
	for i in range(len(ts)):
		adjusted_lons = lons[i] - (2 * np.pi * ts[i]) # Adjusted to include the fact that the earth moon frame is rotating, causing a change in lons
		if i==0:
			Rss =Re*np.array([np.cos(adjusted_lons)*np.cos(lat),np.sin(adjusted_lons)*np.cos(lat),np.sin(lat)])
		else:
			Rss = np.vstack((Rss, Re*np.array([np.cos(adjusted_lons)*np.cos(lat),np.sin(adjusted_lons)*np.cos(lat),np.sin(lat)])))
	return Rss
	













