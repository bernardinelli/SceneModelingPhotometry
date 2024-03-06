from multismp import *


class MovingDetection(MultiDetection):
	"""
	Requires RA and Dec for the detection as well as rate of motion, an exposure and CCD numbers for bookkeeping and
	zero-point retrieval, a band (for finding extra exposures) and an optional color (for astrometry) and name for the
	detection
 
	This will assume that the motion is linear, and user has to provide dra/dt, ddec/dt terms
	"""

	def __init__(self, ra, dec, rarate, decrate, expnum, ccdnum, band, color=0.61, name="", color_bg = 0.61):
		"""
		Constructor class
		"""
		self.ra = ra
		self.dec = dec
		self.rarate = rarate
		self.decrate = decrate
		self.expnum = expnum
		self.ccdnum = ccdnum
		self.band = band
		self.color = color
		self.name = name
		self.color_bg = color_bg
  
	def produceExposureList(self, exposures, detected):
		"""
		The key difference between this and `findAllExposures` is that here I will assume that the list of exposures
		already exists somewhere
		"""
		ccdlist = tb.Table()
		ccdlist['EXPNUM'] = exposures['EXPNUM']
		ccdlist['CCDNUM'] = exposures['CCDNUM']
		ccdlist['MJD_OBS'] = exposures['MJD_OBS']
		ccdlist['BAND'] = exposures['BAND']
		ccdlist.sort("EXPNUM")

		ccdlist["DETECTED"] = False
		ccdlist["DETECTED"][np.isin(ccdlist["EXPNUM"], detected)] = True
  
	
		self.exposures = tb.unique(ccdlist)
		self.exposures['DELTA_T'] = self.exposures['MJD_OBS'] - self.exposures[self.exposures['EXPNUM'] == self.expnum]['MJD_OBS']
  
	def constructPSFs(
		self,
		ra_grid,
		dec_grid,
		pmc=DESMaps(),
		size=30,
		shift_x=0,
		shift_y=0,
		path="",
	
	):
		"""
		Constructs the PIFF PSFs for the detections, requires an array of RA and Decs (ra_grid, dec_grid), a pixmappy instance (pmc),
		a stamp size, a potential offset in pixels for the center (shift_x,y), a path for the
		PIFF files.
		sparse turns on the sparse matrix solution (uses less memory and can be faster, but less stable)
		"""
		psf_matrix = []
		self.psf = {}
		for i in self.exposures:
			try:
				x_cen, y_cen, wcs = self.findPixelCoords(
					i["EXPNUM"],
					int(i["CCDNUM"]),
					pmc=pmc,
					return_wcs=True,
					color=self.color_bg,
				)
				self.psf[i['EXPNUM']] = piff.PSF.read(
					f"{path}/{i['EXPNUM']}/{i['EXPNUM']}_{i['CCDNUM']}_piff.fits"
				)
			except (OSError, ValueError):
				print(f"Missing {i['EXPNUM']} {i['CCDNUM']} psf")
				psf_matrix.append(sp.csr_matrix(np.zeros((size * size, len(ra_grid)))))
				continue
			psf_matrix.append(
				sp.csr_matrix(
					construct_psf_background(
						ra_grid, dec_grid, wcs, self.psf[i['EXPNUM']], x_cen, y_cen, size, flatten=True
					)
				)
			)
		print("PSF matrix")
		self.psf_matrix = sp.vstack(psf_matrix)
		del psf_matrix
		self.source_matrix = []
		self.x_cen = {}
		self.y_cen = {}
		for i in self.exposures[self.exposures["DETECTED"]]:
			self.x_cen[i['EXPNUM']], self.y_cen[i['EXPNUM']] = self.findPixelCoords(ra = self.ra + self.rarate * i['DELTA_T'], dec = self.dec + self.decrate * i['DELTA_T'], expnum = i['EXPNUM'], pmc=pmc, color=self.color[i['EXPNUM']])
			try:
				self.source_matrix.append(
					construct_psf_source(
						self.x_cen[i['EXPNUM']] + shift_x[i["EXPNUM"]],
						self.y_cen[i['EXPNUM']] + shift_y[i["EXPNUM"]],
						psf=self.psf[i["EXPNUM"]],
						stampsize=size,
						x_center=self.x_cen[i['EXPNUM']],
						y_center=self.y_cen[i['EXPNUM']],
						color = self.color[i['EXPNUM']]
					)
				)
			except (OSError):
					print(i['EXPNUM'])
					self.source_matrix.append(np.zeros((size * size)))
     
	def minimizeChisq(
		self, x_init, size=30, background=True, method="Powell"
	):
		from scipy.optimize import minimize

		self.solution = minimize(
			chi2_moving,
			x_init,
			method=method,
			args=(self, size, background),
			options={"xtol": 0.01},
		)
		x_sol = self.solution.x
		self.source_matrix = []
		j = 0
		for i in self.exposures[self.exposures["DETECTED"]]:
			try:
				self.x_cen[i['EXPNUM']], self.y_cen[i['EXPNUM']] = self.findPixelCoords(ra = x_sol[0] + x_sol[2] * i['DELTA_T'], dec = x_sol[1] + x_sol[3] *i['DELTA_T'], expnum = i['EXPNUM'], color=self.color[i['EXPNUM']])
				self.source_matrix.append(
					construct_psf_source(
						self.x_cen[i['EXPNUM']],
						self.y_cen[i['EXPNUM']],
						psf=self.psf[i["EXPNUM"]],
						stampsize=size,
						x_center=self.x_cen[i['EXPNUM']],
						y_center=self.y_cen[i['EXPNUM']],
						color = self.color[i['EXPNUM']]
					)
				)
			except (OSError):
					print(i['EXPNUM'])
					self.source_matrix.append(np.zeros((size * size)))
		self.constructDesignMatrix(size, background)
		self.solvePhotometry(True, True)


def chi2_moving(x, detection, size = 30, background = False):
	detection.source_matrix = []
	for i in detection.exposures[detection.exposures['DETECTED']]: 
		x_trial, y_trial = detection.findPixelCoords(ra = x[0] + x[2] * i['DELTA_T'], dec = x[1] + x[3] *i['DELTA_T'], expnum = i['EXPNUM'], color=detection.color[i['EXPNUM']])
		detection.source_matrix.append(construct_psf_source(x_trial, y_trial, detection.psf[i['EXPNUM']], size, detection.x_cen[i['EXPNUM']], detection.y_cen[i['EXPNUM']]))
	detection.constructDesignMatrix(size, background)
	detection.solvePhotometry(True, True)
	chisq = np.sum(detection.res * detection.res * detection.invwgt)

	return chisq 





## Moving detection only handles the case of multiple flux measurements per object, one per image
## We also need to handle the case where there is a single flux measurement for the object
## this is more optimal for the small S/N case and will help with the minimization procedure 
## to find the best-fit rates
## todo: covariance matrix for the rates?
## todo: binary class

