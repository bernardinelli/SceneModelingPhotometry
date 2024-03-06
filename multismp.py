from smp import *


class MultiDetection(Detection):
	"""
	Main class for SMP. Requires RA and Dec for the detection, an exposure and CCD numbers for bookkeeping and
	zero-point retrieval, a band (for finding extra exposures) and an optional color (for astrometry) and name for the
	detection
	"""

	def __init__(self, ra, dec, expnum, ccdnum, band, color=0.61, name="", color_bg = 0.61):
		"""
		Constructor class
		"""
		self.ra = ra
		self.dec = dec
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
		ccdlist['BAND'] = exposures['BAND']
		ccdlist.sort("EXPNUM")

		ccdlist["DETECTED"] = False
		ccdlist["DETECTED"][np.isin(ccdlist["EXPNUM"], detected)] = True

		self.exposures = tb.unique(ccdlist)

	def findAllExposures(self, survey, detected, return_list=False, reduce_band=True):
		"""
		Requires a list of DECamExposures or DESExposures from `DESTNOSIM`,
		returns all exposures that touch the point.
		If return_list == True, returns this as a list, otherwise saves this
		inside det.exposures
		If reduce_band == True, drops all exposures from different bands (that is,
		keeps only the same band as the detection)
		"""
		from pixmappy import Gnomonic

		x, y = Gnomonic(self.ra, self.dec).toXY(
			np.array(survey.ra), np.array(survey.dec)
		)
		# note that, at RA0, DEC0, x,y = 0
		dist = np.sqrt(x**2 + y**2)

		close = np.where(dist < 1.5)
		ra_arr = np.array([self.ra])
		dec_arr = np.array([self.dec])

		ccdlist = tb.Table(
			names=("EXPNUM", "CCDNUM", "BAND"), dtype=("i8", "i4", "str")
		)
		ccdlist.add_row([self.expnum, self.ccdnum, self.band])

		for i, j in zip(survey.expnum[close], survey.band[close]):
			ccd = survey[i].checkInCCDFast(ra_arr, dec_arr, ccdsize=0.149931)[1]
			if ccd != None:
				ccdlist.add_row([i, ccd, j])
		ccdlist.sort("EXPNUM")

		ccdlist["DETECTED"] = False
		ccdlist["DETECTED"][np.isin(ccdlist["EXPNUM"], detected)] = True

		self.exposures = tb.unique(ccdlist)

		if reduce_band:
			self.exposures = self.exposures[self.exposures["BAND"] == self.band]

		self.exposures.sort("DETECTED")

		if return_list:
			return self.exposures

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
			self.x_cen[i['EXPNUM']], self.y_cen[i['EXPNUM']] = self.findPixelCoords(expnum = i['EXPNUM'],pmc=pmc, color=self.color[i['EXPNUM']])
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


	def constructDesignMatrix(self, size, background=True):
		"""
		Constructs the design matrix for the solution.
		size is the stamp size, sparse turns on the sparse solution
		background defines whether the background is being fit together with the image or not
		"""
		if not background:
			ones = np.ones((size * size, 1))
		else:
			ones = np.zeros((size * size, 1))

		print("Background")
		background = sp.block_diag(len(self.exposures) * [ones])
	
		self.ntot = len(self.exposures)
		self.ndet = len(self.exposures[self.exposures["DETECTED"]])
		psf_zeros = np.zeros((self.psf_matrix.shape[0], self.ndet))
		for i in range(self.ndet):
			psf_zeros[
				(self.ntot - self.ndet + i) * size * size : (self.ntot - self.ndet + i + 1) * size * size, (self.ntot - self.ndet) + i
			] = self.source_matrix[i]

		print("Design")
		self.design = sp.hstack(
			[self.psf_matrix, background, psf_zeros], dtype="float64"
		)

	def solvePhotometry(self, res=True, err=True):
		"""
		Solves the system for the flux as well as background sources
		Solution is saved in det.X, the flux is the -1 entry in this array
		- res: defines if the residuals should be computed
		- err: defines if the errors should be computed (requires an expensive matrix inversion)
		- sparse: turns on sparse routines. Less stable, possibly incompatible with `err`
		"""
		diag = sp.diags(np.sqrt(self.invwgt))
		print("Product")

		prod = diag.dot(self.design)
		print("Solving")
		self.X = sp.linalg.lsqr(prod, self.image * np.sqrt(self.invwgt))[0]
		print("Solved")
		self.flux = self.X[-self.ndet:]

		self.mag = -2.5 * np.log10(self.flux) + 30

		if res:
			self.pred = self.design @ self.X
			self.res = self.pred - self.image

		if err:
			inv_cov = self.design.T @ np.diag(self.invwgt) @ self.design
			try:
				self.cov = np.linalg.inv(inv_cov)
			except LinAlgError:
				self.cov = np.linalg.pinv(inv_cov)

			self.sigma_flux = np.sqrt(self.cov[-self.ndet:, -self.ndet:])
			self.sigma_mag = (
				2.5 * np.sqrt(self.cov[-1, -1] / (self.flux**2)) / np.log(10)
			)

	def runPhotometry(
		self,
		se_path,
		piff_path,
		zp,
		survey,
		detected,
		pmc=DESMaps(),
		n_grid=20,
		size=30,
		offset_x=0,
		offset_y=0,
		err=True,
		res=True,
		background=False,
	):
		"""
		Convenience function that performs all operations required by the photometry
		- se_path: path for the SE postage stamps
		- piff_path: path for the PIFF files
		- zp: zeropoint dictionary
		- survey: `DESTNOSIM` list of exposures
		- pmc: pixmappy instance for astrometry
		- n_grid: grid size for point sources in the background (adds n_grid x n_grid sources)
		- size: stamp size
		- offset_x,y: offset in the x and y pixel coordinates
		- sparse: sparse routines
		- err: turns on error estimation
		- res: computes residuals
		- background: background estimation
		"""
		self.findAllExposures(survey, detected)

		ra_grid, dec_grid = local_grid(
			self.ra,
			self.dec,
			0.35 / 3600,
			n_grid,
		)

		self.constructImages(zp, se_path, size=size, background=background)

		self.constructPSFs(
			ra_grid, dec_grid, pmc, size, offset_x, offset_y, piff_path)

		self.constructDesignMatrix(size, background=background)
		self.solvePhotometry(err=err, res=res)

	def photometryShotNoise(self, stampsize, gain_dict):
		"""
		Adds in shot noise estimates from a previous fit
		"""

		if self.flux > 0:
			## fight gain
			gain_cut = gain(
				self.expnum, self.ccdnum, self.x, self.y, stampsize, gain_dict
			)
			gain_cut /= self.zp

			sigma_photon = self.pred[-stampsize * stampsize :] / gain_cut.flatten()
			sigma_photon[sigma_photon < 0] = 0
			sigma_photon[np.isnan(sigma_photon)] = 0
			sigma_photon[np.isinf(sigma_photon)] = 0
			## update weights
			self.wgt_shotnoise = np.copy(self.wgt)

			self.wgt_shotnoise[-stampsize * stampsize :] += sigma_photon

			self.invwgt_shotnoise = 1 / self.wgt_shotnoise
			self.invwgt_shotnoise[self.wgt_shotnoise == 0] = 0

			self.invwgt_shotnoise[self.wgt_shotnoise < 0] = 0

			self.invwgt_shotnoise[np.isnan(self.invwgt_shotnoise)] = 0
			self.invwgt_shotnoise[np.isinf(self.invwgt_shotnoise)] = 0

			# self.design[np.isnan(self.design)] = 0
			# self.design[np.isinf(self.design)] = 0

			# self.image[np.isnan(self.image)] = 0
			# self.image[np.isinf(self.image)] = 0

			## redo photometry

			self.X_shotnoise = lstsq(
				np.diag(np.sqrt(self.invwgt_shotnoise)) @ self.design,
				self.image * np.sqrt(self.invwgt_shotnoise),
			)[0]

			self.flux_shotnoise = self.X_shotnoise[-1]

			inv_cov = self.design.T @ np.diag(self.invwgt_shotnoise) @ self.design
			try:
				self.cov_shotnoise = np.linalg.inv(inv_cov)
			except LinAlgError:
				self.cov_shotnoise = np.linalg.pinv(inv_cov)
			self.sigma_flux_shotnoise = np.sqrt(self.cov_shotnoise[-1, -1])

		else:
			self.flux_shotnoise = self.flux
			self.sigma_flux_shotnoise = self.sigma_flux
			self.X_shotnoise = self.X
			self.cov_shotnoise = self.cov

		self.pred_shotnoise = self.design @ self.X_shotnoise

		self.mag_shotnoise = -2.5 * np.log10(self.flux_shotnoise) + 30
		self.sigma_mag_shotnoise = (
			2.5
			* self.sigma_flux_shotnoise
			/ np.sqrt((self.flux_shotnoise**2))
			/ np.log(10)
		)

	def minimizeChisq(
		self, x_init, size=30, background=True, method="Powell"
	):
		from scipy.optimize import minimize

		self.solution = minimize(
			chi2_multi,
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
				self.source_matrix.append(construct_psf_source(
						self.x_cen[i['EXPNUM']] + x_sol[j],
						self.y_cen[i['EXPNUM']] + x_sol[j+1],
						psf=self.psf[i["EXPNUM"]],
						stampsize=size,
						x_center=self.x_cen[i['EXPNUM']],
						y_center=self.y_cen[i['EXPNUM']],
						color = self.color[i['EXPNUM']]
					))
				j += 2 #this keeps track of indexing of x_sol
				
			except (OSError):
					print(i['EXPNUM'])
					self.source_matrix.append(np.zeros((size * size)))
		self.constructDesignMatrix(size, background)
		self.solvePhotometry(True, True)

def chi2_multi(x, detection, size = 30, background = False):
	j = 0	
	detection.source_matrix = []
	for i in detection.exposures[detection.exposures['DETECTED']]: 
		detection.source_matrix.append(construct_psf_source(detection.x_cen[i['EXPNUM']] + x[j], detection.y_cen[i['EXPNUM']] + x[j+1], detection.psf[i['EXPNUM']], size, detection.x_cen[i['EXPNUM']], detection.y_cen[i['EXPNUM']]))
		j += 2
	detection.constructDesignMatrix(size, background)
	detection.solvePhotometry(True, True)
	chisq = np.sum(detection.res * detection.res * detection.invwgt)

	return chisq 

