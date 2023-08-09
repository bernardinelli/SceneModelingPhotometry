import numpy as np 
import astropy.table as tb 



def make_mugshots(detection_list, size, filename, shotnoise = False):
	'''
		Makes "mugshots" of a list of detections. Requires Gary's image_tools
		Sometimes this doesn't work as well as it should....
	'''
	from image_tools import clippedMean as cM
	import matplotlib.pyplot as pl 
	import os 

	for i in detection_list:
		image = i.image[-size*size:].reshape((size,size))
		sigma = np.sqrt(cM(image, 4)[1])
		if not shotnoise:
			source = i.flux * i.psf_source
		else:
			source = i.flux_shotnoise * i.psf_source
		source = source.reshape((size, size))

		try:
			if not shotnoise:
				model = i.pred[-size*size:].reshape((size,size))
			else:
				model = i.pred_shotnoise[-size*size:].reshape((size,size))
		except:
			if not shotnoise:
				m = i.design @ i.X
			else:
				m = i.design @ i.X_shotnoise
			model = m[-size*size:].reshape((size,size))

		try:
			i.sigma_flux
		except:
			i.solvePhotometry(False, True)


		pl.subplot(1,4,1)
		pl.imshow(image, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		pl.subplot(1,4,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		

		pl.subplot(1,4,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		
		pl.subplot(1,4,3)
		pl.imshow(image - model + source, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		
		pl.subplot(1,4,4)
		pl.imshow(image - model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.text
		pl.axis('off')
		
		#pl.suptitle(f'{i.expnum} {i.band} flux: {i.flux:.2f} +- {i.sigma_flux:.2f}')
		pl.tight_layout()


		pl.savefig(filename + f'_{i.expnum}_{i.ccdnum}.png', bbox_inches = 'tight')

		pl.close()


def make_mugshots_binary(detection_list, size, filename, shotnoise = False):
	'''
		Makes "mugshots" of a list of detections. Requires Gary's image_tools
		Sometimes this doesn't work as well as it should....
	'''
	from image_tools import clippedMean as cM
	import matplotlib.pyplot as pl 
	import os 

	for i in detection_list:
		image = i.image[-size*size:].reshape((size,size))
		sigma = np.sqrt(cM(image, 4)[1])
		if not shotnoise:
			source = i.flux_primary * i.psf_primary + i.flux_secondary * i.psf_secondary
		else:
			source = i.flux_primary_shotnoise * i.psf_primary + i.flux_secondary_shotnoise * i.psf_secondary
		source = source.reshape((size, size))

		try:
			if not shotnoise:
				model = i.pred[-size*size:].reshape((size,size))
			else:
				model = i.pred_shotnoise[-size*size:].reshape((size,size))
		except:
			if not shotnoise:
				m = i.design @ i.X
			else:
				m = i.design @ i.X_shotnoise
			model = m[-size*size:].reshape((size,size))

		try:
			i.sigma_flux
		except:
			i.solvePhotometry(False, True)


		pl.subplot(1,4,1)
		pl.imshow(image, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		pl.subplot(1,4,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		

		pl.subplot(1,4,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		
		pl.subplot(1,4,3)
		pl.imshow(image - model + source, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		
		pl.subplot(1,4,4)
		pl.imshow(image - model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.text
		pl.axis('off')
		
		#pl.suptitle(f'{i.expnum} {i.band} flux: {i.flux:.2f} +- {i.sigma_flux:.2f}')
		pl.tight_layout()


		pl.savefig(filename + f'_{i.expnum}_{i.ccdnum}.png', bbox_inches = 'tight')

		pl.close()


def make_mugshots_fivepanel(detection_list, size, filename, shotnoise = False):
	'''
		Makes "mugshots" of a list of detections. Requires Gary's image_tools
		Sometimes this doesn't work as well as it should....
	'''
	from image_tools import clippedMean as cM
	import matplotlib.pyplot as pl 
	import os 

	for i in detection_list:
		image = i.image[-size*size:].reshape((size,size))
		sigma = np.sqrt(cM(image, 4)[1])
		if not shotnoise:
			source = i.flux * i.psf_source
		else:
			source = i.flux_shotnoise * i.psf_source
		source = source.reshape((size, size))

		try:
			if not shotnoise:
				model = i.pred[-size*size:].reshape((size,size))
			else:
				model = i.pred_shotnoise[-size*size:].reshape((size,size))
		except:
			if not shotnoise:
				m = i.design @ i.X
			else:
				m = i.design @ i.X_shotnoise
			model = m[-size*size:].reshape((size,size))

		try:
			i.sigma_flux
		except:
			i.solvePhotometry(False, True)


		pl.subplot(1,5,1)
		pl.imshow(image, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		pl.subplot(1,5,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		

		pl.subplot(1,5,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		
		pl.subplot(1,5,3)
		pl.imshow(image - model + source, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		pl.subplot(1,5,4)
		pl.imshow(image - source, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		
		pl.subplot(1,5,5)
		pl.imshow(image - model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.text
		pl.axis('off')
		
		#pl.suptitle(f'{i.expnum} {i.band} flux: {i.flux:.2f} +- {i.sigma_flux:.2f}')
		pl.tight_layout()


		pl.savefig(filename + f'_{i.expnum}_{i.ccdnum}.png', bbox_inches = 'tight')

		pl.close()


def make_mugshots_binary_fivepanel(detection_list, size, filename, shotnoise = False):
	'''
		Makes "mugshots" of a list of detections. Requires Gary's image_tools
		Sometimes this doesn't work as well as it should....
	'''
	from image_tools import clippedMean as cM
	import matplotlib.pyplot as pl 
	import os 

	for i in detection_list:
		image = i.image[-size*size:].reshape((size,size))
		sigma = np.sqrt(cM(image, 4)[1])
		if not shotnoise:
			source = i.flux_primary * i.psf_primary + i.flux_secondary * i.psf_secondary
		else:
			source = i.flux_primary_shotnoise * i.psf_primary + i.flux_secondary_shotnoise * i.psf_secondary
		source = source.reshape((size, size))

		try:
			if not shotnoise:
				model = i.pred[-size*size:].reshape((size,size))
			else:
				model = i.pred_shotnoise[-size*size:].reshape((size,size))
		except:
			if not shotnoise:
				m = i.design @ i.X
			else:
				m = i.design @ i.X_shotnoise
			model = m[-size*size:].reshape((size,size))

		try:
			i.sigma_flux
		except:
			i.solvePhotometry(False, True)


		pl.subplot(1,5,1)
		pl.imshow(image, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		pl.subplot(1,5,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		

		pl.subplot(1,5,2)
		pl.imshow(model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')
		
		pl.subplot(1,5,3)
		pl.imshow(image - model + source, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		pl.subplot(1,5,4)
		pl.imshow(image - source, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.axis('off')

		
		pl.subplot(1,5,5)
		pl.imshow(image - model, vmin=-3*sigma, vmax = 4*sigma, cmap='gray_r', origin='lower')
		pl.text
		pl.axis('off')
		
		#pl.suptitle(f'{i.expnum} {i.band} flux: {i.flux:.2f} +- {i.sigma_flux:.2f}')
		pl.tight_layout()


		pl.savefig(filename + f'_{i.expnum}_{i.ccdnum}.png', bbox_inches = 'tight')

		pl.close()



def generate_table(detection_list):
	'''
		Goes from a list of detections to a summary table with magnitudes, exposures, fluxes, errors, etc
	'''
	exp = []
	flux = []
	flux_err = []
	mag = []
	mag_err = []
	band = []

	for i in detection_list:
		exp.append(i.expnum)
		flux.append(i.flux)
		flux_err.append(i.sigma_flux)
		mag.append(i.mag)
		mag_err.append(i.sigma_mag)
		band.append(i.band)

	table = tb.Table()
	table['EXPNUM'] = exp
	table['FLUX'] = flux 
	table['FLUX_ERR'] = flux_err
	table['MAG'] = mag
	table['MAG_ERR'] = mag_err
	table['BAND'] = band
	return table

def generate_table_shotnoise(detection_list):
	'''
		Goes from a list of detections to a summary table with magnitudes, exposures, fluxes, errors, etc
	'''
	exp = []
	band = []
	flux = []
	flux_err = []
	mag = []
	mag_err = []
	flux_sn = [] 
	flux_err_sn = [] 
	mag_sn = [] 
	mag_err_sn = [] 


	for det in detection_list:
		#det = Detection.read(i)
		exp.append(det.expnum)
		flux.append(det.flux)
		flux_err.append(det.sigma_flux)
		mag.append(det.mag)
		mag_err.append(det.sigma_mag)
		band.append(det.band)
		flux_sn.append(det.flux_shotnoise)
		flux_err_sn.append(det.sigma_flux_shotnoise)
		mag_sn.append(det.mag_shotnoise)
		mag_err_sn.append(det.sigma_mag_shotnoise)

	table = tb.Table()
	table['EXPNUM'] = exp
	table['FLUX'] = flux 
	table['FLUX_ERR'] = flux_err

	table['MAG'] = mag
	table['MAG_ERR'] = mag_err
	table['BAND'] = band

	table['FLUX_SN'] = flux_sn 
	table['FLUX_ERR_SN'] = flux_err_sn
	table['MAG_SN'] = mag_sn
	table['MAG_ERR_SN'] = mag_err_sn

	return table


def generate_table_binary(detection_list):
	'''
		Goes from a list of detections to a summary table with magnitudes, exposures, fluxes, errors, etc
	'''
	exp = []
	band = []
	flux = []
	flux_err = []
	mag = []
	mag_err = []
	flux_sn = [] 
	flux_err_sn = [] 
	mag_sn = [] 
	mag_err_sn = [] 

	flux_s = []
	flux_err_s = []
	mag_s = []
	mag_err_s = []
	flux_sn_s = [] 
	flux_err_sn_s = [] 
	mag_sn_s = [] 
	mag_err_sn_s = [] 

	corr = []
	corr_sn = []

	sol = []


	for det in detection_list:
		#det = Detection.read(i)
		exp.append(det.expnum)
		flux.append(det.flux_primary)
		flux_err.append(det.sigma_flux_primary)
		mag.append(det.mag_primary)
		mag_err.append(det.sigma_mag_primary)
		band.append(det.band)
		flux_sn.append(det.flux_primary_shotnoise)
		flux_err_sn.append(det.sigma_flux_primary_shotnoise)
		mag_sn.append(det.mag_primary_shotnoise)
		mag_err_sn.append(det.sigma_mag_primary_shotnoise)

		flux_s.append(det.flux_secondary)
		flux_err_s.append(det.sigma_flux_secondary)
		mag_s.append(det.mag_secondary)
		mag_err_s.append(det.sigma_mag_secondary)
		flux_sn_s.append(det.flux_secondary_shotnoise)
		flux_err_sn_s.append(det.sigma_flux_secondary_shotnoise)
		mag_sn_s.append(det.mag_secondary_shotnoise)
		mag_err_sn_s.append(det.sigma_mag_secondary_shotnoise)
		corr.append(det.cov[-1,-2])
		corr_sn.append(det.cov_shotnoise[-1,-2])
		sol.append(det.solution.x)

	table = tb.Table()
	table['EXPNUM'] = exp
	table['BAND'] = band

	table['FLUX_PRIMARY'] = flux 
	table['FLUX_ERR_PRIMARY'] = flux_err

	table['MAG_PRIMARY'] = mag
	table['MAG_ERR_PRIMARY'] = mag_err

	table['FLUX_SECONDARY'] = flux_s 
	table['FLUX_ERR_SECONDARY'] = flux_err_s 

	table['MAG_SECONDARY'] = mag_s
	table['MAG_ERR_SECONDARY'] = mag_err_s

	table['FLUX_SN_PRIMARY'] = flux_sn 
	table['FLUX_ERR_SN_PRIMARY'] = flux_err_sn
	table['MAG_SN_PRIMARY'] = mag_sn
	table['MAG_ERR_SN_PRIMARY'] = mag_err_sn

	table['FLUX_SN_SECONDARY'] = flux_sn_s
	table['FLUX_ERR_SN_SECONDARY'] = flux_err_sn_s
	table['MAG_SN_SECONDARY'] = mag_sn_s
	table['MAG_ERR_SN_SECONDARY'] = mag_err_sn_s

	table['COV_PRIMARY_SECONDARY'] = corr 
	table['COV_SN_PRIMARY_SECONDARY'] = corr_sn
	table['SOL'] = sol 


	return table
