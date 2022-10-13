#%% 
import numpy as np
from astroquery.gaia import Gaia
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

'''mlp format'''
plt.rc('font', family='serif')
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'legend.labelspacing':0.25, 'legend.fontsize': 12})
mpl.rcParams.update({'errorbar.capsize': 4})

'''GAIA query'''
def get_GAIA(ra, dec, angle):
    Gaia.ROW_LIMIT = -1

    ra = ra
    dec = dec
    angl = angle
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    query = Gaia.launch_job_async  ("SELECT * "
                                "FROM gaiadr3.gaia_source "
                                "WHERE parallax_over_error > 5 and ruwe < 1.4 and phot_g_mean_mag < 17 and 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS', "+str(ra)+" , " + str(dec) + " , " + str(angl) + "))"
                                "ORDER by source_id")

    return query.get_results()

#np.save('gaia_request', get_GAIA(272.9, -18.53932068, 2))
#np.save('gaia_request2', get_GAIA(266.4167, -29.0078, 2))



#%% 

'''Create dataframes'''

gaia_request = np.load('gaia_request.npy', allow_pickle=True)

data = pd.DataFrame(gaia_request,
                    columns=['ra','dec','parallax','pmra','pmdec'])

hr_data = pd.DataFrame(gaia_request,
                    columns=['bp_rp','phot_g_mean_mag'])


'''pre-processing'''

data = data.dropna()

'''Plot requested region'''
plt.figure(figsize=(10,8))
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.hexbin(data['ra'], data['dec'])
plt.xlabel('RA (deg)', fontdict={'fontsize': 20})
plt.ylabel('DEC (deg)', fontdict={'fontsize': 20})
#plt.title('Density of stars (Region 1)', fontdict={'fontsize': 22})
plt.show()

#%% 
'''Plot BP-RP Spectra'''

plt.scatter(hr_data['bp_rp'], hr_data['phot_g_mean_mag'], s=4, alpha=(0.3))
plt.title('HR Diagram')
plt.xlabel('Bp - Rp (mag)')
plt.ylabel('g (mag)')
plt.ylim(np.max(hr_data['phot_g_mean_mag']), np.min(hr_data['phot_g_mean_mag']))
plt.show()

