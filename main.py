if False:
    import nilearn
    from nilearn import datasets, plotting

    # Allen RSN networks
    allen = datasets.fetch_atlas_allen_2011()

    data_xyzt = nilearn.image.load_img(allen['maps']).get_fdata()
    print(data_xyzt.shape)
    cluster_xyzp = nilearn.image.load_img(allen['rsn28']).get_fdata()
    print(cluster_xyzp.shape)

    print(1)
    # "contours" example
    cluster_xyzp_pred = (cluster_xyzp - cluster_xyzp.min()) / (cluster_xyzp.max() - cluster_xyzp.min())
    new_niimg_with_pred = nilearn.image.new_img_like(nilearn.image.load_img(allen['rsn28']), cluster_xyzp_pred)
    plotting.plot_prob_atlas(new_niimg_with_pred, title="Allen2011", view_type= "continuous", threshold= cluster_xyzp_pred.mean() + (cluster_xyzp_pred.max() - cluster_xyzp_pred.mean()) * 0.4)
    plotting.show()

import nilearn
from nilearn import datasets, plotting
import gacc
from time import time
import numpy as np

ctime = time()
# Allen RSN networks
allen = datasets.fetch_atlas_allen_2011()
data_xyzt = nilearn.image.load_img(allen['maps']).get_fdata()
cluster_xyzp = nilearn.image.load_img(allen['rsn28']).get_fdata()
num_clusters_gt = cluster_xyzp.shape[3]
num_clusters_pred = 50
cluster_threshold = cluster_xyzp.mean() + (cluster_xyzp.max() - cluster_xyzp.mean()) * 0.4

if False:
    cluster_xyzp_full = np.zeros(list(cluster_xyzp.shape[:3]) + [num_clusters_pred])
    cluster_xyzp_full[:, :, :, :num_clusters_gt] = cluster_xyzp
    new_niimg_with_pred = nilearn.image.new_img_like(nilearn.image.load_img(allen['rsn28']), cluster_xyzp_full) ## https://neurostars.org/t/how-to-create-a-niftimage-object/22719/2
    display = plotting.plot_prob_atlas(new_niimg_with_pred, title="Allen2011_input", view_type= "continuous", threshold= cluster_threshold)
    display.savefig('./out/Allen2011_input.png')
    display.close()

cluster_obj = gacc.GACC(num_clusters= num_clusters_pred, run_eagerly= True, min_max_scale_range= [0., 1.]) ## change run_eagerly= True or False vice versa to debug in both sides.
cluster_obj.fit(data_xyzt= data_xyzt, cluster_xyzp= cluster_xyzp, num_batches= 10, epochs= 40)
cluster_xyzp_pred = cluster_obj.predict(data_xyzt= data_xyzt)
cluster_xyzp_pred[:, :, :, :num_clusters_gt] = np.where(cluster_xyzp >= cluster_threshold, cluster_xyzp, cluster_xyzp_pred[:, :, :, :num_clusters_gt])

if True:
    new_niimg_with_pred = nilearn.image.new_img_like(nilearn.image.load_img(allen['rsn28']), cluster_xyzp_pred) ## https://neurostars.org/t/how-to-create-a-niftimage-object/22719/2
    display = plotting.plot_prob_atlas(new_niimg_with_pred, title="Allen2011_GACC", view_type= "continuous", threshold= cluster_threshold)
    display.savefig('./out/Allen2011_GACC.png')
    display.close()
    # plotting.plot_prob_atlas(new_niimg_with_pred, title="Allen2011_pred", view_type= "continuous", threshold= cluster_threshold)
    # plotting.show()

print(f"Clustering took {time() - ctime} secs.")