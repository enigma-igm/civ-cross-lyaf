now=`date`
echo "Start: $now"

python /mnt/quasar/xinsheng/code/compute_model_grid_CIV_lya.py \
--nproc 30 --fwhm 10.0 --samp 3.0 --SNR 50.0 --nqsos 20 --delta_z 0.8 \
--vmin 10.0 --vmax 2500.0 --dv1 10.0 --dv2 50.0 --v_end 1000.0 \
--ncovar 1000000 --nmock 300 --seed 1259761 \
--logZmin -4.5 --logZmax -2.0 --nlogZ 15 --logM_interval 0.2 --R_Mpc_interval 0.2

now=`date`
echo "Finish: $now"

# April 5, 2021:
# nohup ./igm_cluster_compute_model_grid_civ_enrichment.sh > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/igm_cluster_compute_model_grid_civ_enrichment_fine.log &

# Requested path length for nqsos=20 covering delta_z=1.000 corresponds to requested total dz_req = 20.000,
# or 92.743 total skewers. Rounding to 93 or dz_tot=20.055."

# ran in IGM on March 5, 2021:
# nohup nice -19 ./igm_cluster_compute_model_grid_civ_enrichment.sh > /mnt/quasar/xinsheng/log/igm_cluster_compute_model_grid_civ_enrichment.log &
