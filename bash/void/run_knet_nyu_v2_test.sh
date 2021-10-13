export CUDA_VISIBLE_DEVICES=1

python src/run_knet.py \
--image_path testing/nyu_v2/nyu_v2_test_image_corner.txt \
--sparse_depth_path testing/nyu_v2/nyu_v2_test_sparse_depth_corner.txt \
--intrinsics_path testing/nyu_v2/nyu_v2_test_intrinsics_corner.txt \
--ground_truth_path testing/nyu_v2_test_ground_truth_corner.txt \
--normalized_image_range 0 1 \
--min_pool_sizes_sparse_to_dense_pool 15 17 \
--max_pool_sizes_sparse_to_dense_pool 23 27 29 \
--avg_pool_sizes_sparse_to_dense_pool 0 \
--n_convolution_sparse_to_dense_pool 3  \
--n_filter_sparse_to_dense_pool 8 \
--encoder_type knet_v1 fusion_conv_previous sparse_to_dense_pool_v1 \
--input_type sparse_depth validity_map \
--input_channels_image 3 3 3 3 0 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--n_resolutions_encoder_intrinsics 0 1 2 3 \
--skip_types image depth \
--decoder_type multi-scale \
--n_filters_decoder 256 128 128 64 12 \
--deconv_type up \
--output_kernel_size 3 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--outlier_removal_method remove \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--save_outputs \
--depth_model_restore_path \
trained_knet_v1/void1500/convprev1proj1_poolv1_c3f8_min1517_max232729_spp0_outk3_co015_st085_sz200_sm200_wp1e8_lr0-1e4_10-5e5_15_remove30-60_noise1e2_pose-res18_cali0123/depth_model-67000.pth \
--output_path \
trained_knet_v1/void1500/convprev1proj1_poolv1_c3f8_min1517_max232729_spp0_outk3_co015_st085_sz200_sm200_wp1e8_lr0-1e4_10-5e5_15_remove30-60_noise1e2_pose-res18_cali0123/evaluation_results/nyu_v2 \
--device gpu
