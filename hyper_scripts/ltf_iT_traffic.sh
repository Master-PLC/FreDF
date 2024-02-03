exp_name=srh_lbd_lr_large

gpu=0

num_proc=2

for mdl in "iTransformer"; do
    for dst in "Traffic"; do
        # for lbd in 0.85 0.87 0.89 0.91 0.93 0.95; do
        for lbd in 0.92 0.94 0.96 0.97 0.98; do
            for lr in 0.001; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${num_proc}")
                # command_args+=(96)
                command_args+=(96 192 336 720)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done


# for mdl in "iTransformer"; do
#     for dst in "ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Traffic" "Weather"; do
#         for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#             for lr in 0.0002 0.0005 0.001 0.00002 0.00005; do
#                 job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#                 bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}"
#             done
#         done
#     done
# done
