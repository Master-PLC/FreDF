
exp_name=srh_lbd_lr_large
out_root="${exp_name}_others"

lr=0.0005
gpu=7


for dst in "Weather"; do
    for mdl in "Autoformer"; do
        for lbd in 0.5 0.6 0.9 1.0; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${out_root}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            command_args+=(720)

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done


    # for mdl in "Transformer" "TimesNet" "LSTM" "TCN" "PatchTST"; do
    #     for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #         bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336 720
    #     done
    # done
done


# for dst in "ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Traffic" "Weather"; do
#     for mdl in "DLinear" "PatchTST" "Autoformer" "Transformer" "TimesNet" "LSTM" "TCN"; do
#         for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}"
#         done
#     done
# done
