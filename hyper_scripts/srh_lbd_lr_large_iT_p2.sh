exp_name=srh_lbd_lr_large

gpu=1

for mdl in "iTransformer"; do
    # for dst in "ETTh1"; do
    #     for lbd in 0.6; do
    #         for lr in 0.0005; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}" ${gpu} 
    #         done
    #     done


    #     for lbd in 0.0; do
    #         for lr in 0.00002; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}" ${gpu} 96
    #         done
    #     done
    # done



    # for dst in "ETTm1"; do
    #     for lbd in 0.9; do
    #         for lr in 0.00005; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}" ${gpu} 192
    #         done
    #     done
    # done



    # for dst in "Weather"; do
    #     for lbd in 1.0; do
    #         for lr in 0.0005; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}" ${gpu} 192
    #         done
    #     done

    #     for lbd in 0.6; do
    #         for lr in 0.001; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}" ${gpu} 720
    #         done
    #     done

    #     for lbd in 1.0; do
    #         for lr in 0.00002 0.00005; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_iT_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${exp_name}/${job_name}" ${gpu} 192 720
    #         done
    #     done
    # done

    for dst in "Traffic"; do
        # for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        # for lbd in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        for lbd in 0.7 0.9; do
            for lr in 0.0005; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")

                if [[ $lbd == "0.9" ]]; then
                    command_args+=(336)
                else
                    command_args+=(336 720)
                fi
                
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
