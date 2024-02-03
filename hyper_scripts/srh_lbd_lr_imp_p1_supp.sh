exp_name=srh_lbd_lr_imp

gpu=2

for mdl in "iTransformer"; do
    # for dst in "ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Weather"; do
    #     for lbd in 0.0 0.2 0.4 0.6 0.8 1.0; do
    #         for lr in 0.0002 0.0005 0.001; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             save_dir="./${exp_name}/${job_name}"
    #             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
    #             command_args+=(0.125 0.25 0.375 0.5)
    #             # if [[ $lbd == 0.0 ]] && [[ $lr == 0.0002 ]]; then
    #             #     command_args+=(0.375 0.5)
    #             # else
    #             #     command_args+=(0.125 0.25 0.375 0.5)
    #             # fi

    #             mkdir -p "${save_dir}/"
    #             (
    #                 echo "Running command for $job_name"
    #                 bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
    #             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #         done
    #     done
    # done

    for dst in "ECL"; do
        for lr in 0.0002; do
            for lbd in 0.4 1.0; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
                command_args+=(0.25)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done

        
        # for lr in 0.0005; do
        #     for lbd in 1.0; do
        #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
        #         save_dir="./${exp_name}/${job_name}"
        #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
        #         command_args+=(0.125 0.375)

        #         mkdir -p "${save_dir}/"
        #         (
        #             echo "Running command for $job_name"
        #             bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
        #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        #     done
        # done

        # for lr in 0.001; do
        #     for lbd in 0.2 0.8; do
        #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
        #         save_dir="./${exp_name}/${job_name}"
        #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")

        #         if [[ $lbd == 0.2 ]]; then
        #             command_args+=(0.25)
        #         else
        #             command_args+=(0.5)
        #         fi

        #         mkdir -p "${save_dir}/"
        #         (
        #             echo "Running command for $job_name"
        #             bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
        #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        #     done
        # done
    done



    # for dst in "ETTm2"; do
    #     for lr in 0.001; do
    #         for lbd in 0.0; do
    #             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #             save_dir="./${exp_name}/${job_name}"
    #             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
    #             command_args+=(0.375 0.5)

    #             mkdir -p "${save_dir}/"
    #             (
    #                 echo "Running command for $job_name"
    #                 bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
    #             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #         done
    #     done
    # done
done