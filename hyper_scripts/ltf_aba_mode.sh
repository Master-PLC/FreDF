exp_name=ltf_aba_mode

gpu=0
pl=336
dst=ECL

for mdl in "iTransformer"; do
    # for lbd in 0.2 0.4 0.6 0.8 1.0; do
    #     for lr in 0.0002 0.0005 0.001; do
    #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #         save_dir="./${exp_name}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${pl}")
    #         command_args+=("mag" "phase" "mag-phase")

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_aba_mode/${mdl}_${dst}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done


    # for lr in 0.0002; do
    #     for lbd in 0.2 0.4 0.6 0.8; do
    #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #         save_dir="./${exp_name}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${pl}")
    #         if [[ $lbd == 0.2 ]] || [[ $lbd == 0.8 ]]; then
    #             command_args+=("phase")
    #         elif [[ $lbd == 0.4 ]] || [[ $lbd == 0.6 ]]; then
    #             command_args+=("mag" "mag-phase")
    #         fi

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_aba_mode/${mdl}_${dst}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done

    # for lr in 0.0005; do
    #     for lbd in 0.4; do
    #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #         save_dir="./${exp_name}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${pl}")
    #         command_args+=("mag" "mag-phase")

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_aba_mode/${mdl}_${dst}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done


    for lbd in 0.0; do
        for lr in 0.0002 0.0005 0.001; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${pl}")
            command_args+=("mag" "phase" "mag-phase")

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/ltf_aba_mode/${mdl}_${dst}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done