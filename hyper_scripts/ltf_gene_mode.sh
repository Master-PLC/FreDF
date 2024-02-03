exp_name=ltf_gene_mode

gpu=1

for mdl in "iTransformer"; do
    # for dst in "ECL" "Traffic" "Weather"; do
    #     if [[ $dst == "ECL" ]]; then
    #         lr=0.001
    #         lbd=0.1
    #     elif [[ $dst == "Traffic" ]]; then
    #         lr=0.001
    #         lbd=0.9
    #     elif [[ $dst == "Weather" ]]; then
    #         lr=0.0005
    #         lbd=0.4
    #     fi

    #     for auxi_mode in "rfft-D" "rfft-2D"; do
    #         job_name=${exp_name}_dst_${dst}_am_${auxi_mode}
    #         save_dir="./${exp_name}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
    #         command_args+=(96 192 336 720)

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done

    
    # for dst in "ECL" "Traffic" "Weather"; do
    #     if [[ $dst == "ECL" ]]; then
    #         lr=0.001
    #         lbd=0.1
    #     elif [[ $dst == "Traffic" ]]; then
    #         lr=0.001
    #         lbd=0.9
    #     elif [[ $dst == "Weather" ]]; then
    #         lr=0.0005
    #         lbd=0.4
    #     fi

    #     for auxi_mode in "rfft-D" "rfft-2D"; do
    #         job_name=${exp_name}_dst_${dst}_am_${auxi_mode}
    #         save_dir="./${exp_name}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
    #         if [[ $auxi_mode == "rfft-D" ]]; then
    #             command_args+=(96 192)
    #         else
    #             command_args+=(96 720)
    #         fi

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done


    for dst in "Traffic"; do
        if [[ $dst == "ECL" ]]; then
            lr=0.001
            lbd=0.1
        elif [[ $dst == "Traffic" ]]; then
            lr=0.001
            lbd=0.9
        elif [[ $dst == "Weather" ]]; then
            lr=0.0005
            lbd=0.4
        fi

        for auxi_mode in "rfft-2D"; do
            job_name=${exp_name}_dst_${dst}_am_${auxi_mode}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
            command_args+=(192)

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done