exp_name=ltf_gene_mode_supp

gpu=1

for mdl in "iTransformer"; do
    # for dst in "ECL" "Traffic" "Weather"; do
    #     if [[ $dst == "ECL" ]]; then
    #         lr=0.001
    #         lbd=0.1
    #     elif [[ $dst == "Traffic" ]]; then
    #         lr=0.001
    #         lbd=0.93
    #     elif [[ $dst == "Weather" ]]; then
    #         lr=0.0005
    #         lbd=0.4
    #     fi

    #     for auxi_mode in "rfft-2D"; do
    #         job_name=${exp_name}_dst_${dst}_am_${auxi_mode}
    #         save_dir="./${exp_name}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
    #         # command_args+=(96 192 336 720)
    #         command_args+=(96)

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done

    # for dst in "Traffic" "Weather"; do
    #     if [[ $dst == "Traffic" ]]; then
    #         lr=0.001
    #         lbd_list=(0.5 0.6 0.7 0.8)
    #     elif [[ $dst == "Weather" ]]; then
    #         lr=0.0005
    #         lbd_list=(0.1 0.2 0.3)
    #     fi

    #     for lbd in ${lbd_list[@]}; do
    #         for auxi_mode in "rfft-2D"; do
    #             job_name=${exp_name}_dst_${dst}_am_${auxi_mode}_lbd_${lbd}_lr_${lr}
    #             save_dir="./${exp_name}/${job_name}"
    #             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
    #             # command_args+=(96 192 336 720)
    #             command_args+=(96 192 336 720)

    #             mkdir -p "${save_dir}/"
    #             (
    #                 echo "Running command for $job_name"
    #                 bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
    #             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #         done
    #     done
    # done

    for dst in "Weather"; do
        if [[ $dst == "Traffic" ]]; then
            lr=0.001
            lbd_list=(0.91 0.92 0.93)
        elif [[ $dst == "Weather" ]]; then
            lr=0.0005
            # lbd_list=(0.5 0.6 0.7)
            lbd_list=(0.6)
        fi

        for lbd in ${lbd_list[@]}; do
            for auxi_mode in "rfft-2D"; do
                job_name=${exp_name}_dst_${dst}_am_${auxi_mode}_lbd_${lbd}_lr_${lr}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
                command_args+=(720)
                # command_args+=(192)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done



# for mdl in "iTransformer"; do
#     for dst in "ETTm1" "ETTh1"; do
#         if [[ $dst == "ETTm1" ]]; then
#             lr=0.0005
#             lbd_list=
#         elif [[ $dst == "ETTh1" ]]; then
#             lr=0.001
#             lbd=0.2
#         fi

#         for auxi_mode in "rfft-D" "rfft-2D"; do
#             job_name=${exp_name}_dst_${dst}_am_${auxi_mode}
#             save_dir="./${exp_name}/${job_name}"
#             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${auxi_mode}")
#             # command_args+=(96 192 336 720)
#             command_args+=(96)

#             mkdir -p "${save_dir}/"
#             (
#                 echo "Running command for $job_name"
#                 bash ./scripts/ICML2024/ltf_gene_mode/${dst}_script/${mdl}.sh "${command_args[@]}"
#             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#         done
#     done
# done