
exp_name=srh_lbd_lr_large
out_root="${exp_name}_others"

lr=0.0005
gpu=5


for dst in "ETTm1"; do
    for mdl in "PatchTST" "Transformer" "TimesNet" "TCN"; do
        for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336 720
        done
    done

    for mdl in "LSTM"; do
        for lbd in 0.1 0.4 0.5 0.6 0.7 0.8 0.9; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            if [[ $lbd == "0.1" ]]; then
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 192 336 720
            elif [[ $lbd == "0.4" ]]; then
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 192
            elif [[ $lbd == "0.5" ]] || [[ $lbd == "0.6" ]] || [[ $lbd == "0.9" ]]; then
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 192 336
            elif [[ $lbd == "0.7" ]]; then
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336
            else
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336 720
            fi
        done
    done
done


for dst in "ETTm2"; do
    for mdl in "PatchTST" "Transformer" "TimesNet" "TCN"; do
        for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336 720
        done
    done

    for mdl in "LSTM"; do
        for lbd in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            if [[ $lbd == "0.1" ]] || [[ $lbd == "0.2" ]] || [[ $lbd == "0.4" ]] || [[ $lbd == "0.5" ]] || [[ $lbd == "0.6" ]]; then
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336
            elif [[ $lbd == "0.3" ]]; then
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192
            else
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}" ${gpu} 96 192 336 720
            fi
        done
    done
done



# for dst in "ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Traffic" "Weather"; do
#     for mdl in "DLinear" "PatchTST" "Autoformer" "Transformer" "TimesNet" "LSTM" "TCN"; do
#         for lbd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh ${lbd} ${lr} "./${out_root}/${job_name}"
#         done
#     done
# done
