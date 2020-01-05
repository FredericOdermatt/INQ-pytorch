    nlp_architect train transformer_glue \
        --task_name cola \
        --model_name_or_path bert-base-uncased \
        --model_type quant_bert \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --output_dir ~/deeplearning/tmp/cola-8bit_v2 \
        --evaluate_during_training \
        --data_dir ~/deeplearning/glue_data/CoLA \
        --do_lower_case
