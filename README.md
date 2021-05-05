# pstage_03_KLUE_Relation_extraction

### 프로젝트 Wrap up report 여기서 확인가능 합니다.
* https://www.notion.so/P2-Relation-Extraction-Wrap-up-Report-f367c0a5ad0b483e8defdc9206e26e0f

### training
* python train_roberta.py --name=[training_name] --random_seed=[seed]

### training K-fold 
* python train_roberta_kfold.py --name=[training_name] --random_seed=[seed]

### inference
* python inference_roberta.py --model_dir=[model_path]
* ex) python inference_roberta.py --model_dir=./results/checkpoint-500

