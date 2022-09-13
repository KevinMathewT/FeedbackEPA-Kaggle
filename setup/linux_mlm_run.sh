BASE_DIR="/home/working/"
cd $BASE_DIR/
git init
git pull https://www.github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.mlm.mlm_no_trainer
ls $BASE_DIR/weights/
python3 -m setup.save_weights
