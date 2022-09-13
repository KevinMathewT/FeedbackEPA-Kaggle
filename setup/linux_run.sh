BASE_DIR="/home/working/"
cd $BASE_DIR/
git init
git pull https://www.github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.create_folds
git pull https://www.github.com/KevinMathewT/FeedbackEPA-Kaggle
accelerate launch -m src.run
ls $BASE_DIR/weights/
python3 -m setup.save_weights
