pwd
sudo apt-get install unzip
BASE_DIR="/home/working/"
rm -rf $BASE_DIR
mkdir $BASE_DIR/
mkdir $BASE_DIR/input/
mkdir $BASE_DIR/input/mlm/
mkdir $BASE_DIR/generated/
mkdir $BASE_DIR/weights/

cd $BASE_DIR/
git init
git pull https://www.github.com/KevinMathewT/FeedbackEPA-Kaggle
sleep 2

export KAGGLE_USERNAME=<kaggle_username>
export KAGGLE_KEY=<kaggle_key>

cd $BASE_DIR/input/
kaggle competitions download -c feedback-prize-effectiveness
unzip feedback-prize-effectiveness.zip -d . > /dev/null

mkdir $BASE_DIR/input/feedback-bertopic/
cd $BASE_DIR/input/feedback-bertopic/
# kaggle datasets download -d kevinmathewt/feedback-bertopic
# unzip feedback-bertopic.zip -d . > /dev/null 

cd $BASE_DIR/input/mlm/
# kaggle datasets download -d <mlm pretrained model saved to kaggle>
# unzip <filename>.zip -d . > /dev/null

mkdir $BASE_DIR/src/mlm/input/
mkdir $BASE_DIR/src/mlm/generated/
mkdir $BASE_DIR/src/mlm/input/feedback-prize-effectiveness
cd $BASE_DIR/src/mlm/input/feedback-prize-effectiveness
# kaggle competitions download -c feedback-prize-effectiveness
# unzip feedback-prize-effectiveness.zip -d . > /dev/null

mkdir $BASE_DIR/src/mlm/input/feedback-prize-2021
cd $BASE_DIR/src/mlm/input/feedback-prize-2021
# kaggle competitions download -c feedback-prize-2021
# unzip feedback-prize-2021.zip -d . > /dev/null

pip install --upgrade pip
pip install accelerate
pip install git+https://github.com/huggingface/transformers
pip install datasets
pip install sentencepiece
# pip install bertopic
pip install wandb
wandb login f12aade4c663d619038a864a35fbeb34dd8c5aad

cd $BASE_DIR/
git init
git pull https://www.github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.create_folds
git pull https://www.github.com/KevinMathewT/FeedbackEPA-Kaggle
accelerate launch -m src.run
ls $BASE_DIR/weights/
python3 -m setup.save_weights
