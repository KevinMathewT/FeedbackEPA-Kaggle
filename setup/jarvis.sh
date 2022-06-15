pwd
mkdir /home/input/
mkdir /home/generated/
mkdir /home/weights/
cd /home/input/
kaggle competitions download -c feedback-prize-effectiveness
sudo apt-get install unzip
unzip feedback-prize-effectiveness.zip -d .
cd /home/
git init
git pull https://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@github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.create_folds
git pull https://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@github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.run
ls /input/weights/