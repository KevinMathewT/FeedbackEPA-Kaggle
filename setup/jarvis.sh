pwd
sudo apt-get install unzip
rm -rf /home/working/
mkdir /home/working/input/
mkdir /home/working/generated/
mkdir /home/working/weights/

cd /home/working/
git init
git pull https://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@github.com/KevinMathewT/FeedbackEPA-Kaggle
cp /home/working/setup/kaggle.json /home/.kaggle/kaggle.json

cd /home/working/input/
kaggle competitions download -c feedback-prize-effectiveness
unzip feedback-prize-effectiveness.zip -d .

cd /home/working/
git init
git pull https://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@github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.create_folds
git pull https://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@github.com/KevinMathewT/FeedbackEPA-Kaggle
python3 -m src.run
ls /input/weights/