import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

os.system("cls")

data = pd.DataFrame({"message": [], "class": []})

# _____________این فانکشن تمامی فایل های کوجود را استخرااج میکند و هر خط از فایل  ها را در ارایه اضافه میکند
def read_files(directorypath, classification):
    data = []
    index = []
    for direc in os.listdir(directorypath):
        path = directorypath + "/" + direc
        index.append(path)
        file = open(path, "r", encoding="utf8", errors="ignore")
        in_body = False
        for line in file.readlines():
            if in_body:
                if line != None and line != "\n":
                    data.append({"message": line, "class": classification})
            elif line == "\n":
                in_body = True
        file.close()
    return pd.DataFrame(data)

# ________ایمیل های spam شده را در یه ارایه میریزد
spam_data = read_files("./emails/spam", "spam")

# ________ایمیل های اسپم نشده را در یک ارایه میریزد
not_spam_data = read_files("./emails/not_spam/", "not_spam")
data = pd.concat([spam_data, not_spam_data])
vectorizer = CountVectorizer()

# _____این تابع به ما میگویید هر کلمه چند بار تکرار شده و کلمات مهم را پیدا میکند
count = vectorizer.fit_transform(data["message"].values)
target = data["class"].values

# ______در اینجا مدل را اموزش میدهیم و پارامتر دوم لیبل های ما است که دو مورد بیشتر نیست اسپم و نات اسپم
classifire = MultinomialNB()
classifire.fit(count, target)

# در اینجا دیتای ازمایشی درست میکنیم تا مدل را تست کنیم
example = ["buy free", "how are you"]
example_count = vectorizer.transform(example)

# در اینجا مدل را ازمایش میکنیم
predict = classifire.predict(example_count)

print(predict)
