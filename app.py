import spacy
from flask import Flask, render_template, request, session
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
app.secret_key = "your_secret_key" 

# spaCyモデルのロード
nlp = spacy.load("ja_core_news_trf")

# ランダムに選択する文の数
num_questions = 5

# CSVファイルから英文と和訳のペアを読み込む
translations_df = pd.read_csv("translations.csv", encoding="shift_jis")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_answers = []

        selected_indices = session.get("selected_indices", [])

        # ユーザーが入力した和訳を取得
        for i in range(num_questions):
            user_answers.append(request.form.get(f"user_answer_{i}"))

        # 正解数のカウント
        correct_count = 0
        incorrect_questions = []
        incorrent_user_answers=[]
        
        # 正解かどうかを判定
        for i in range(num_questions):
            selected_question = translations_df.loc[selected_indices[i]]
            original_translation = selected_question["translation"]
            user_input_translation = user_answers[i]

            texts = [original_translation, user_input_translation]

            vectorizer = CountVectorizer(tokenizer=lambda text: [token.text for token in nlp(text)], token_pattern=None)
            X = vectorizer.fit_transform(texts)

            similarity_score = cosine_similarity(X)[0, 1]

            threshold = 0.7

            if similarity_score > threshold:
                correct_count += 1
            else:
                incorrect_questions.append(selected_question)
                incorrent_user_answers.append(user_input_translation)

        return render_template("result.html", correct_count=correct_count, num_questions=num_questions,incorrect_questions=incorrect_questions,user_answers=incorrent_user_answers)
    
    # ランダムに問題を選択
    selected_indices = random.sample(range(len(translations_df)), num_questions)
    session["selected_indices"] = selected_indices  # セッションに選択されたインデックスを保存
    selected_questions = translations_df.loc[selected_indices]
    

    return render_template("index.html", selected_questions=selected_questions)

if __name__ == "__main__":
    app.run(debug=True)
