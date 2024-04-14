import joblib
from flask import Flask
from flask import request
import numpy as np

app = Flask(__name__)
MODEL_PATH = "project.joblib"

COLUMNS = ["Тип-жилья", "Кол-во-комнат", "Станция-метро", "Время-в-пути","Этаж","Общая-площадь-комнат","Наличие-ремонта","Тип-дома","Год"]

with open(MODEL_PATH, "rb") as fid:
    model = joblib.load(fid)


@app.route('/')
def index():
    return """<html>
    <head>Price prediction</head>
        <body>
            <form method="post" action="generate">
                <label for="Тип-жилья">Тип жилья:</label>
                <input type="text" value="Вторичка" name="Тип-жилья" /> <br>
                
                <label for="Кол-во-комнат">Кол-во комнат:</label>
                <input type="text" value="5-комн" name="Кол-во-комнат" /> <br>
                
                <label for="Станция-метро">Станция метро:</label>
                <input type="text" value="Парк культуры" name="Станция-метро" /><br>

                <label for="Время-в-пути">Время в пути(пешком):</label>
                <input type="text" value="2.0" name="Время-в-пути" /><br>

                <label for="Этаж">Этаж:</label>
                <input type="text" value="2" name="Этаж" /><br>

                <label for="Общая-площадь-комнат">Общая площадь комнат:</label>
                <input type="text" value="242.0" name="Общая-площадь-комнат" /><br>

                <label for="Наличие-ремонта">Наличие ремонта:</label>
                <input type="text" value="Дизайнерский" name="Наличие-ремонта" /><br>

                <label for="Тип-дома">Тип дома:</label>
                <input type="text" value="Кирпичный" name="Тип-дома" /><br>

                <label for="Год">Год:</label>
                <input type="text" value="1900" name="Год" /><br>

                <button type="submit">Predict!</button>
            </form>
        </body>
    </html>"""


@app.route('/generate', methods=['POST'])
def generate():
    data = [
        request.values[column]
        for column in COLUMNS
    ]
    X = np.array(list([data])).reshape(1, -1)
    Xdiskret=np.hstack([X[:,2:6],X[:,8:9]])
    Encoder=np.hstack([X[:,0:2],X[:,6:8]])
    
    with open("enc.joblib", "rb") as file:
    	enc = joblib.load(file)
    Encoders=enc.transform(Encoder)
    Xnew=np.hstack([Xdiskret,Encoders])

    zeros_station=['Курская', 'Театральная', 'Крестьянская застава', 'Смоленская', 'Марксистская', 'Проспект мира', 'Площадь революции', 'Динамо', 'Библиотека им. ленина', 'Третьяковская', 'Чеховская', 'Чистые пруды', 'Фрунзенская', 'Новокузнецкая', 'Охотный ряд', 'Китай-город', 'Трубная', 'Боровицкая', 'Павелецкая', 'Цветной бульвар', 'Парк культуры', 'Цска', 'Александровский сад', 'Кропоткинская', 'Хорошевская', 'Добрынинская', 'Тургеневская', 'Киевская', 'Новослободская', 'Пушкинская', 'Петровский парк', 'Таганская', 'Чкаловская', 'Беговая', 'Октябрьская', 'Кузнецкий мост', 'Красносельская', 'Сухаревская', 'Тверская', 'Полянка', 'Арбатская', 'Серпуховская', 'Краснопресненская', 'Маяковская', 'Лубянка', 'Баррикадная', 'Комсомольская', 'Красные ворота', 'Белорусская', 'Сретенский бульвар']
    first_station=['Римская', 'Ломоносовский проспект', 'Дмитровская', 'Бульвар рокоссовского', 'Полежаевская', 'Кожуховская', 'Алексеевская', 'Спартак', 'Преображенская площадь', 'Воробьевы горы', 'Раменки', 'Ботанический сад', 'Студенческая', 'Панфиловская', 'Балтийская', 'Выставочная', 'Крымская', 'Шаболовская', 'Рижская', 'Сокольники', 'Вднх', 'Шоссе энтузиастов', 'Авиамоторная', 'Фонвизинская', 'Лужники', 'Шелепиха', 'Улица 1905 года', 'Сходненская', 'Спортивная', 'Бауманская', 'Парк победы', 'Коптево', 'Зил', 'Кутузовская', 'Ленинский проспект', 'Локомотив', 'Площадь гагарина', 'Сокол', 'Угрешская', 'Деловой центр', 'Минская', 'Петровско-разумовская', 'Зорге', 'Октябрьское поле', 'Черкизовская', 'Стрешнево', 'Тимирязевская', 'Савеловская', 'Лихоборы', 'Бутырская', 'Международная', 'Автозаводская', 'Щукинская', 'Тушинская', 'Достоевская', 'Менделеевская', 'Дубровка', 'Аэропорт', 'Хорошёво', 'Войковская', 'Тульская', 'Водный стадион', 'Ростокино', 'Марьина роща', 'Окружная', 'Белокаменная']
    metro=Xdiskret[:,1]
    for i in metro: key=lambda i: i.capitalize()
    zero_condition = np.isin(metro, first_station)
    first_condition = np.isin(metro, zeros_station)

    metro[first_condition] =1
    metro[zero_condition] =0
    metro[~(first_condition | zero_condition )] = 2
    metro =metro.astype(int)
    Xnew[:, 0] = metro


    with open("standartd_scaler2.joblib", "rb") as file:
    	standartd_scaler2 = joblib.load(file) 
    Xnew_st=standartd_scaler2.transform(Xnew)

    with open("standartd_scaler3.joblib", "rb") as file:
    	standartd_scaler3 = joblib.load(file)

    predicted_label = model.predict(Xnew_st)
    predicted = standartd_scaler3.inverse_transform(predicted_label)
   
    return f"""<html>
    <head></head>
        <body>
            Predicted data, rub: {predicted[0]}
            <br>
            <a href=/> Back </a>
        </body>
    </html>"""
