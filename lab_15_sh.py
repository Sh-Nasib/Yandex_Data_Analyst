import numpy as np, array
import pandas as pd
from IPython.display import display
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

#создаю файл Лаборатроная работа 15
file=open('Лабораторная работа15.csv','w')
file.close()

#данные таблицы
data = [['Year','Sales'],
        [2011,13],
        [2012,15],
        [2013,19],
        [2014,21],
        [2015,27],
        [2016,35],
        [2017,47],
        [2018,49],
        [2019,57]]

#записваю данные таблицы в файл
with open('Лабораторная работа15.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in data:
        csvwriter.writerow(row)

#присваиваю данные переменной         
data = pd.read_csv('Лабораторная работа15.csv')
#вывожу данные в виде таблицы

print('Исходные данные')
display( data.head(10))


#создаю график данных 
#задаю параметры для графика, модуль matplotlib 
plt.figure(figsize = (12,5))
#расписываю график
plt.title('Year&sales')
sns.lineplot( x = 'Year',
              y = 'Sales',
              data = data, marker = 'o',
              label = 'Sales result')
#нижняя подпись графика
plt.xlabel ('Year')
#лист.позиций по ОСИ Х
x_ticks = [2011,2012,2013,2014,2015,2016,2017,2018,2019]
#подпись лист.позиций по ОСИ Х
x_labels = ['2011','2012','2013','2014','2015',
            '2016','2017','2018','2019']
#присваивание граффику инф о ОСИ Х
plt.xticks ( ticks = x_ticks , labels = x_labels )
#присваивание граффику инф о ОСИ Y
plt.ylabel ('Sales')
plt.show()


''' Поиск скользящего срендего '''


#добовляем столбец прогноз и результат прогнозирования
#минамльна 3 позция, так как ранее не хватает данных
data ['Predict'] = data.Sales.rolling(3).mean()
predictSales = data.Sales.rolling(3).mean() 
#вывожу уже новыую таблицу с прогнозом
print ('\n','График скользящего среднего')
display(data.head(10))

#задаю параметры для графика, модуль matplotlib 
plt.figure(figsize = (12,5))
#расписываю график
plt.title('Predict.Year&sales')
sns.lineplot( x = 'Year',
              y = 'Sales',
              data = data,marker = 'o',
              label = 'Sales result')
#накладываю на график прогноз
sns.lineplot( x = 'Year',
              y = 'Predict',
              data = data,marker = 'o',
              label = 'Predict result')
#нижняя подпись графика
plt.xlabel ('Year')
#лист.позиций по ОСИ Х
x = data.values[0::1, 0]
y = data.values[0::1, 1]
x_ticks = x #[2011,2012,2013,2014,2015,2016,2017,2018,2019]
#подпись лист.позиций по ОСИ Х
x_labels = ['2011','2012','2013','2014','2015',
            '2016','2017','2018','2019']
#присваивание граффику инф о ОСИ Х
plt.xticks ( ticks = x_ticks , labels = x_labels )
#присваивание граффику инф о ОСИ Y
plt.ylabel ('Sales')
#вывожу график
plt.show()


''' Прогнозирование с помощью функции регерессии '''

print ('\n','Прогноз с помощью функции регрессии')

data = pd.read_csv( 'Лабораторная работа15.csv' , sep=","  ,
                    skiprows = 1, header = None)
x = data.values[0::1, 0]
y = data.values[0::1, 1]
x= x.reshape(-1, 1)
y= y.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
data1 = model.predict(np.array([x[-1] + 1, x[-1] + 2]).reshape(-1, 1))
x = data.values[0::1, 0]
y = data.values[0::1, 1]
next_year = x [-1] + [[1],[2]]
display(data1)

plt.figure(figsize = (12,5))
plt.plot(x,y,color = 'green',marker = 'o', label = 'Sales')
plt.plot(next_year, data1,color = 'red', marker = 'o',label='Predict')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Refression_Predict.Year&sales')
plt.legend()
plt.show()


''' Прогнозирование с помощью линии тренда '''

print ('\n','Прогноз с помощью линии тренда')

data = pd.read_csv( 'Лабораторная работа15.csv' , sep=","  ,
                    skiprows = 1, header = None)
x = data.values[0::1, 0]
y = data.values[0::1, 1]
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
trend_line = p(x)
display (trend_line)

plt.figure(figsize = (12,5))
plt.title('Predict_trend.Year&sales')
plt.plot(x, y, marker = 'o', label='Sales')
plt.plot(x, trend_line, marker = 'o', label='Trend line')
plt.xlabel('Year')
plt.ylabel('Sales')
next_year = x[-1] + 1
forecast = p(next_year)
plt.plot(next_year, forecast, marker = 'o',
         label=f'Predict {next_year}')
plt.legend()
plt.show()

