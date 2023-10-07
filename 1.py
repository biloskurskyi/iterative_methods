#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sympy import *  # для похідної, синуси та конуси. Імпортимо з модуля всі функції які є і можемо не вказувати звідки
import numpy as np  # nympy скорочуємо до np
import math

# In[2]:


import math


def ctg(x):  # функція для котангенса
    return cos(x) / sin(x)


# In[3]:


x = Symbol('x')  # щоб замість х були числа і можна було рахувати
fun = x ** 2 - 64  # наше рівняння


# In[4]:


def funvalueinx(fun, xnow):  # значення функції в х(xnow це цей конретний графік)
    fun = lambdify(x, fun)  # перетворення str в int(не можна підставити замість х щось)
    return fun(xnow)  # значення функції при певному х, ми його задаємо


# In[5]:


def fundiff(fun, number):  # похідна від функції. number - якого порядку похідна
    while number > 0:
        fun = fun.diff(x)  # diff робить похідну, з бібліотеки sympy
        number -= 1
    return fun


# In[6]:


def condition1(a, b, fun):  # умова перша
    result = True  # якщо result не зміниться, значить умова виконується
    for i in np.arange(a, b + 1, 0.01):  # можемо працювати з числами з плаваючою крапкою
        if funvalueinx(fun, b) * funvalueinx(fundiff(fun, 2), i) <= 0:  # Якщо f(b)*f’’(x)>0 на [a, b], то (x0=a)
            result = False
            break
    return result


# In[7]:


def condition2(a, b, fun):  # умова друга
    result = True  # якщо result не зміниться, значить умова виконується
    for i in np.arange(a, b + 1, 0.01):  # можемо працювати з числами з плаваючою крапкою
        if funvalueinx(fun, a) * funvalueinx(fundiff(fun, 2), i) <= 0:  # Якщо f(a)*f’’(x)>0 на [a, b], то (x0=b)
            result = False
            break
    return result


# In[8]:


def сhordfun1(b, x_n, fun):  # якщо умова перша правдива, то виконується(для методу хорд)
    return x_n - (funvalueinx(fun, x_n) / (funvalueinx(fun, b) - funvalueinx(fun, x_n)) * (
            b - x_n))  # щоб не писати кожного разу


# lambdify ми одразу передбачили це у функції зверху і тепер без проблем можемо розвязувати рівняння


# In[9]:


def сhordfun2(a, x_n, fun):  # якщо умова друга правдива, то виконується(для методу хорд)
    return a - (funvalueinx(fun, a) / (funvalueinx(fun, x_n) - funvalueinx(fun, a)) * (x_n - a))


# In[10]:


def tangentfun(x_n, fun):  # якщо умова правдива, то виконується(для методу дотичних)
    return x_n - (funvalueinx(fun, x_n) / funvalueinx(fundiff(fun, 1), x_n))


# In[11]:


def calculate(a, b, fun, eps, method):  # основна функція
    counter1 = 0  # кількістьть ітерацій

    xn = 0  # Хn
    xn1 = 0  # Xn+1

    # вибираємо метод
    if method == 'Chord':
        if condition1(a, b,
                      fun):  # перевіряє чи виконується умова, якщо ні то виведе 0(приклад нище) або перевірить іншу умову, в іншому разі обчислює
            xn = a
            xn1 = сhordfun1(b, xn, fun)
            counter1 += 1
            while abs(xn1 - xn) >= eps:  # умова зупинка циклу 
                xn = xn1
                xn1 = сhordfun1(b, xn, fun)
                counter1 += 1
            return xn1, counter1

        elif condition2(a, b, fun):
            xn = b
            xn1 = сhordfun2(a, xn, fun)
            counter1 += 1
            while abs(xn1 - xn) >= eps:
                xn = xn1
                xn1 = сhordfun2(a, xn, fun)
                counter += 1
            return xn1, counter1

    elif method == 'Tangent':
        if condition1(a, b, fun):
            xn = b
            xn1 = tangentfun(xn, fun)
            counter1 += 1
            while abs(xn1 - xn) >= eps:
                xn = xn1
                xn1 = tangentfun(xn, fun)
                counter1 += 1
            return xn1, counter1

        elif condition2(a, b, fun):
            xn = a
            xn1 = tangentfun(xn, fun)
            counter1 += 1
            while abs(xn1 - xn) >= eps:
                xn = xn1
                xn1 = tangentfun(xn, fun)
                counter1 += 1
            return xn1, counter1

    elif method == 'Combination':
        counter2 = 0  # к-ть ітерацій
        _xn = 0  # -Xn
        _xn1 = 0  # -Xn+1

        if condition1(a, b, fun):
            xn = b
            _xn = a

            xn1 = tangentfun(xn, fun)
            _xn1 = сhordfun1(b, _xn, fun)

            counter1 += 1
            counter2 += 1
            while abs(xn1 - xn) >= eps:
                xn = xn1
                xn1 = tangentfun(xn, fun)
                counter1 += 1
            while abs(_xn1 - _xn) >= eps:
                _xn = _xn1
                _xn1 = сhordfun1(b, _xn, fun)
                counter2 += 1
            return xn1, counter1, _xn1, counter2

        elif condition2(a, b, fun):
            xn = a
            _xn = b

            xn1 = tangentfun(xn, fun)
            _xn1 = сhordfun2(a, _xn, fun)

            counter1 += 1
            counter2 += 1
            while abs(xn1 - xn) >= eps:
                xn = xn1
                xn1 = tangentfun(xn, fun)
                counter1 += 1
            while abs(_xn1 - _xn) >= eps:
                _xn1 = сhordfun2(a, _xn, fun)
                counter2 += 1
            return xn1, counter1, _xn1, counter2
    else:
        return 'The data is not entered correctly'
    return 0


# In[12]:


# print(calculate(0.1, 0.2, fun, 0.001, 'Chord'))
print(calculate(7, 9, fun, 0.001, 'Tangent'))
# print(calculate(0.1, 0.2, fun, 0.001, 'Combination'))
# print(calculate(0.1, 0.2, fun, 0.001, 'Other method'))
# print(calculate(1, 2, fun, 0.001, 'Chord'))


# In[13]:


from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt  # набір команд який дозволяє нам будувати графік

x = Symbol('x')
fun = ctg(x) - x / 3


def schedule(a, b, fun):  # функція для графіка
    x = np.linspace(a,
                    b)  # повертає одномірний масив із заданої кількості елементів, значення яких рівномірно розподілені всередині заданого інтервалу
    y = funvalueinx(fun, x)

    plt.plot(x, y)  # створює граік
    plt.show()  # показує графік


schedule(-3, 3, fun)  # наш графік

# In[14]:


import numpy as np


def function2(both):  # функція яка зберігає нашу систему функцій
    x, y = both
    return np.array([np.sin(x + 0.5) - y - 1, np.cos(y - 2) + x])


# In[15]:


import numpy as np


def Derivativefx(x, y):  # похідна від нашої першої ф-ції
    return np.cos(x + 1 / 2) - 1


# In[16]:


import numpy as np


def Derivativefy(x, y):  # похідна від нашої другої ф-ції
    return -(np.sin(y - 2)) + 1


# In[17]:


import numpy as np


def Newton(both, epsilon):  # both -(x,y)
    x, y = function2(both)  # получаємо нові х та у
    guesses = np.array([both])  # масив масивів

    while abs(arr[-1][0] - arr[-2][0]) >= epsilon or abs(arr[-1][1] - arr[-2][1]) >= epsilon:

        fx, fy = function2(np.array([x, y]))  # знаходимо нове значення х та у
        Dfx = Derivativefx(x, y)  # dfx - обчислена похідна першого рівняння
        Dfy = Derivativefy(x, y)  # dfy - обчислена похідна другого(точка в похідній)
        if Dfx == 0 or Dfy == 0:
            print('There is no solution')
            return None
        x = x - fx / Dfx
        y = y - fy / Dfy
        guesses = np.append(guesses, [[x, y]], axis=0)

    return guesses


# In[18]:


guesses = Newton([1, 1], 0.001)  # [1,1] - початкове припущення, 
print(guesses)

# In[19]:


import numpy as np

x = np.arange(-10, 10.01, 0.01)
y = np.arange(-10, 10.01, 0.01)
plt.figure(figsize=(10, 5))
plt.plot(np.sin(x + 0.5) - y - 1)
plt.plot(np.cos(y - 2) + x)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$f(x)$', fontsize=14)
plt.grid(True)
plt.show()
