# Домашнее задание № 3
## Реализация методов оптимизации

В задании вы сами реализуете несколько алгоритмов оптимизации и примените их к определению космологических параметров Вселенной: параметра Хаббла и доли темной энергии в составе Вселенной.

Вам дан файл `jla_mub.txt`, содержащий две колонки: [красное смещение](http://www.astronet.ru/db/msg/1162269) сверхновых и модуль расстояния до них (см. [заметку](http://www.astronet.ru/db/msg/1162269) о космологическом красном смещении).
Модуль расстояния μ задается следующим соотношением:

$$
\mu\equiv 5\lg\frac{d}{\text{пк}}-5,
$$

$$
d=\frac{c}{H_0}(1+z)\int_0^z\frac{dz'}{\sqrt{(1-\Omega)(1+z')^3+\Omega}},
$$

$$
d=\frac{3\times10^{11}\text{пк}}{H_0/(\text{км}/\text{с}/\text{Мпк})}(1+z)\int_0^z\frac{dz'}{\sqrt{(1-\Omega)(1+z')^3+\Omega}},
$$

Здесь
  - $z$ — красное смещение,
  - $d$ — это фотометрическое расстояние,
  - $c$ — скорость света,
  - $H_0$ — постоянная Хаббла (обычно выражается в км/с/Мпк),
  - $\Omega$ — доля темной энергии в составе Вселененной (от 0 до 1).

> Для численного взятия интеграла рекомендуется использовать функцию `scipy.integrate.quad`. 

В астрономии расстояние часто измеряется в [парсеках](http://www.astronet.ru/db/msg/1162328) (пк), но вам ничего не нужно в них переводить, просто используйте вторую формулу для расстояния, а постоянную Хаббла подставляйте в единицах км/с/Мпк.

**Дедлайн 13 марта 2025 в 23:55**

- Вы должны реализовать следующие алгоритмы в файле `opt.py`:

  1. Метод Гаусса—Ньютона в виде функции `gauss_newton(y, f, j, x0, k=1, tol=1e-4)`, где
     - `y` — это массив с измерениями,
     - `f(*x)` — это функция от неивестных параметров, возвращающая значения, рассчитанные в соответствии с моделью, в виде одномерного массива размера `y.size`,
     - `j(*x)` — это функция от неизвестных параметров, возвращающая якобиан в виде двумерного массива `(y.size, x0.size)`,
     - `x0` — массив с начальными приближениями параметров,
     - `k` — положительное число меньше единицы, параметр метода,
     - `tol` — относительная ошибка, условие сходимости, параметр метода.

     Функция должна возвращать объект класса `Result`, реализованного в файле `opt.py`
  2. Метод Левенберга—Марквардта в фиде функции `lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4)`, где все одноимённые аргументы имеют те же значения, что и для `gauss_newton`,
     - `lmbd0` — начальное значение парметра $\lambda$ метода,
     - `nu` — мультипликатор для параметра $\lambda$.

     Функция должна возвращать объект класса `Result`

- Примените реализованные вами методы Гаусса—Ньютона и Левенберга—Марквардта к решению задачи об определении космологиических постоянных.
  Вам нужно будет восстановить параметры модели, используя ваши оптимизаторы, в файле `cosmology.py`:
  - Загрузить данные из файла `jla_mub.txt`
  - Восстановить параметры, используя ваши оптимизаторы, подберите параметры оптимизаторов, которые хорошо подходят для данной задачи.
    В качестве начального приближения используйте значения $H_0$ = 50, $\Omega$ = 0.5.
  - Оцените ошиби определения параметров в приближении малых ошибок: для этого удобно воспользоваться ранее написанной функцией вычисления якобиана.
  - Нанести точки из файла на график зависимости μ от z. Там же построить модельную кривую, используя найденные параметры. Сохраните график в файл `mu-z.png`
  - На другом графике построить зависимость функции потерь `sum(0.5 * (y-f)^2)` от итерационного шага для обоих алгоритмов. Сохраните график в файл `cost.png`
  - Используйте технику *bootstrap*, чтобы проверить насколько правильно работает формула оценки ошибок в приближении малых ошибок. Для этого сгенерируйте 10000 подвыборок с повторами такого же размера, как исходные данные, для каждой подвыборки оцените значения параметров $H_0$ и $\Omega$, а затем рассчитайте их выборочные стандартные отклонения. Для ускорения работы кода разумно использовать ранее определенные значения параметров в качестве начального приближения для каждой подвыборки.
  - Выведите в файл `parameters.json` результаты в виде (значения даны для примера формата):

  ```json
  {
    "Gauss-Newton": {
      "H0": 55,
      "H0_err": 2,
      "H0_err_bootstrap": 3,
      "Omega": 0.55,
      "Omega_err": 0.1,
      "Omega_err_bootstrap": 0.05,
      "nfev": 155
    },
    "Levenberg-Marquardt": {
      "H0": 45,
      "H0_err": 3,
      "H0_err_bootstrap": 4,
      "Omega": 0.45,
      "Omega_err": 0.2,
      "Omega_err_bootstrap": 0.15,
      "nfev": 145
    }
  }
  ```

**В этом задании запрещено пользоваться готовыми пакетами для оптимизации, в том числе `scipy.optimize`.**
