"""
Улучшенная система сбора экономических данных для TAA (Tactical Asset Allocation)
Версия 2.0 - с исправлением проблем квартальных данных и многими улучшениями
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
import os
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Настройка параметров
START_DATE = '1973-01-01'  # Оптимальная точка - полные данные для большинства индикаторов
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')

# Создаем директории для вывода, если не существуют
os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)


class ImprovedEconomicDataCollector:
    """
    Улучшенный класс для сбора экономических данных.
    Исправлены проблемы с квартальными данными и добавлены новые функции.
    """

    def __init__(self, api_key: str, start_date: str = START_DATE, end_date: str = END_DATE):
        """
        Инициализация с API ключом как параметром для безопасности
        """
        self.start_date = start_date
        self.end_date = end_date
        self.fred = Fred(api_key=api_key)
        self.data_quality_report = {}  # Для отчета о качестве данных

        # Расширенный словарь индикаторов с дополнительными параметрами
        self.indicators = {
            # ИНДИКАТОРЫ ЭКОНОМИЧЕСКОГО РОСТА
            'growth_indicators': {
                'GDPC1': {
                    'name': 'Real GDP',
                    'description': 'Реальный ВВП США (с поправкой на инфляцию)',
                    'frequency': 'quarterly',
                    'transform': 'yoy_growth',
                    'publication_lag': 30,  # Публикуется с задержкой ~30 дней
                    'start_date_override': None  # Используем общую дату начала
                },
                'INDPRO': {
                    'name': 'Industrial Production Index',
                    'description': 'Индекс промышленного производства',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 15
                },
                'PAYEMS': {
                    'name': 'Nonfarm Payrolls',
                    'description': 'Занятость в несельскохозяйственном секторе',
                    'frequency': 'monthly',
                    'transform': 'mom_change',
                    'publication_lag': 5
                },
                'UNRATE': {
                    'name': 'Unemployment Rate',
                    'description': 'Уровень безработицы',
                    'frequency': 'monthly',
                    'transform': 'level',
                    'publication_lag': 5
                },
                'ICSA': {
                    'name': 'Initial Jobless Claims',
                    'description': 'Первичные заявки на пособие по безработице',
                    'frequency': 'weekly',
                    'transform': '4week_ma',
                    'publication_lag': 5
                },
                'HOUST': {
                    'name': 'Housing Starts',
                    'description': 'Начало строительства жилья',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 18
                },
                'RSXFS': {
                    'name': 'Retail Sales Ex Food Services',
                    'description': 'Розничные продажи без учета питания',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 14
                }
            },

            # ИНДИКАТОРЫ ИНФЛЯЦИИ
            'inflation_indicators': {
                'CPILFESL': {
                    'name': 'Core CPI',
                    'description': 'Базовая инфляция (без еды и энергии)',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 12
                },
                'CPIAUCSL': {
                    'name': 'CPI All Items',
                    'description': 'Общий индекс потребительских цен',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 12
                },
                'PPIACO': {
                    'name': 'PPI All Commodities',
                    'description': 'Индекс цен производителей',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 14
                },
                'DCOILWTICO': {
                    'name': 'WTI Oil Price',
                    'description': 'Цена нефти WTI',
                    'frequency': 'daily',
                    'transform': 'monthly_avg_yoy',
                    'publication_lag': 1
                },
                'T5YIE': {
                    'name': '5-Year Breakeven Inflation',
                    'description': '5-летние инфляционные ожидания',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '2003-01-01'  # Данные доступны с 2003
                },
                'T10YIE': {
                    'name': '10-Year Breakeven Inflation',
                    'description': '10-летние инфляционные ожидания',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '2003-01-01'
                }
            },

            # МОНЕТАРНЫЕ И ФИНАНСОВЫЕ ИНДИКАТОРЫ
            'monetary_indicators': {
                'DFF': {
                    'name': 'Federal Funds Rate',
                    'description': 'Ставка ФРС',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'DGS10': {
                    'name': '10-Year Treasury Rate',
                    'description': 'Доходность 10-летних облигаций',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'DGS2': {
                    'name': '2-Year Treasury Rate',
                    'description': 'Доходность 2-летних облигаций',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'T10Y2Y': {
                    'name': '10Y-2Y Spread',
                    'description': 'Спред доходности (кривая доходности)',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'M2SL': {
                    'name': 'M2 Money Supply',
                    'description': 'Денежная масса M2',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 14
                }
            },

            # ИНДИКАТОРЫ РЫНОЧНОГО СТРЕССА И НАСТРОЕНИЙ
            'market_indicators': {
                'VIXCLS': {
                    'name': 'VIX',
                    'description': 'Индекс волатильности (страха)',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '1990-01-01'  # VIX начинается с 1990
                },
                'DEXUSEU': {
                    'name': 'USD/EUR Exchange Rate',
                    'description': 'Курс доллара к евро',
                    'frequency': 'daily',
                    'transform': 'monthly_avg_yoy',
                    'publication_lag': 1,
                    'start_date_override': '1999-01-01'  # EUR появился в 1999
                },
                'BAMLH0A0HYM2': {
                    'name': 'High Yield Spread',
                    'description': 'Спред высокодоходных облигаций',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '1996-12-31'
                },
                'UMCSENT': {
                    'name': 'Consumer Sentiment',
                    'description': 'Потребительские настроения',
                    'frequency': 'monthly',
                    'transform': 'level',
                    'publication_lag': 0
                }
            },

            # ОПЕРЕЖАЮЩИЕ ИНДИКАТОРЫ
            'leading_indicators': {
                'AWHMAN': {
                    'name': 'Average Weekly Hours Manufacturing',
                    'description': 'Средняя рабочая неделя в производстве',
                    'frequency': 'monthly',
                    'transform': 'yoy_change',
                    'publication_lag': 5
                },
                'NEWORDER': {
                    'name': 'Manufacturers New Orders',
                    'description': 'Новые заказы производителей',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 35
                },
                'PERMIT': {
                    'name': 'Building Permits',
                    'description': 'Разрешения на строительство',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 18
                }
            }
        }

    def fetch_single_indicator(self, series_id: str, indicator_info: Dict) -> Optional[pd.DataFrame]:
        """
        Улучшенная загрузка индикатора с обработкой особых случаев
        """
        try:
            print(f"Загружаю {indicator_info['name']} ({series_id})...")

            # Определяем дату начала для конкретного индикатора
            start_date = indicator_info.get('start_date_override', self.start_date)

            # Получаем сырые данные
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=self.end_date
            )

            if data is None or len(data) == 0:
                print(f"  ⚠️  Нет данных для {series_id}")
                return None

            # Преобразуем в DataFrame
            df = pd.DataFrame(data, columns=[series_id])
            df.index = pd.to_datetime(df.index)

            # Сохраняем информацию о качестве данных
            self.data_quality_report[series_id] = {
                'name': indicator_info['name'],
                'raw_count': len(df),
                'first_date': df.index[0],
                'last_date': df.index[-1],
                'frequency': indicator_info['frequency']
            }

            # Применяем трансформации
            df_transformed = self.apply_transformation(
                df,
                series_id,
                indicator_info['transform'],
                indicator_info['frequency']
            )

            print(f"  ✓ Загружено: {len(df_transformed)} наблюдений")

            return df_transformed

        except Exception as e:
            print(f"  ✗ Ошибка при загрузке {series_id}: {str(e)}")
            return None

    def apply_transformation(self, df: pd.DataFrame, series_id: str,
                           transform_type: str, frequency: str) -> pd.DataFrame:
        """
        Улучшенное применение трансформаций с учетом частоты данных
        """
        # Сначала применяем базовые трансформации
        if transform_type == 'level':
            result = df

        elif transform_type == 'yoy_growth':
            if frequency == 'quarterly':
                # Для квартальных данных - рост к соответствующему кварталу прошлого года
                df[f'{series_id}_YOY'] = df[series_id].pct_change(periods=4) * 100
            else:
                # Для месячных данных - к соответствующему месяцу
                df[f'{series_id}_YOY'] = df[series_id].pct_change(periods=12) * 100
            result = df[[f'{series_id}_YOY']]

        elif transform_type == 'mom_change':
            df[f'{series_id}_MOM'] = df[series_id].diff()
            result = df[[f'{series_id}_MOM']]

        elif transform_type == '4week_ma':
            # Применяем 4-недельную скользящую среднюю
            df_ma = df.copy()
            df_ma[f'{series_id}_4WMA'] = df[series_id].rolling(window=4, min_periods=1).mean()
            # Для недельных данных сначала делаем ресемпл к месячной частоте
            # берем среднее значение за месяц
            df_monthly = df_ma[[f'{series_id}_4WMA']].resample('M').mean()
            result = df_monthly

        elif transform_type == 'monthly_avg':
            df_monthly = df.resample('M').mean()
            df_monthly.columns = [f'{series_id}_MAVG']
            result = df_monthly

        elif transform_type == 'monthly_avg_yoy':
            df_monthly = df.resample('M').mean()
            df_monthly[f'{series_id}_MAVG_YOY'] = df_monthly[series_id].pct_change(periods=12) * 100
            result = df_monthly[[f'{series_id}_MAVG_YOY']]

        elif transform_type == 'yoy_change':
            df[f'{series_id}_YOY_CHG'] = df[series_id].diff(periods=12)
            result = df[[f'{series_id}_YOY_CHG']]

        else:
            result = df

        return result

    def standardize_frequency_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Продвинутая стандартизация частоты с интеллектуальным сглаживанием

        Ключевые принципы:
        1. Все данные приводятся к месячной частоте через усреднение
        2. Высокочастотные данные сглаживаются перед агрегацией
        3. Квартальные данные распределяются правильно
        4. Минимизация потери информации
        """
        print("\nСтандартизация частоты данных...")

        # Определяем временные границы
        first_valid = df.first_valid_index()
        last_valid = df.last_valid_index()

        if first_valid is None or last_valid is None:
            return df

        # Создаем единый месячный индекс
        monthly_index = pd.date_range(
            start=first_valid.replace(day=1),
            end=last_valid,
            freq='M'
        )

        # Результирующий DataFrame
        df_monthly = pd.DataFrame(index=monthly_index)

        # Обрабатываем каждый столбец индивидуально
        for col in df.columns:
            print(f"  Обработка {col}...", end="")
            series = df[col].dropna()

            if len(series) == 0:
                print(" пропущен (нет данных)")
                continue

            # Анализируем частоту данных
            freq_info = self._analyze_frequency(series)

            if freq_info['type'] == 'quarterly':
                # Специальная обработка квартальных данных
                monthly_data = self._quarterly_to_monthly_proper(series)
                df_monthly[col] = monthly_data
                print(f" квартальные → месячные (метод распределения)")

            elif freq_info['type'] == 'monthly':
                # Месячные данные просто выравниваем по индексу
                series_aligned = series.resample('M').last()
                df_monthly[col] = series_aligned
                print(f" месячные (выравнивание)")

            elif freq_info['type'] == 'weekly':
                # Недельные данные - усредняем по месяцам
                # Сначала применяем легкое сглаживание
                series_smooth = series.rolling(
                    window=2,  # 2-недельное окно
                    min_periods=1,
                    center=True
                ).mean()
                series_monthly = series_smooth.resample('M').mean()
                df_monthly[col] = series_monthly
                print(f" недельные → месячные (среднее)")

            elif freq_info['type'] == 'daily':
                # Дневные данные - применяем адаптивное сглаживание
                # Определяем оптимальное окно сглаживания на основе волатильности
                volatility = series.pct_change().std()

                if volatility > 0.05:  # Высокая волатильность (например, VIX, валюты)
                    window = 10
                    print_suffix = "высокая волатильность"
                elif volatility > 0.02:  # Средняя волатильность
                    window = 7
                    print_suffix = "средняя волатильность"
                else:  # Низкая волатильность (например, процентные ставки)
                    window = 5
                    print_suffix = "низкая волатильность"

                # Применяем адаптивное сглаживание
                series_smooth = self._adaptive_smooth(series, window=window)
                series_monthly = series_smooth.resample('M').mean()
                df_monthly[col] = series_monthly
                print(f" дневные → месячные ({print_suffix}, окно={window})")

            else:
                # Неопределенная частота - используем безопасный подход
                series_monthly = series.resample('M').mean()
                df_monthly[col] = series_monthly
                print(f" неопределенная частота → месячные (среднее)")

        # Интеллектуальное заполнение пропусков
        print("\nЗаполнение пропусков...")
        df_monthly = self._smart_fill_missing(df_monthly)

        print(f"\nСтандартизация завершена:")
        print(f"  Итоговая форма: {df_monthly.shape}")
        print(f"  Период: {df_monthly.index.min()} - {df_monthly.index.max()}")

        return df_monthly

    def _analyze_frequency(self, series: pd.Series) -> dict:
        """
        Анализирует частоту временного ряда
        """
        if len(series) < 2:
            return {'type': 'unknown', 'days': None}

        # Пробуем определить частоту через pandas
        freq_guess = pd.infer_freq(series.index)

        if freq_guess:
            if 'Q' in freq_guess:
                return {'type': 'quarterly', 'freq': freq_guess}
            elif 'M' in freq_guess:
                return {'type': 'monthly', 'freq': freq_guess}
            elif 'W' in freq_guess:
                return {'type': 'weekly', 'freq': freq_guess}
            elif 'D' in freq_guess or 'B' in freq_guess:
                return {'type': 'daily', 'freq': freq_guess}

        # Если pandas не смог определить, анализируем вручную
        # Берем медианную разницу между наблюдениями
        time_diffs = series.index.to_series().diff().dt.days.dropna()
        median_days = time_diffs.median()

        if 85 <= median_days <= 95:
            return {'type': 'quarterly', 'days': median_days}
        elif 28 <= median_days <= 32:
            return {'type': 'monthly', 'days': median_days}
        elif 6 <= median_days <= 8:
            return {'type': 'weekly', 'days': median_days}
        elif median_days <= 5:
            return {'type': 'daily', 'days': median_days}
        else:
            return {'type': 'unknown', 'days': median_days}

    def _adaptive_smooth(self, series: pd.Series, window: int) -> pd.Series:
        """
        Адаптивное сглаживание с обработкой выбросов
        """
        # Шаг 1: Определяем и обрабатываем выбросы
        # Используем метод IQR (межквартильный размах)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        # Определяем границы выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Создаем копию для обработки
        series_clean = series.copy()

        # Заменяем выбросы на границы (винзоризация)
        series_clean = series_clean.clip(lower=lower_bound, upper=upper_bound)

        # Шаг 2: Применяем скользящую среднюю с разными весами
        # Используем экспоненциально взвешенную среднюю для более плавного результата
        series_smooth = series_clean.ewm(
            span=window,
            min_periods=int(window * 0.6),  # Минимум 60% окна для валидного значения
            adjust=True  # Корректировка для начала ряда
        ).mean()

        return series_smooth

    def _smart_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Интеллектуальное заполнение пропущенных значений
        """
        df_filled = df.copy()

        for col in df.columns:
            missing_count = df[col].isna().sum()

            if missing_count == 0:
                continue

            # Определяем характер пропусков
            total_count = len(df[col])
            missing_pct = missing_count / total_count

            if missing_pct > 0.5:
                # Если больше половины данных отсутствует, не заполняем
                print(f"  {col}: слишком много пропусков ({missing_pct:.1%}), оставляем как есть")
                continue

            # Определяем паттерн пропусков
            is_start_missing = df[col].iloc[:12].isna().sum() > 6
            is_end_missing = df[col].iloc[-12:].isna().sum() > 6

            if is_start_missing and not is_end_missing:
                # Пропуски в начале - используем backward fill с ограничением
                df_filled[col] = df[col].fillna(method='bfill', limit=6)
                print(f"  {col}: заполнение пропусков в начале (bfill)")

            elif is_end_missing and not is_start_missing:
                # Пропуски в конце - используем forward fill с ограничением
                df_filled[col] = df[col].fillna(method='ffill', limit=6)
                print(f"  {col}: заполнение пропусков в конце (ffill)")

            else:
                # Пропуски в середине или распределены - используем интерполяцию
                # Сначала заполняем краевые значения
                df_filled[col] = df[col].fillna(method='ffill', limit=3)
                df_filled[col] = df_filled[col].fillna(method='bfill', limit=3)

                # Затем интерполируем оставшиеся
                if df_filled[col].isna().sum() > 0:
                    df_filled[col] = df_filled[col].interpolate(
                        method='polynomial',
                        order=2,  # Квадратичная интерполяция
                        limit=6,  # Максимум 6 месяцев подряд
                        limit_direction='both'
                    )
                print(f"  {col}: интерполяция внутренних пропусков")

        return df_filled

    def _quarterly_to_monthly_proper(self, quarterly_series: pd.Series) -> pd.Series:
        """
        Правильное преобразование квартальных данных в месячные
        Решает проблему смещения First_Valid
        """
        # Создаем результирующую серию
        result = pd.Series(dtype=float)

        for date, value in quarterly_series.items():
            # Определяем квартал и год
            year = date.year
            quarter = date.quarter

            # Определяем месяцы для квартала
            if quarter == 1:
                months = [1, 2, 3]
            elif quarter == 2:
                months = [4, 5, 6]
            elif quarter == 3:
                months = [7, 8, 9]
            else:
                months = [10, 11, 12]

            # Присваиваем значение всем месяцам квартала
            for month in months:
                # Используем последний день месяца для консистентности
                month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                result[month_end] = value

        return result

    def add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Улучшенное создание составных индикаторов с проверками
        """
        print("\nСоздание составных индикаторов...")

        # Стандартизируем данные перед созданием композитов
        df_standardized = df.copy()

        # Z-score нормализация для корректного усреднения
        for col in df_standardized.columns:
            if df_standardized[col].std() > 0:
                df_standardized[col] = (df_standardized[col] - df_standardized[col].mean()) / df_standardized[col].std()

        # Индикатор финансового стресса
        stress_components = []
        if 'VIX' in df.columns:
            stress_components.append(df_standardized['VIX'])
        if 'High Yield Spread' in df.columns:
            stress_components.append(df_standardized['High Yield Spread'])
        if '10Y-2Y Spread' in df.columns:
            # Инвертируем, так как отрицательный спред = стресс
            stress_components.append(-df_standardized['10Y-2Y Spread'])

        if stress_components:
            df['Financial_Stress_Index'] = pd.concat(stress_components, axis=1).mean(axis=1)
            print("  ✓ Создан Financial_Stress_Index")

        # Композитный индикатор роста
        growth_cols = ['Real GDP', 'Industrial Production Index', 'Retail Sales Ex Food Services',
                      'Nonfarm Payrolls']
        available_growth = [col for col in growth_cols if col in df.columns]
        if len(available_growth) >= 2:
            df['Composite_Growth'] = df_standardized[available_growth].mean(axis=1)
            print(f"  ✓ Создан Composite_Growth из {len(available_growth)} компонентов")

        # Композитный индикатор инфляции
        inflation_cols = ['Core CPI', 'CPI All Items', 'PPI All Commodities']
        available_inflation = [col for col in inflation_cols if col in df.columns]
        if len(available_inflation) >= 2:
            df['Composite_Inflation'] = df_standardized[available_inflation].mean(axis=1)
            print(f"  ✓ Создан Composite_Inflation из {len(available_inflation)} компонентов")

        # Индикатор экономического режима (простая версия)
        if 'Composite_Growth' in df.columns and 'Composite_Inflation' in df.columns:
            df['Economic_Regime_Score'] = df['Composite_Growth'] - 0.5 * df['Composite_Inflation']
            print("  ✓ Создан Economic_Regime_Score")

        return df

    def apply_publication_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет реалистичные публикационные лаги к данным
        """
        print("\nПрименение публикационных лагов...")

        df_lagged = df.copy()

        # Собираем информацию о лагах
        lag_info = {}
        for category in self.indicators.values():
            for series_id, info in category.items():
                if info['name'] in df.columns:
                    lag_days = info.get('publication_lag', 0)
                    if lag_days > 0:
                        lag_info[info['name']] = lag_days

        # Применяем лаги
        for col, lag_days in lag_info.items():
            if col in df_lagged.columns:
                # Сдвигаем данные на количество дней публикационного лага
                df_lagged[col] = df_lagged[col].shift(periods=lag_days // 30)  # Приблизительно в месяцах
                print(f"  Применен лаг {lag_days} дней к {col}")

        return df_lagged

    def create_data_quality_plots(self, df: pd.DataFrame):
        """
        Создает визуализации для оценки качества данных
        """
        print("\nСоздание визуализаций качества данных...")

        # 1. Heatmap доступности данных
        plt.figure(figsize=(20, 12))

        # Создаем бинарную матрицу доступности
        availability = (~df.isna()).astype(int)

        # Агрегируем по годам для лучшей визуализации
        yearly_availability = availability.resample('Y').mean()

        sns.heatmap(yearly_availability.T, cmap='RdYlGn', cbar_kws={'label': 'Доля доступных данных'},
                   xticklabels=[str(d.year) for d in yearly_availability.index[::5]],
                   yticklabels=yearly_availability.columns)

        plt.title('Доступность данных по годам', fontsize=16, pad=20)
        plt.xlabel('Год', fontsize=12)
        plt.ylabel('Индикатор', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/plots/data_availability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. График покрытия данных по времени
        plt.figure(figsize=(15, 8))

        coverage = availability.sum(axis=1) / len(df.columns) * 100
        coverage_smooth = coverage.rolling(window=12).mean()

        plt.plot(coverage.index, coverage, alpha=0.3, label='Месячное покрытие')
        plt.plot(coverage_smooth.index, coverage_smooth, linewidth=2, label='12-месячное среднее')

        # Добавляем важные экономические события
        events = {
            '1973-10': 'Нефтяной кризис',
            '1979-07': 'Второй нефтяной шок',
            '1987-10': 'Черный понедельник',
            '1990-07': 'Рецессия начала 90-х',
            '2000-03': 'Крах доткомов',
            '2008-09': 'Финансовый кризис',
            '2020-03': 'COVID-19'
        }

        for date, event in events.items():
            event_date = pd.to_datetime(date)
            if event_date in coverage.index:
                plt.axvline(x=event_date, color='red', alpha=0.3, linestyle='--')
                plt.text(event_date, plt.ylim()[1]*0.95, event, rotation=90,
                        verticalalignment='top', fontsize=9)

        plt.title('Покрытие данных во времени', fontsize=16)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Процент доступных индикаторов', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/plots/data_coverage_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Визуализации сохранены в output/plots/")

    def generate_quality_report(self, df: pd.DataFrame):
        """
        Генерирует детальный отчет о качестве данных
        """
        report = []
        report.append("="*80)
        report.append("ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ")
        report.append("="*80)
        report.append(f"\nПериод данных: {df.index.min()} - {df.index.max()}")
        report.append(f"Всего индикаторов: {len(df.columns)}")
        report.append(f"Всего наблюдений: {len(df)}")

        # Статистика по покрытию
        report.append("\n" + "-"*60)
        report.append("СТАТИСТИКА ПОКРЫТИЯ")
        report.append("-"*60)

        for col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                coverage_pct = (len(valid_data) / len(df)) * 100
                report.append(f"\n{col}:")
                report.append(f"  Первое значение: {valid_data.index[0]}")
                report.append(f"  Последнее значение: {valid_data.index[-1]}")
                report.append(f"  Покрытие: {coverage_pct:.1f}%")
                report.append(f"  Пропущенных значений: {df[col].isna().sum()}")

        # Сохраняем отчет
        with open('output/data_quality_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print("\n  ✓ Отчет о качестве данных сохранен в output/data_quality_report.txt")

    def collect_all_data(self) -> pd.DataFrame:
        """
        Основной метод сбора всех данных с улучшениями
        """
        all_data = {}
        failed_indicators = []

        # Проходим по всем категориям индикаторов
        for category, indicators in self.indicators.items():
            print(f"\n{'='*60}")
            print(f"Загрузка категории: {category}")
            print(f"{'='*60}")

            for series_id, info in indicators.items():
                df = self.fetch_single_indicator(series_id, info)
                if df is not None and len(df) > 0:
                    # Сохраняем с понятным именем
                    column_name = df.columns[0]
                    all_data[info['name']] = df[column_name]
                else:
                    failed_indicators.append((series_id, info['name']))

        if failed_indicators:
            print(f"\n⚠️  Не удалось загрузить {len(failed_indicators)} индикаторов:")
            for series_id, name in failed_indicators:
                print(f"   - {name} ({series_id})")

        # Объединяем все данные в один DataFrame
        combined_df = pd.DataFrame(all_data)

        # Применяем улучшенную стандартизацию частоты
        combined_df = self.standardize_frequency_improved(combined_df)

        print(f"\n✓ Сбор данных завершен!")
        print(f"  Форма итогового датафрейма: {combined_df.shape}")
        print(f"  Период: {combined_df.index.min()} - {combined_df.index.max()}")

        return combined_df

    def save_data(self, df: pd.DataFrame, df_lagged: Optional[pd.DataFrame] = None):
        """
        Сохраняет данные с улучшенной организацией
        """
        # Сохраняем основные данные
        df.to_csv('output/economic_indicators_raw.csv')
        print(f"\n✓ Сырые данные сохранены в: output/economic_indicators_raw.csv")

        # Сохраняем данные с лагами, если предоставлены
        if df_lagged is not None:
            df_lagged.to_csv('output/economic_indicators_lagged.csv')
            print(f"✓ Данные с лагами сохранены в: output/economic_indicators_lagged.csv")

        # Создаем детальное описание данных
        description = pd.DataFrame({
            'Indicator': df.columns,
            'First_Valid': df.apply(lambda x: x.first_valid_index()),
            'Last_Valid': df.apply(lambda x: x.last_valid_index()),
            'Count': df.apply(lambda x: x.count()),
            'Missing_Count': df.apply(lambda x: x.isna().sum()),
            'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2),
            'Mean': df.mean().round(2),
            'Std': df.std().round(2),
            'Min': df.min().round(2),
            'Max': df.max().round(2)
        })

        description.to_csv('output/data_description_detailed.csv')
        print(f"✓ Детальное описание сохранено в: output/data_description_detailed.csv")

        # Сохраняем отчет о качестве данных в JSON
        with open('output/data_quality_report.json', 'w') as f:
            json.dump(self.data_quality_report, f, indent=2, default=str)
        print(f"✓ Отчет о качестве в JSON: output/data_quality_report.json")


def collect_economic_data_improved(api_key: str,
                                 create_plots: bool = True,
                                 apply_lags: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Улучшенная функция для сбора экономических данных

    Parameters:
    -----------
    api_key : str
        Ваш FRED API ключ
    create_plots : bool
        Создавать ли визуализации качества данных
    apply_lags : bool
        Применять ли публикационные лаги

    Returns:
    --------
    tuple : (df_raw, df_lagged)
        df_raw - сырые данные без лагов
        df_lagged - данные с применением публикационных лагов
    """
    print("="*80)
    print("УЛУЧШЕННАЯ СИСТЕМА СБОРА ЭКОНОМИЧЕСКИХ ДАННЫХ v2.0")
    print("="*80)

    # Создаем экземпляр коллектора
    collector = ImprovedEconomicDataCollector(api_key=api_key)

    # Собираем все базовые индикаторы
    df = collector.collect_all_data()

    # Добавляем составные индикаторы
    df = collector.add_composite_indicators(df)

    # Создаем версию с публикационными лагами
    df_lagged = None
    if apply_lags:
        df_lagged = collector.apply_publication_lag(df)

    # Создаем визуализации
    if create_plots:
        collector.create_data_quality_plots(df)

    # Генерируем отчет о качестве
    collector.generate_quality_report(df)

    # Сохраняем результаты
    collector.save_data(df, df_lagged)

    # Выводим итоговую статистику
    print("\n" + "="*80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*80)
    print(f"\nВсего собрано индикаторов: {len(df.columns)}")
    print(f"Период данных: {df.index.min()} - {df.index.max()}")
    print(f"Всего месячных наблюдений: {len(df)}")

    # Статистика покрытия по декадам
    print("\nПокрытие данных по декадам:")
    for decade in range(1970, 2030, 10):
        decade_start = pd.Timestamp(f'{decade}-01-01')
        decade_end = pd.Timestamp(f'{decade+9}-12-31')
        decade_data = df[(df.index >= decade_start) & (df.index <= decade_end)]
        if len(decade_data) > 0:
            coverage = (decade_data.count() / len(decade_data)).mean() * 100
            print(f"  {decade}s: {coverage:.1f}%")

    return df, df_lagged


# Пример использования
if __name__ == "__main__":
    # ВАЖНО: Замените на ваш реальный API ключ!
    API_KEY = '853c1faa729f41dc3f06e369d4bd66bd'

    # Запускаем улучшенный сбор данных
    df_raw, df_lagged = collect_economic_data_improved(
        api_key=API_KEY,
        create_plots=True,
        apply_lags=True
    )

    print("\n✓ Процесс завершен! Проверьте папку output/ для результатов.")