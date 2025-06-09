"""
Анализ и визуализация подготовленных данных для TAA модели
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings

warnings.filterwarnings('ignore')


def analyze_prepared_data(data_path='output/taa_prepared_data.csv',
                          returns_path='output/taa_returns_data.csv'):
    """
    Анализирует подготовленные данные и создает визуализации
    """

    print("Загрузка подготовленных данных...")
    features_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # Загружаем маппинг групп
    with open('output/asset_groups_mapping.json', 'r') as f:
        asset_groups = json.load(f)

    # Статистика по периодам
    print("\n" + "=" * 60)
    print("АНАЛИЗ ДАННЫХ")
    print("=" * 60)
    print(f"Период данных: {features_df.index[0].date()} - {features_df.index[-1].date()}")
    print(f"Количество месяцев: {len(features_df)}")
    print(f"Количество features: {len(features_df.columns)}")

    # Анализ доступности данных по периодам
    print("\nДоступность данных по десятилетиям:")
    decades = {
        '1970s': ('1970-01-01', '1979-12-31'),
        '1980s': ('1980-01-01', '1989-12-31'),
        '1990s': ('1990-01-01', '1999-12-31'),
        '2000s': ('2000-01-01', '2009-12-31'),
        '2010s': ('2010-01-01', '2019-12-31'),
        '2020s': ('2020-01-01', '2025-12-31')
    }

    for decade, (start, end) in decades.items():
        decade_data = returns_df[start:end]
        if len(decade_data) > 0:
            non_zero_cols = (decade_data != 0).sum()
            available_assets = (non_zero_cols > len(decade_data) * 0.5).sum()
            print(f"  {decade}: {available_assets} активов с данными")

    # Статистика доходностей по группам
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ДОХОДНОСТЕЙ ПО ГРУППАМ")
    print("=" * 60)

    group_stats = {}

    for group_name, assets in asset_groups.items():
        return_cols = [f'{asset}_return' for asset in assets
                       if f'{asset}_return' in returns_df.columns]

        if return_cols:
            group_returns = returns_df[return_cols]

            # Рассчитываем статистику
            stats_dict = {
                'mean_return': group_returns.mean().mean() * 12,  # Годовая
                'volatility': group_returns.std().mean() * np.sqrt(12),  # Годовая
                'sharpe': (group_returns.mean().mean() * 12) / (group_returns.std().mean() * np.sqrt(12)),
                'min_return': group_returns.min().min(),
                'max_return': group_returns.max().max(),
                'skewness': group_returns.apply(lambda x: stats.skew(x.dropna())).mean(),
                'kurtosis': group_returns.apply(lambda x: stats.kurtosis(x.dropna())).mean()
            }

            group_stats[group_name] = stats_dict

            print(f"\n{group_name}:")
            print(f"  Средняя годовая доходность: {stats_dict['mean_return']:.2%}")
            print(f"  Годовая волатильность: {stats_dict['volatility']:.2%}")
            print(f"  Коэффициент Шарпа: {stats_dict['sharpe']:.3f}")
            print(f"  Асимметрия: {stats_dict['skewness']:.3f}")
            print(f"  Эксцесс: {stats_dict['kurtosis']:.3f}")

    # Анализ корреляций между группами
    print("\n" + "=" * 60)
    print("КОРРЕЛЯЦИИ МЕЖДУ ГРУППАМИ АКТИВОВ")
    print("=" * 60)

    # Создаем средние доходности по группам
    group_returns_df = pd.DataFrame(index=returns_df.index)

    for group_name, assets in asset_groups.items():
        return_cols = [f'{asset}_return' for asset in assets
                       if f'{asset}_return' in returns_df.columns]
        if return_cols:
            group_returns_df[group_name] = returns_df[return_cols].mean(axis=1)

    # Корреляционная матрица
    corr_matrix = group_returns_df.corr()

    print("\nКорреляционная матрица групп активов:")
    print(corr_matrix.round(3))

    # Визуализации
    create_visualizations(returns_df, features_df, group_returns_df,
                          group_stats, corr_matrix, asset_groups)

    # Анализ по режимам (если доступны)
    if 'regime' in features_df.columns:
        analyze_regimes(returns_df, features_df, asset_groups)

    return features_df, returns_df, group_stats


def create_visualizations(returns_df, features_df, group_returns_df,
                          group_stats, corr_matrix, asset_groups):
    """
    Создает визуализации для анализа данных
    """

    # Настройка стиля
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Корреляционная матрица групп
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Корреляции между группами активов', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('taa_group_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Риск-доходность по группам
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']

    for i, (group_name, stats) in enumerate(group_stats.items()):
        ax.scatter(stats['volatility'], stats['mean_return'],
                   s=200, c=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=group_name, alpha=0.7, edgecolors='black')

        # Добавляем подписи
        ax.annotate(group_name,
                    (stats['volatility'], stats['mean_return']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.7)

    ax.set_xlabel('Волатильность (годовая)', fontsize=12)
    ax.set_ylabel('Доходность (годовая)', fontsize=12)
    ax.set_title('Профиль риск-доходность по группам активов', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Добавляем линию Шарпа = 0.5
    x_range = np.array(ax.get_xlim())
    ax.plot(x_range, 0.5 * x_range, 'k--', alpha=0.3, label='Sharpe = 0.5')

    plt.tight_layout()
    plt.savefig('output/taa_risk_return_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Временной ряд кумулятивных доходностей по группам
    fig, ax = plt.subplots(figsize=(14, 8))

    cumulative_returns = (1 + group_returns_df).cumprod()

    for i, column in enumerate(cumulative_returns.columns):
        ax.plot(cumulative_returns.index, cumulative_returns[column],
                label=column, linewidth=2, alpha=0.8)

    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Кумулятивная доходность', fontsize=12)
    ax.set_title('Кумулятивные доходности групп активов', fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('taa_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Rolling корреляции между акциями и облигациями
    if 'Equities_ETF' in group_returns_df.columns and 'Bonds_ETF' in group_returns_df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        rolling_corr = group_returns_df['Equities_ETF'].rolling(
            window=36, min_periods=12).corr(group_returns_df['Bonds_ETF'])

        ax.plot(rolling_corr.index, rolling_corr, linewidth=2, color='darkblue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(rolling_corr.index, 0, rolling_corr,
                        where=(rolling_corr < 0), alpha=0.3, color='red',
                        label='Отрицательная корреляция')
        ax.fill_between(rolling_corr.index, 0, rolling_corr,
                        where=(rolling_corr >= 0), alpha=0.3, color='green',
                        label='Положительная корреляция')

        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Корреляция', fontsize=12)
        ax.set_title('36-месячная скользящая корреляция: Акции vs Облигации',
                     fontsize=14, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)

        plt.tight_layout()
        plt.savefig('taa_rolling_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("\nВизуализации сохранены:")
    print("  - taa_group_correlations.png")
    print("  - taa_risk_return_profile.png")
    print("  - taa_cumulative_returns.png")
    print("  - taa_rolling_correlation.png")


def analyze_regimes(returns_df, features_df, asset_groups):
    """
    Анализирует доходности по экономическим режимам
    """

    print("\n" + "=" * 60)
    print("АНАЛИЗ ДОХОДНОСТЕЙ ПО ЭКОНОМИЧЕСКИМ РЕЖИМАМ")
    print("=" * 60)

    regime_names = {
        1: 'Goldilocks',
        2: 'Reflation',
        3: 'Deflation',
        4: 'Stagflation'
    }

    # Объединяем режимы с доходностями
    regime_returns = returns_df.copy()
    regime_returns['regime'] = features_df['regime']

    # Анализ по группам и режимам
    results = []

    for group_name, assets in asset_groups.items():
        return_cols = [f'{asset}_return' for asset in assets
                       if f'{asset}_return' in returns_df.columns]

        if return_cols:
            for regime_id, regime_name in regime_names.items():
                regime_data = regime_returns[regime_returns['regime'] == regime_id][return_cols]

                if len(regime_data) > 0:
                    results.append({
                        'Group': group_name,
                        'Regime': regime_name,
                        'Avg_Return': regime_data.mean().mean() * 12,
                        'Volatility': regime_data.std().mean() * np.sqrt(12),
                        'Sharpe': (regime_data.mean().mean() * 12) /
                                  (regime_data.std().mean() * np.sqrt(12) + 1e-6),
                        'Observations': len(regime_data)
                    })

    results_df = pd.DataFrame(results)

    # Выводим сводную таблицу
    pivot_return = results_df.pivot(index='Group', columns='Regime', values='Avg_Return')
    pivot_volatility = results_df.pivot(index='Group', columns='Regime', values='Volatility')
    pivot_sharpe = results_df.pivot(index='Group', columns='Regime', values='Sharpe')

    print("\nСредняя годовая доходность по режимам (%):")
    print(pivot_return.round(2))

    print("\nГодовая волатильность по режимам (%):")
    print(pivot_volatility.round(2))

    print("\nКоэффициент Шарпа по режимам:")
    print(pivot_sharpe.round(3))

    # Визуализация: тепловая карта доходностей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Доходности
    sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, ax=ax1, cbar_kws={'label': 'Годовая доходность (%)'})
    ax1.set_title('Средняя годовая доходность по группам и режимам', fontsize=12)
    ax1.set_xlabel('Экономический режим')
    ax1.set_ylabel('Группа активов')

    # Коэффициенты Шарпа
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax2, cbar_kws={'label': 'Коэффициент Шарпа'})
    ax2.set_title('Коэффициент Шарпа по группам и режимам', fontsize=12)
    ax2.set_xlabel('Экономический режим')
    ax2.set_ylabel('Группа активов')

    plt.tight_layout()
    plt.savefig('taa_regime_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nВизуализация сохранена: taa_regime_performance.png")

    # Сохраняем результаты
    results_df.to_csv('taa_regime_analysis.csv', index=False)
    print("Детальный анализ сохранен: taa_regime_analysis.csv")


if __name__ == "__main__":
    # Анализируем подготовленные данные
    features_df, returns_df, group_stats = analyze_prepared_data()

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 60)
    print("\nДанные готовы для построения модели TAA!")
    print("\nСледующие шаги:")
    print("1. Использовать taa_prepared_data.csv для обучения модели")
    print("2. Учитывать результаты анализа по режимам из taa_regime_analysis.csv")
    print("3. Применить оптимизацию портфеля с учетом ограничений")