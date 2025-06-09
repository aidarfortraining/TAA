"""
Подготовка и улучшение данных для модели тактического распределения активов (TAA)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def prepare_taa_data(filepath='output/merged_monthly_data.csv',
                     regime_data_path='output/regime_classification/regime_history.csv'):
    """
    Подготовка данных для модели TAA

    Parameters:
    -----------
    filepath : str
        Путь к файлу с данными активов
    regime_data_path : str
        Путь к файлу с историей экономических режимов

    Returns:
    --------
    pd.DataFrame : Подготовленные данные с доходностями и features
    """

    print("Загрузка данных...")
    # Загрузка основных данных
    df = pd.read_csv(filepath)

    # Преобразование даты
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Разделение на группы активов
    # 1. Kenneth French факторы (уже в виде доходностей)
    ff_factors = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems',
                  'BusEq', 'Telcm', 'Utils', 'Shops', 'Hlth', 'Money', 'Other']

    # 2. Традиционные активы (требуют расчета доходностей)
    traditional_assets = ['10Y_Treasury', '30Y_Treasury', '10Y_TIPS',
                          'IG_Corporate', 'HY_Bond']

    # 3. Товары (требуют расчета доходностей)
    commodities = ['WTI_Oil', 'Copper', 'Wheat', 'Gold']

    # 4. Товарные ETF
    commodity_etfs = ['DBA', 'DBB', 'DBE', 'GLD']

    # 5. Облигационные ETF
    bond_etfs = ['HYG', 'IEF', 'LQD', 'SHY', 'TIP', 'TLT']

    # 6. Секторальные ETF
    sector_etfs = ['VNQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK',
                   'XLP', 'XLU', 'XLV', 'XLY']

    # Все активы требующие расчета доходностей
    price_based_assets = (traditional_assets + commodities +
                          commodity_etfs + bond_etfs + sector_etfs)

    print("Расчет доходностей...")
    # Создаем DataFrame для доходностей
    returns_df = pd.DataFrame(index=df.index)

    # Копируем FF факторы (уже доходности)
    for factor in ff_factors:
        if factor in df.columns:
            returns_df[f'{factor}_return'] = df[factor]

    # Рассчитываем доходности для остальных активов
    for asset in price_based_assets:
        if asset in df.columns:
            # Простая доходность: (P_t - P_{t-1}) / P_{t-1}
            returns_df[f'{asset}_return'] = df[asset].pct_change()

    print("Заполнение пропущенных значений...")
    # Стратегия заполнения:
    # 1. Для ETF до их создания - используем прокси из похожих активов

    # Mapping ETF к их прокси
    etf_proxy_mapping = {
        # Облигационные ETF
        'IEF': '10Y_Treasury',  # iShares 7-10 Year Treasury
        'TLT': '30Y_Treasury',  # iShares 20+ Year Treasury
        'TIP': '10Y_TIPS',  # iShares TIPS Bond
        'LQD': 'IG_Corporate',  # iShares Investment Grade Corporate
        'HYG': 'HY_Bond',  # iShares High Yield Corporate
        'SHY': '10Y_Treasury',  # iShares 1-3 Year Treasury (используем 10Y как прокси)

        # Товарные ETF
        'GLD': 'Gold',  # SPDR Gold Shares
        'DBE': 'WTI_Oil',  # Energy ETF
        'DBB': 'Copper',  # Base metals ETF (copper как основной компонент)
        'DBA': 'Wheat',  # Agriculture ETF (wheat как прокси)
    }

    # Заполняем пропущенные значения ETF доходностями их прокси
    for etf, proxy in etf_proxy_mapping.items():
        etf_col = f'{etf}_return'
        proxy_col = f'{proxy}_return'

        if etf_col in returns_df.columns and proxy_col in returns_df.columns:
            # Заполняем только те значения, где ETF = NaN, а прокси != NaN
            mask = returns_df[etf_col].isna() & returns_df[proxy_col].notna()
            returns_df.loc[mask, etf_col] = returns_df.loc[mask, proxy_col]

    # Для секторальных ETF используем FF факторы как прокси
    sector_proxy_mapping = {
        'XLB': 'Manuf',  # Materials ≈ Manufacturing
        'XLE': 'Enrgy',  # Energy
        'XLF': 'Money',  # Financials
        'XLI': 'Manuf',  # Industrials ≈ Manufacturing
        'XLK': 'BusEq',  # Technology ≈ Business Equipment
        'XLP': 'NoDur',  # Consumer Staples
        'XLU': 'Utils',  # Utilities
        'XLV': 'Hlth',  # Healthcare
        'XLY': 'Shops',  # Consumer Discretionary ≈ Shops
        'VNQ': 'Other',  # Real Estate (используем Other как прокси)
    }

    for etf, ff_factor in sector_proxy_mapping.items():
        etf_col = f'{etf}_return'
        ff_col = f'{ff_factor}_return'

        if etf_col in returns_df.columns and ff_col in returns_df.columns:
            mask = returns_df[etf_col].isna() & returns_df[ff_col].notna()
            returns_df.loc[mask, etf_col] = returns_df.loc[mask, ff_col]

    print("Добавление статистических features...")
    # Добавляем rolling features для каждого актива
    features_df = returns_df.copy()

    # Список всех активов для анализа
    all_assets = [col.replace('_return', '') for col in returns_df.columns]

    for asset in all_assets:
        return_col = f'{asset}_return'
        if return_col in features_df.columns:
            # 3-месячная скользящая волатильность
            features_df[f'{asset}_vol_3m'] = features_df[return_col].rolling(
                window=3, min_periods=2).std()

            # 6-месячная скользящая волатильность
            features_df[f'{asset}_vol_6m'] = features_df[return_col].rolling(
                window=6, min_periods=3).std()

            # Momentum (3-месячная кумулятивная доходность)
            features_df[f'{asset}_momentum_3m'] = features_df[return_col].rolling(
                window=3, min_periods=2).apply(lambda x: (1 + x).prod() - 1)

            # Momentum (6-месячная кумулятивная доходность)
            features_df[f'{asset}_momentum_6m'] = features_df[return_col].rolling(
                window=6, min_periods=3).apply(lambda x: (1 + x).prod() - 1)

            # Скользящее среднее доходности
            features_df[f'{asset}_ma_3m'] = features_df[return_col].rolling(
                window=3, min_periods=2).mean()

    print("Добавление межклассовых корреляций...")
    # Рассчитываем rolling корреляции между классами активов
    # Акции vs Облигации (используем XLK vs IEF как прокси)
    if 'XLK_return' in features_df.columns and 'IEF_return' in features_df.columns:
        features_df['stock_bond_corr'] = features_df['XLK_return'].rolling(
            window=12, min_periods=6).corr(features_df['IEF_return'])

    # Товары vs Акции (используем DBC/GLD vs XLK)
    if 'GLD_return' in features_df.columns and 'XLK_return' in features_df.columns:
        features_df['commodity_stock_corr'] = features_df['GLD_return'].rolling(
            window=12, min_periods=6).corr(features_df['XLK_return'])

    print("Загрузка данных о режимах...")
    # Загружаем экономические режимы если файл существует
    try:
        regime_df = pd.read_csv(regime_data_path, index_col=0, parse_dates=True)
        # Переименовываем колонку если нужно
        if 'regime' not in regime_df.columns and len(regime_df.columns) == 1:
            regime_df.columns = ['regime']

        # Объединяем с основными данными
        features_df = features_df.merge(regime_df, left_index=True, right_index=True, how='left')

        # Создаем one-hot encoding для режимов
        for regime_id in [1, 2, 3, 4]:
            features_df[f'regime_{regime_id}'] = (features_df['regime'] == regime_id).astype(int)

        print("Режимы успешно добавлены")
    except:
        print("Предупреждение: Файл с режимами не найден, продолжаем без режимов")

    print("Финальная очистка данных...")
    # Удаляем строки где слишком много пропущенных значений (первые месяцы)
    # Сохраняем строки где хотя бы 50% данных доступно
    threshold = len(features_df.columns) * 0.5
    features_df = features_df.dropna(thresh=threshold)

    # Заполняем оставшиеся пропуски нулями (предполагаем нулевую доходность)
    features_df = features_df.fillna(0)

    print(f"\nИтоговый размер данных: {features_df.shape}")
    print(f"Период: {features_df.index[0]} - {features_df.index[-1]}")
    print(f"Количество features: {len(features_df.columns)}")

    # Сохраняем подготовленные данные
    output_path = 'output/taa_prepared_data.csv'
    features_df.to_csv(output_path)
    print(f"\nДанные сохранены в: {output_path}")

    # Создаем также упрощенную версию только с основными доходностями
    returns_only = returns_df.dropna(thresh=len(returns_df.columns) * 0.5).fillna(0)
    returns_only.to_csv('output/taa_returns_data.csv')
    print(f"Только доходности сохранены в: taa_returns_data.csv")

    return features_df


def create_asset_groups_mapping():
    """
    Создает маппинг активов по группам для использования в модели
    """
    asset_groups = {
        'Equities_FF': ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems',
                        'BusEq', 'Telcm', 'Utils', 'Shops', 'Hlth', 'Money', 'Other'],

        'Equities_ETF': ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
                         'XLU', 'XLV', 'XLY', 'VNQ'],

        'Bonds_Traditional': ['10Y_Treasury', '30Y_Treasury', '10Y_TIPS',
                              'IG_Corporate', 'HY_Bond'],

        'Bonds_ETF': ['IEF', 'TLT', 'TIP', 'LQD', 'HYG', 'SHY'],

        'Commodities': ['WTI_Oil', 'Copper', 'Wheat', 'Gold'],

        'Commodities_ETF': ['DBA', 'DBB', 'DBE', 'GLD']
    }

    # Сохраняем маппинг
    import json
    with open('output/asset_groups_mapping.json', 'w') as f:
        json.dump(asset_groups, f, indent=2)

    print("Маппинг групп активов сохранен в: asset_groups_mapping.json")

    return asset_groups


if __name__ == "__main__":
    # Подготавливаем данные
    prepared_data = prepare_taa_data()

    # Создаем маппинг групп активов
    asset_groups = create_asset_groups_mapping()

    # Выводим статистику по группам
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПО ГРУППАМ АКТИВОВ")
    print("=" * 60)

    for group_name, assets in asset_groups.items():
        available_assets = [asset for asset in assets
                            if f'{asset}_return' in prepared_data.columns]
        print(f"\n{group_name}:")
        print(f"  Доступно: {len(available_assets)} из {len(assets)}")
        print(f"  Активы: {', '.join(available_assets[:5])}" +
              (" ..." if len(available_assets) > 5 else ""))