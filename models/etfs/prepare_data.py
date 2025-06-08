import pandas as pd
import numpy as np


def load_and_merge_financial_data():
    """
    Загружает данные из трех файлов и объединяет их в один датафрейм
    """

    # 1. Загрузка CSV файла с данными по облигациям и товарам
    bonds_df = pd.read_csv('input/bonds_others.csv')
    # Первая колонка без названия - это дата
    bonds_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    bonds_df['Date'] = pd.to_datetime(bonds_df['Date'])
    print(f"Bonds data: {bonds_df.shape[0]} строк, {bonds_df.shape[1]} колонок")
    print(f"Период: {bonds_df['Date'].min()} - {bonds_df['Date'].max()}")

    # 2. Загрузка данных ETF (лист Prices)
    etf_df = pd.read_excel('input/etf_data.xlsx', sheet_name='Prices')
    etf_df['Date'] = pd.to_datetime(etf_df['Date'])
    print(f"\nETF data: {etf_df.shape[0]} строк, {etf_df.shape[1]} колонок")
    print(f"Период: {etf_df['Date'].min()} - {etf_df['Date'].max()}")

    # 3. Загрузка данных Fama-French 12 отраслей (лист Monthly_Returns)
    ff12_df = pd.read_excel('input/ff12_data.xlsx', sheet_name='Monthly_Returns')
    ff12_df['Date'] = pd.to_datetime(ff12_df['Date'])
    print(f"\nFF12 data: {ff12_df.shape[0]} строк, {ff12_df.shape[1]} колонок")
    print(f"Период: {ff12_df['Date'].min()} - {ff12_df['Date'].max()}")

    # Приведение всех дат к концу месяца для корректного объединения месячных данных
    # FF12 данные уже по месяцам, поэтому приводим остальные к концу месяца
    bonds_df['Date_Month'] = bonds_df['Date'].dt.to_period('M').dt.to_timestamp('M')
    etf_df['Date_Month'] = etf_df['Date'].dt.to_period('M').dt.to_timestamp('M')
    ff12_df['Date_Month'] = ff12_df['Date'].dt.to_period('M').dt.to_timestamp('M')

    # Для дневных данных (bonds и ETF) берем последнее значение месяца
    bonds_monthly = bonds_df.groupby('Date_Month').last().reset_index()
    bonds_monthly = bonds_monthly.drop('Date', axis=1)
    bonds_monthly.rename(columns={'Date_Month': 'Date'}, inplace=True)

    etf_monthly = etf_df.groupby('Date_Month').last().reset_index()
    etf_monthly = etf_monthly.drop('Date', axis=1)
    etf_monthly.rename(columns={'Date_Month': 'Date'}, inplace=True)

    # Для FF12 просто переименовываем колонку
    ff12_monthly = ff12_df.drop('Date', axis=1)
    ff12_monthly.rename(columns={'Date_Month': 'Date'}, inplace=True)

    # Объединение всех датафреймов по дате
    # Сначала объединяем bonds и ETF
    merged_df = pd.merge(bonds_monthly, etf_monthly, on='Date', how='outer',
                         suffixes=('_bonds', '_etf'))

    # Затем добавляем FF12 данные
    merged_df = pd.merge(merged_df, ff12_monthly, on='Date', how='outer')

    # Сортировка по дате
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    print(f"\n=== Результат объединения ===")
    print(f"Итоговый датафрейм: {merged_df.shape[0]} строк, {merged_df.shape[1]} колонок")
    print(f"Период: {merged_df['Date'].min()} - {merged_df['Date'].max()}")
    print(f"\nКолонки:")
    print(
        f"- Из bonds_others.csv: {[col for col in merged_df.columns if col in bonds_monthly.columns and col != 'Date']}")
    print(f"- Из etf_data.xlsx: {[col for col in merged_df.columns if col in etf_monthly.columns and col != 'Date']}")
    print(f"- Из ff12_data.xlsx: {[col for col in merged_df.columns if col in ff12_monthly.columns and col != 'Date']}")

    # Подсчет заполненности данных
    print(f"\nЗаполненность данных:")
    non_null_counts = merged_df.notna().sum()
    for col in merged_df.columns:
        if col != 'Date':
            pct = (non_null_counts[col] / len(merged_df)) * 100
            print(f"{col}: {non_null_counts[col]} значений ({pct:.1f}%)")

    return merged_df


# Альтернативная версия с объединением дневных данных
def load_and_merge_daily_data():
    """
    Загружает и объединяет данные, сохраняя дневную гранулярность для bonds и ETF
    """

    # 1. Загрузка CSV файла
    bonds_df = pd.read_csv('input/bonds_others.csv')
    bonds_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    bonds_df['Date'] = pd.to_datetime(bonds_df['Date'])

    # 2. Загрузка данных ETF
    etf_df = pd.read_excel('input/etf_data.xlsx', sheet_name='Prices')
    etf_df['Date'] = pd.to_datetime(etf_df['Date'])

    # 3. Загрузка данных FF12 (месячные)
    ff12_df = pd.read_excel('input/ff12_data.xlsx', sheet_name='Monthly_Returns')
    ff12_df['Date'] = pd.to_datetime(ff12_df['Date'])

    # Объединяем дневные данные bonds и ETF
    daily_merged = pd.merge(bonds_df, etf_df, on='Date', how='outer')

    # Для объединения с месячными данными FF12, создаем колонку с концом месяца
    daily_merged['Month_End'] = daily_merged['Date'].dt.to_period('M').dt.to_timestamp('M')
    ff12_df['Month_End'] = ff12_df['Date'].dt.to_period('M').dt.to_timestamp('M')

    # Объединяем с FF12 по месяцу (FF12 значения будут повторяться для всех дней месяца)
    final_df = pd.merge(daily_merged, ff12_df.drop('Date', axis=1),
                        on='Month_End', how='left')

    # Удаляем вспомогательную колонку
    final_df = final_df.drop('Month_End', axis=1)

    # Сортировка по дате
    final_df = final_df.sort_values('Date').reset_index(drop=True)

    print(f"\n=== Результат объединения (дневные данные) ===")
    print(f"Итоговый датафрейм: {final_df.shape[0]} строк, {final_df.shape[1]} колонок")
    print(f"Период: {final_df['Date'].min()} - {final_df['Date'].max()}")

    return final_df


# Пример использования
if __name__ == "__main__":
    # Вариант 1: Месячные данные
    print("Объединение с приведением к месячным данным:")
    monthly_df = load_and_merge_financial_data()

    # Сохранение результата
    monthly_df.to_csv('merged_monthly_data.csv', index=False)
    print("\nДанные сохранены в 'merged_monthly_data.csv'")

    # Вариант 2: Дневные данные
    print("\n" + "=" * 50)
    print("Объединение с сохранением дневной гранулярности:")
    daily_df = load_and_merge_daily_data()

    # Сохранение результата
    daily_df.to_csv('merged_daily_data.csv', index=False)
    print("\nДанные сохранены в 'merged_daily_data.csv'")