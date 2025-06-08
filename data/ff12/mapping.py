def convert_returns(ff12_returns):
    etf_returns = {}

    etf_returns['XLE'] = ff12_returns['Enrgy']
    etf_returns['XLU'] = ff12_returns['Utils']
    etf_returns['XLV'] = ff12_returns['Hlth']
    etf_returns['XLF'] = ff12_returns['Money']
    etf_returns['XLP'] = ff12_returns['NoDur']
    etf_returns['XLI'] = ff12_returns['Manuf']
    etf_returns['XLB'] = ff12_returns['Chems']
    etf_returns['VNQ'] = ff12_returns['Other']
    etf_returns['XLY'] = (ff12_returns['Durbl'] + ff12_returns['Shops']) / 2
    etf_returns['XLK'] = (ff12_returns['BusEq'] + ff12_returns['Telcm']) / 2

    return etf_returns
