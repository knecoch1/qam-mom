import pandas as pd
import wrds
import numpy as np
from scipy import stats
import pandas_datareader

def fill_dates(group):
    """
    Fills dates in a return series where there are months with empty returns

    """
    # Generate all period range
    min_period = group['period'].min()
    max_period = group['period'].max()
    all_periods = pd.period_range(start=min_period, end=max_period, freq='M')
    all_periods_df = pd.DataFrame(all_periods, columns=['period'])
    all_periods_df['permno'] = group['permno'].iloc[0]
    
    # Merge with original data
    merged_group = pd.merge(all_periods_df, group, on=['period', 'permno'], how='left')
    return merged_group

def bins3(df, K, s, d, use_cutoff_flag = False):
    df = df.copy()
    # Defining the momentum percentiles
    df['brkpts_flag'] = True
    if use_cutoff_flag:
        df['brkpts_flag'] = (df['cutoff_flag'] == 1)

    def diff_brkpts(x,K):
        brkpts_flag = x['brkpts_flag']
        x = x[s]
        loc_nyse = x.notna() & brkpts_flag  
        if np.sum(loc_nyse) > 0:
            breakpoints = pd.qcut(x[loc_nyse], K, retbins=True, labels=False)[1]
            breakpoints[0] = -np.inf
            breakpoints[K] = np.inf
            y = pd.cut(x, bins=breakpoints, labels=False) + 1
        else:
            y = x + np.nan
        return y

    df['bin'] = df.groupby(d).apply(lambda x:diff_brkpts(x,K)).reset_index()[s]
    df['bin'] = df['bin'].astype('float')
    df.drop('brkpts_flag',axis=1,inplace=True)
    return df

def calcLongShort(df, bin_col='bin', return_col='return', high_bin_col=10, low_bin_col=1):
    """
    Calculate the difference between the return of the specified high bin and the specified low bin for each date.
    Add this difference as a new bin (bin 11) for each date.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the necessary columns.
    bin_col (str): Column name for the bins.
    return_col (str): Column name for the returns.
    high_bin_col (int or str): Value of the bin to use as the high bin.
    low_bin_col (int or str): Value of the bin to use as the low bin.

    Returns:
    pd.DataFrame: DataFrame with the original data and the return difference as bin 11.
    """
    # Pivot the DataFrame to have bins as columns
    pivot_df = df.pivot(index='date', columns=bin_col, values=return_col)
    
    # Check if the specified bin columns exist in the pivoted DataFrame
    if high_bin_col not in pivot_df.columns or low_bin_col not in pivot_df.columns:
        raise ValueError(f"Specified bin columns {high_bin_col} or {low_bin_col} not found in the DataFrame")

    # Calculate the difference between the specified high bin and low bin
    pivot_df['11'] = pivot_df[high_bin_col] - pivot_df[low_bin_col]

    # Melt the DataFrame back to long format
    result_df = pivot_df.reset_index().melt(id_vars=['date'], var_name=bin_col, value_name=return_col)
    
    # Convert bin to integer and sort the DataFrame
    result_df[bin_col] = result_df[bin_col].astype(int)
    result_df = result_df.sort_values(by=['date', bin_col]).reset_index(drop=True)

    return result_df

def loadStockMonthly(reload = False):

    if reload:
        conn = wrds.Connection(wrds_username="kevinneco")

        # Share codes and exchage code that we'll use
        min_shrcd = 10
        max_shrcd = 11
        possible_exchcd = (1, 2, 3)

        # Time period
        min_year = 1926
        max_year = 2023

        dcrsp_raw = conn.raw_sql("""
                            select a.permno, b.shrcd, b.exchcd, a.date, a.ret, a.shrout, a.prc
                            from crsp.msf as a
                            left join crspq.dsenames as b
                            on a.permno=b.permno
                            and b.namedt<=a.date
                            and a.date<=b.nameendt
                            where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
                            and a.date between '01/01/""" +str(min_year)+ """' and '12/31/""" +str(max_year)+ """'
                            and b.exchcd in """ + str(possible_exchcd) + """
                            """)
        dcrsp_raw['date'] = pd.to_datetime(dcrsp_raw['date'])
        dcrsp_raw.sort_values(['permno', 'date'], inplace=True)
        dcrsp_raw.reset_index(drop=True, inplace=True)

        dlret_raw = conn.raw_sql("""
                            select a.permno, a.dlstdt, a.dlret
                            from crspq.msedelist as a
                            left join crspq.dsenames as b
                            on a.permno=b.permno
                            and b.namedt<=a.dlstdt
                            and a.dlstdt<=b.nameendt
                            where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
                            and a.dlstdt between '01/01/""" +str(min_year)+ """' and '12/31/""" +str(max_year)+ """'
                            and b.exchcd in """ + str(possible_exchcd) + """
                            """)

        dlret_raw.rename(columns={'dlstdt': 'date'}, inplace=True)
        dlret_raw['date'] = pd.to_datetime(dlret_raw['date'])
        dlret_raw['date'] = dlret_raw['date'] + pd.tseries.offsets.MonthEnd(0)
        dlret_raw.sort_values(['permno', 'date'], inplace=True)
        dlret_raw.reset_index(drop=True, inplace=True)

        # Merge datasets returns
        dlret = dlret_raw.copy()
        df = pd.merge(dcrsp_raw, dlret, on=['date', 'permno'], how='outer')
        df.sort_values(['permno', 'date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Calculate total return
        df['tret'] = df['ret']
        df['tret'] = np.where(df['tret'].notna() & df['dlret'].notna(), 
                                (1+df['tret'])*(1+df['dlret'])-1, df['tret'])
        df['tret'] = np.where(df['tret'].isna()  & df['dlret'].notna(), df['dlret'], df['tret'])
        df = df[df['tret'].notna()].copy() # Dropping missing returns.

        # Calculate market cap
        df['me'] = df['prc'].abs() * df['shrout']

        # Order by permno and date for future order sensitive operations
        dfs = df[['permno', 'date','exchcd', 'tret', 'me']].copy()
        dfs.sort_values(['permno', 'date'], inplace=True)
        dfs.reset_index(drop=True, inplace=True)

        # Change date type and create period column
        dfs['date'] = pd.to_datetime(dfs['date'])
        dfs['period'] = dfs['date'].dt.to_period('M')

        # Fill entries where there were months with no returns
        dffull = dfs.groupby('permno').apply(fill_dates).reset_index(drop=True)

        # Create lag market cap
        dffull.sort_values(['permno', 'period'], inplace=True)
        dffull.reset_index(drop=True, inplace=True)
        dffull['lagme'] = dffull.groupby('permno')['me'].shift(1)

        # Make sure new NA entries have an exchanged code
        dffull['exchcd'] = dffull.groupby('permno')['exchcd'].fillna(method='ffill')

        # save to picle file
        dffull.to_pickle('monthlyRet.pkl')
    else:
        dffull = pd.read_pickle('monthlyRet.pkl')


    return dffull

def assignMomDeciles(reload = False):

    if reload:
        df = loadStockMonthly(reload = False)

        # Calculate rolling 11 month returns and add 2 month lag column (this will be signal)
        df['mltp'] = df.tret.fillna(0) + 1
        df['cumret'] = df.groupby('permno').mltp.cumprod()
        df['ct'] = 1
        df['ct'] = df.groupby('permno').ct.cumsum()
        i = 11
        df['ret11m'] = np.where(df['ct'] == i,df.cumret-1,df.cumret / df.groupby('permno').cumret.shift(i)-1)
        df['ret11mLag2'] = df.groupby('permno')['ret11m'].shift(2)

        df = df[['period', 'permno', 'exchcd', 'lagme', 'tret', 'ret11mLag2', 'ct']].copy()

        i = 11

        # Find when a rolling return has NAs in monthly return series
        df['isRetNA'] = df['tret'].isna()
        df['cumNA'] = df.groupby('permno')['isRetNA'].cumsum()
        df['NA11m'] = np.where(df['ct'] == i,df.cumNA,df.cumNA - df.groupby('permno').cumNA.shift(i))
        df['NA11mLag2'] = df.groupby('permno')['NA11m'].shift(2)

        # Only keep signal with less than or equal to 3 NA values in rolling return calculation
        df = df[df['NA11mLag2'] <= 3]
        df.sort_values(['period', 'permno'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Assign deciles to returns (all data breakpoints)
        dd = bins3(df,10,'ret11mLag2','period',False)
        dds = dd[['period', 'permno', 'lagme', 'tret', 'exchcd', 'bin']]
        dds = dds.rename(
            columns = {
                'period': 'date',
                'lagme': 'lag_Mkt_cap',
                'tret': 'ret',
                'bin': 'DM_Decile'
            }
        )

        df['cutoff_flag'] = df['exchcd']
        ddny = bins3(df,10,'ret11mLag2','period',True)
        ddnys = ddny[['period', 'permno', 'bin']]
        ddnys = ddnys.rename(
            columns = {
                'period': 'date',
                'bin': 'KRF_Decile'
            }
        )

        ddAll = pd.merge(dds, ddnys, on = ['date', 'permno'], how = 'left')

        # Save to pickle file
        ddAll.to_pickle('momDeciles.pkl')
    else:
        ddAll = pd.read_pickle('momDeciles.pkl')

    return ddAll

def calcMomDeciles(reload = False):
    if reload:
        # Assign deciles to returns (all data breakpoints)

        dd = assignMomDeciles(reload = False)

        mktVal = dd.groupby(['date', 'DM_Decile'])['lag_Mkt_cap'].sum()
        mktVal = mktVal.reset_index()
        mktVal.rename(columns={'lag_Mkt_cap': 'lagMV'}, inplace=True)

        ddm = pd.merge(dd.copy(), mktVal, on = ['date', 'DM_Decile'], how = 'left')

        # Calculate value wtd return
        ddm['wgt'] = ddm['lag_Mkt_cap'] / ddm['lagMV']
        ddm['wtdRet'] = ddm['ret'] * ddm['wgt']

        # Add valued wtd returns to calculate total vw ret
        ddmRet = ddm.groupby(['date', 'DM_Decile'])['wtdRet'].sum()
        ddmRet = ddmRet.reset_index()
        ddmRet = ddmRet.rename(columns={
            'wtdRet': 'DM_Ret' # Change 'mvWtdRet' to 'new_mvWtdRet_name'
        })
        ddmRet['DM_Decile'] = ddmRet['DM_Decile'].astype(int)

        # df['cutoff_flag'] = df['exchcd']
        # ddny = bins3(df,10,'ret11mLag2','period',True)

        # mktVal = ddny.groupby(['period', 'bin'])['lagme'].sum()
        # mktVal = mktVal.reset_index()
        # mktVal.rename(columns={'lagme': 'lagMV'}, inplace=True)

        # ddnym = pd.merge(dd.copy(), mktVal, on = ['period', 'bin'], how = 'left')

        # # Calculate value wtd return
        # ddnym['wgt'] = ddnym['lagme'] / ddnym['lagMV']
        # ddnym['wtdRet'] = ddnym['tret'] * ddnym['wgt']

        # # Add valued wtd returns to calculate total vw ret
        # ddnymRet = ddnym.groupby(['period', 'bin'])['wtdRet'].sum()
        # ddnymRet = ddnymRet.reset_index()
        # ddnymRet = ddnymRet.rename(columns={
        #     'period': 'date',  # Change 'period' to 'new_period_name'
        #     'bin': 'decile',        # Change 'bin' to 'new_bin_name'
        #     'wtdRet': 'KRF_Ret' # Change 'mvWtdRet' to 'new_mvWtdRet_name'
        # })

        # ddAll = pd.merge(ddnymRet, ddmRet, on = ['date', 'decile'], how = 'left')

        # Save to pickle file
        ddmRet.to_pickle('momDecilesRet.pkl')
    else:
        ddmRet = pd.read_pickle('momDecilesRet.pkl')

    return ddmRet

def calcExRet(df, return_col, isDatePeriod = False):
    """
    Adjust the return column by subtracting the risk-free rate from the Fama-French dataset.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the necessary columns.
    return_col (str): Column name for the returns to be adjusted.

    Returns:
    pd.DataFrame: DataFrame with the adjusted return column and without the risk-free rate column.
    """
    # Load Fama-French risk-free rate data
    rf_reader = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1926', end='2024')
    rf = rf_reader.read()[0] / 100
    
    # Reset index and prepare for merging
    rf.reset_index(inplace=True)
    rf.rename(columns={'Date': 'date'}, inplace=True)
    rf.drop(columns=['SMB', 'Mkt-RF', 'HML'], inplace=True)
    
    # Merge the risk-free rate with the input DataFrame
    df = pd.merge(df, rf, on='date', how='left')
    
    # Adjust the return column by subtracting the risk-free rate
    df[return_col] = df[return_col] - df['RF']
    
    # Remove the risk-free rate column
    df.drop(columns=['RF'], inplace=True)
    
    return df

# Main function
def main():
    df = assignMomDeciles(reload = False)
    print(df)

# Standard way to call the main function
if __name__ == "__main__":
    main()  