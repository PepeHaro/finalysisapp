import numpy as np
import pandas as pd
import requests
import csv
from scipy import stats
import statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import seaborn as sns
import plotly.express as px
from finvizfinance.quote import finvizfinance
import plotly.graph_objects as go
from urllib.parse import quote
import plotly.express as px
from PIL import Image
from io import BytesIO
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator
import datetime
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os



# Obtener los tokens desde los secretos de Streamlit
try:
    token1 = st.secrets["token1"]
    token2 = st.secrets["token2"]
except KeyError as e:
    raise ValueError(f"No se encontró el token {e} en los secretos de Streamlit")

# Verificar si los tokens se cargaron correctamente
if token1 is None:
    raise ValueError("No se encontró el token1 en los secretos de Streamlit")
if token2 is None:
    raise ValueError("No se encontró el token2 en los secretos de Streamlit")
#FOTO
ruta_foto = "https://raw.githubusercontent.com/PepeHaro/finalysisapp/master/Fot.jpg"
st.image(ruta_foto,caption= "José Federico Haro Velasco",use_column_width=True)
st.sidebar.title("EXPLORE:bar_chart:")
page = st.sidebar.selectbox("Select one", ["Company Overview","Sector Dashboard","Implied Volatility","News"])

# COMPANY OVERVIEW
if page == "Company Overview":
    st.header("Welcome to the Financial Analysis APP")
    st.write("Get a detailed view of any listed company by entering its ticker, including description, key data and performance charts.")

    # INFORMACIÓN DE LA EMPRESA
    st.markdown("## Company Overview :chart_with_upwards_trend:")
    # Campo de texto para que el usuario ingrese el ticker
    ticker = st.text_input("Enter a stock ticker (e.g., NVDA)").strip()

    if ticker:
        try:
            company = yf.Ticker(ticker)
            info = company.info

            # Mostrar la información disponible para depuración
            #st.write("Información completa:", info)

            # Obtener el precio en tiempo real y los datos de hoy
            todays_data = company.history(period='1d')
            current_price = todays_data['Close'][0]
            open_price = todays_data['Open'][0]
            price_change = current_price - open_price
            percent_change = (price_change / open_price) * 100

            # Determinar el color y el signo del cambio
            if price_change >= 0:
                change_color = "green"
                sign = "+"
            else:
                change_color = "red"
                sign = ""

            # Mostrar el precio en tiempo real junto con el cambio diario
            st.markdown(f"#### Precio actual de {ticker}: ${current_price:.2f} <span style='color:{change_color}; font-size:0.9em;'>{sign}{price_change:.2f} ({sign}{percent_change:.2f})% hoy</span>", unsafe_allow_html=True)

            # Organizar la disposición en dos columnas
            col1, col2 = st.columns([1, 2])

            # Mostrar los otros datos en la columna izquierda
            with col1:
                st.subheader(info.get("longName", "Nombre de la empresa no disponible"))

                # Mostrar el logo
                website = info.get('website')
                if website:
                    logo_url = f"https://logo.clearbit.com/{website.replace('http://','').replace('https://','').split('/')[0]}"
                    response = requests.get(logo_url)
                    if response.status_code == 200:
                        logo_image = Image.open(BytesIO(response.content))
                        st.image(logo_image)

                # Mostrar datos solo si están disponibles
                st.write(f"· WebPage: {website if website else 'N/A'}")
                st.write(f"· Industry: {info.get('industry', 'N/A')}")
                st.write(f"· Sector: {info.get('sector', 'N/A')}")
                st.write(f"· Total Income: ${'{:,.0f}'.format(info['totalRevenue']) if 'totalRevenue' in info and isinstance(info['totalRevenue'], (int, float)) else 'N/A'}")
                st.write(f"· MarketCap: ${'{:,.0f}'.format(info['marketCap']) if 'marketCap' in info and isinstance(info['marketCap'], (int, float)) else 'N/A'}")
                st.write(f"· Total Debt: ${'{:,.0f}'.format(info['totalDebt']) if 'totalDebt' in info and isinstance(info['totalDebt'], (int, float)) else 'N/A'}")
                st.write(f"· Total Cash: ${'{:,.0f}'.format(info['totalCash']) if 'totalCash' in info and isinstance(info['totalCash'], (int, float)) else 'N/A'}")
                st.write(f"· Net Income: ${'{:,.0f}'.format(info['netIncomeToCommon']) if 'netIncomeToCommon' in info and isinstance(info['netIncomeToCommon'], (int, float)) else 'N/A'}")
                st.write(f"· Operating Margin: {info['operatingMargins']*100:.2f}%" if 'operatingMargins' in info and isinstance(info['operatingMargins'], (int, float)) else 'N/A')
                st.write(f"· P/E: {info['trailingPE']:.2f}" if 'trailingPE' in info and isinstance(info['trailingPE'], (int, float)) else 'N/A')
                st.write(f"· Number of employees: {info['fullTimeEmployees']:,}" if 'fullTimeEmployees' in info and isinstance(info['fullTimeEmployees'], (int, float)) else 'N/A')

            # Mostrar la descripción de la empresa en la columna derecha
            with col2:
                description = info.get('longBusinessSummary', 'Descripción de la empresa no disponible')
                st.write(description)

            st.markdown("---")
            st.header("FINANCIAL STATEMENTS")

            # Income Statement
            financials = company.financials
            if financials is not None and 'Total Revenue' in financials.index and 'Net Income' in financials.index:
                financial_data = financials.loc[['Total Revenue', 'Net Income']].tail(5)
                financial_data.columns = [str(year.year) for year in financial_data.columns]  # Asegurar que las columnas son cadenas

                # Crear gráfica con Plotly
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=financial_data.columns,
                    y=financial_data.loc['Total Revenue'],
                    name='Total Revenue',
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    x=financial_data.columns,
                    y=financial_data.loc['Net Income'],
                    name='Net Income',
                    marker_color='indigo'
                ))

                fig.update_layout(
                    title='REVENUE AND NET INCOME',
                    xaxis_tickmode='array',
                    xaxis_tickvals=financial_data.columns,
                    xaxis_ticktext=[str(year) for year in financial_data.columns],
                    yaxis=dict(
                        title='Amount ($)',
                        tickprefix='$',
                        ticks='outside',
                        tickformat=',.0f'
                    ),
                    barmode='group',
                    legend_title_text='Metric'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Financial data for Total Revenue and/or Net Income is not available.")

            # Sección de Activos y Pasivos
            balance_sheet = company.balance_sheet
            if 'Total Assets' in balance_sheet.index and 'Total Debt' in balance_sheet.index:
                balance_data = balance_sheet.loc[['Total Assets', 'Total Debt']].tail(5)
                balance_data.columns = balance_data.columns.year  # Cambia las columnas a años

                # Crear otra gráfica con Plotly para Activos y Pasivos
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=balance_data.columns,
                    y=balance_data.loc['Total Assets'],
                    name='Total Assets',
                    marker_color='gray'
                ))
                fig.add_trace(go.Bar(
                    x=balance_data.columns,
                    y=balance_data.loc['Total Debt'],
                    name='Total Debt',
                    marker_color='red'
                ))

                fig.update_layout(
                    title='ASSETS AND LIABILITIES',
                    xaxis_tickmode='array',
                    xaxis_tickvals=balance_data.columns,
                    xaxis_ticktext=[str(year) for year in balance_data.columns],
                    yaxis=dict(
                        title='Amount ($)',
                        tickprefix='$',
                        ticks='outside',
                        tickformat=',.0f'
                    ),
                    barmode='group',
                    legend_title_text='Metric'
                )
                st.plotly_chart(fig, use_container_width=True)    
            else:
                st.write(f"Balance Sheet data for Assets and/or Liabilities of {ticker} is not available.")

            # Free Cash Flow Section
            cash_flow = company.cashflow
            if 'Free Cash Flow' in cash_flow.index:
                cash_flow_data = cash_flow.loc['Free Cash Flow'].tail(5)
                years = cash_flow.columns.year  # Capturamos los años en las columnas para usarlos como eje x

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(year) for year in years],  # Convertimos los años a strings si es necesario
                    y=cash_flow_data,
                    name='Free Cash Flow',
                    marker_color='purple'
                ))

                fig.update_layout(
                    title='FREE CASH FLOW',
                    xaxis=dict(
                        title='',
                        tickmode='array',
                        tickvals=[str(year) for year in years],  # Usamos los años como etiquetas del eje x
                    ),
                    yaxis=dict(
                        title='Amount ($)',
                        tickprefix='$',
                        ticks='outside',
                        tickformat=',.0f'
                    ),
                    barmode='group',
                    legend_title_text='Metric'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"Free Cash Flow data for {ticker} is not available.")

        except Exception as e:
            st.warning(f"Please type the correct ticker. Error: {str(e)}")
            
#SECTOR DASHBOARD
if page == "Sector Dashboard":
    # Configuración inicial
    FinViz_Structure = {
        'Overview': '111',
        'Valuation': '121',
        'Financial': '161',
        'Ownership': '131',
        'Performance': '141',
        'Technical': '171'
    }

    sectores_disponibles = {
        'Any': '',
        'Basic Materials': 'sec_basicmaterials',
        'Communication Services': 'sec_communicationservices',
        'Consumer Cyclical': 'sec_consumercyclical',
        'Consumer Defensive': 'sec_consumerdefensive',
        'Energy': 'sec_energy',
        'Financial': 'sec_financial',
        'Healthcare': 'sec_healthcare',
        'Industrials': 'sec_industrials',
        'Real Estate': 'sec_realestate',
        'Technology': 'sec_technology',
        'Utilities': 'sec_utilities'
    }

    End_Point_1 = "https://elite.finviz.com/export.ashx?v="
    
    # Guardar el último sector descargado
    last_downloaded_sector = st.session_state.get('last_downloaded_sector', '')

    st.markdown("""
    ## DOWNLOAD SECTOR-BASED DATA 	:classical_building:

    In this section, you can download detailed financial data from Finviz based on the sector you select. This data pertains to companies listed on NASDAQ and NYSE that are part of the S&P 500 index. You can filter the companies by sectors such as Technology, Healthcare, Financial, and more.

    The data available for download includes companies of various market capitalizations, from small-cap to large-cap, which are optionable. This feature is particularly useful for investors looking to analyze different sectors within the major U.S. stock exchanges for investment opportunities or market research.

    Please select the sector from the dropdown menu and click on 'Download Data' to begin the download of sector-specific financial data.
    """)
    st.markdown("---")
    # Selector de sector
    st.markdown("### Select the sector and category")
    selected_sector = st.selectbox("Sector", list(sectores_disponibles.keys()))

    # Selección de categoría y visualización de datos
    selected_category = st.selectbox("Category", list(FinViz_Structure.keys()))

    # Descarga de datos
    if st.button("Download Data"):
        st.write("Wait a moment...:clock3:")

        downloaded_successfully = True
        sector_filter = sectores_disponibles[selected_sector] if selected_sector != 'Any' else ''

        for key, value in FinViz_Structure.items():
            url = f"{End_Point_1}{value}&f=cap_largeover|cap_midunder,exch_nyse|nasd,idx_sp500,{sector_filter},sh_opt_option&auth={token1}"
            response = requests.get(url)
            if response.status_code == 200:
                filename = f"{key}.csv"
                with open(filename, "wb") as file:
                    file.write(response.content)
            else:
                downloaded_successfully = False

            time.sleep(2)  # Pausa para evitar limitaciones de la API

        if downloaded_successfully:
            st.session_state['last_downloaded_sector'] = selected_sector
            st.success(f"The data of the {selected_sector} sector  has been downloaded successfully")
            

    # Mostrar datos si el sector coincide, usamos st.session_state.get
    if selected_category and (selected_sector == st.session_state.get('last_downloaded_sector', '')):
        filename = f"{selected_category}.csv"
        data = pd.read_csv(filename, index_col='No.')
        st.markdown(f"### DASHBOARD: {selected_sector}:chart:")

        # Mostrar métricas según el ticker seleccionado
        selected_ticker = st.selectbox("Select Ticker", data['Ticker'].unique())
        selected_ticker_data = data[data['Ticker'] == selected_ticker]

        if selected_ticker_data.empty:
            st.warning("No data available for the selected ticker.")
        else:

            # Configurar la cuadrícula para el dashboard
            col1, col2, col3 = st.columns(3)
            # Mostrar las métricas específicas para cada categoría
            if selected_category == 'Overview':
                col1.metric("Market Cap", value=f"${selected_ticker_data['Market Cap'].iloc[0]:,.2f}")
                col2.metric("Price", value=f"${selected_ticker_data['Price'].iloc[0]:,.2f}")
                
                # Obtener el valor del cambio y definir el color correspondiente
                change_value = selected_ticker_data['Change'].iloc[0]
                if '%' in change_value:
                    change_value_float = float(change_value.strip('%'))
                    if change_value_float > 0:
                        change_color = "green"
                    elif change_value_float < 0:
                        change_color = "red"
                    else:
                        change_color = "black"  # Si es 0, no se aplica ningún color
                else:
                    change_color = "black"  # Por defecto, si no contiene '%', no se aplica ningún color
                # Usé HTML para mostrar la palabra "Change" y el cambio con el color correspondiente
                col3.markdown(f"<div style='display: flex; flex-direction: column; align-items: center;'><span>Change</span><span style='color: {change_color}; font-weight: bold; font-size: 24px;'>{change_value}</span></div>", unsafe_allow_html=True)
                st.markdown("---")
                col1.metric("P/E Ratio", value=selected_ticker_data['P/E'].iloc[0])
                col2.metric("Volume", value=f"{selected_ticker_data['Volume'].iloc[0]:,.0f}")

            elif selected_category == 'Valuation':
                col1.metric("Market Cap", value=f"${selected_ticker_data['Market Cap'].iloc[0]:,.2f}")
                col2.metric("P/E Ratio", value=selected_ticker_data['P/E'].iloc[0])
                col3.metric("Forward P/E", value=selected_ticker_data['Forward P/E'].iloc[0])
                st.markdown("---")
                col1.metric("PEG", value=selected_ticker_data['PEG'].iloc[0])
                col2.metric("P/S", value=selected_ticker_data['P/S'].iloc[0])
                col3.metric("P/B", value=selected_ticker_data['P/B'].iloc[0])

            elif selected_category == 'Financial':
                col1.metric("Dividend Yield", value=selected_ticker_data['Dividend Yield'].iloc[0])
                col2.metric("ROA", value=selected_ticker_data['Return on Assets'].iloc[0])
                col3.metric("ROE", value=selected_ticker_data['Return on Equity'].iloc[0])
                st.markdown("---")
                col1.metric("Gross Margin", value=selected_ticker_data['Gross Margin'].iloc[0])
                col2.metric("Operating Margin", value=selected_ticker_data['Operating Margin'].iloc[0])
                col3.metric("Profit Margin", value=selected_ticker_data['Profit Margin'].iloc[0])

            elif selected_category == 'Ownership':
                col1.metric("Insider Ownership", value=selected_ticker_data['Insider Ownership'].iloc[0])
                col2.metric("Inst Ownership", value=selected_ticker_data['Institutional Ownership'].iloc[0])
                col3.metric("Float Short", value=selected_ticker_data['Float Short'].iloc[0])
                st.markdown("---")
                col1.metric("Short Ratio", value=selected_ticker_data['Short Ratio'].iloc[0])
                col2.metric("Avg Volume", value=selected_ticker_data['Average Volume'].iloc[0])

            elif selected_category == 'Performance':
                col1.metric("Performance (Week)", value=selected_ticker_data['Performance (Week)'].iloc[0])
                col2.metric("Performance (Month)", value=selected_ticker_data['Performance (Month)'].iloc[0])
                col3.metric("Performance (Quarter)", value=selected_ticker_data['Performance (Quarter)'].iloc[0])

            elif selected_category == 'Technical':
                col1.metric("Beta", value=selected_ticker_data['Beta'].iloc[0])
                col2.metric("ATR", value=selected_ticker_data['Average True Range'].iloc[0])
                col3.metric("SMA20", value=selected_ticker_data['20-Day Simple Moving Average'].iloc[0])

# GRÁFICAS DEPENDIENDO LA CATEGORÍA
    if selected_category and (selected_sector == st.session_state.get('last_downloaded_sector', '')):
        filename = f"{selected_category}.csv"
        data = pd.read_csv(filename, index_col='No.')

        if selected_category == 'Overview':
            # Gráfico de Dispersión para Overview
            if 'P/E' in data.columns and 'Market Cap' in data.columns and 'Volume' in data.columns and 'Sector' in data.columns:
                fig = px.scatter(data, x='P/E', y='Market Cap', color='Sector',
                                size='Volume', hover_name='Company',
                                labels={'P/E': 'Price to Earnings Ratio', 'Market Cap': 'Market Capitalization ($)'},
                                title='Market Cap vs. P/E Ratio by Sector')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### *Please ensure the data includes 'P/E', 'Market Cap', 'Volume', and 'Sector' columns to display the scatter plot*")

        elif selected_category == 'Valuation':
            # Gráfica de burbujas para Valuation
            if 'P/E' in data.columns and 'EPS growth next year' in data.columns and 'Market Cap' in data.columns:
                fig = px.scatter(data, x='P/E', y='EPS growth next year',
                                size='Market Cap', color='Market Cap',
                                hover_name='Ticker', log_x=True, size_max=40,
                                title='P/E Ratio vs. Expected EPS Next Year',
                                labels={'P/E': 'P/E Ratio', 'EPS next Y': 'EPS growth next year'},
                                range_x=[data['P/E'].min(), data['P/E'].max()])  # Ajustar el rango de X

                # Añadir línea punteada horizontal en y=0
                fig.add_shape(type='line',
                            x0=data['P/E'].min(), x1=data['P/E'].max(),
                            y0=0, y1=0,
                            line=dict(color='Red', width=2, dash='dash'),
                            xref='x', yref='y')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### *Please download the data to see the bubble chart*")

        elif selected_category == 'Financial':
            # Gráfico de Barras para el Dividend Yield
            if 'Dividend Yield' in data.columns and 'Ticker' in data.columns:
                fig = px.bar(data, x='Ticker', y='Dividend Yield', title='Dividend Yield by Company',
                            labels={'Dividend Yield': 'Dividend Yield', 'Ticker': 'Ticker'},
                            color='Dividend Yield', color_continuous_scale=px.colors.sequential.Inferno)  
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### *Please ensure the data includes 'Dividend Yield' and 'Ticker' columns to display the chart*")


        elif selected_category == 'Ownership':
            # Gráfico de Barras Apiladas: Proporción de Propiedad Institucional y de Insiders
            if 'Insider Ownership' in data.columns and 'Institutional Ownership' in data.columns:
                # Preparar los datos
                data['Insider Ownership'] = pd.to_numeric(data['Insider Ownership'].str.replace('%', ''), errors='coerce')
                data['Institutional Ownership'] = pd.to_numeric(data['Institutional Ownership'].str.replace('%', ''), errors='coerce')
                
                fig = px.bar(data, x='Ticker', y=['Insider Ownership', 'Institutional Ownership'],
                            title="Ownership Structure by Company",
                            labels={'value': 'Ownership Percentage', 'variable': 'Type of Owner'},
                            barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### *Please ensure the data has 'Insider Own' and 'Inst Own' columns to display the chart*")

        elif selected_category == 'Performance':
            # Configurar y mostrar gráfica para Technical
            pass  # Aquí colocarás el código específico para la gráfica de Performance


        elif selected_category == 'Technical':
                # Gráfico de Dispersión con Bubble: Beta vs RSI con tamaño de burbuja constante
                if 'Beta' in data.columns and 'Relative Strength Index (14)' in data.columns:
                    fig = px.scatter(data, x='Beta', y='Relative Strength Index (14)',
                                    size_max=10,  # Tamaño constante para todas las burbujas
                                    color='Ticker', hover_name='Ticker',
                                    labels={'Beta': 'Beta', 'Relative Strength Index (14)': 'RSI'},
                                    title='Beta vs RSI with Constant Bubble Size')

                    # Añadir línea punteada vertical en x=1
                    fig.add_shape(
                        type="line", 
                        x0=1, y0=0, x1=1, y1=data['Relative Strength Index (14)'].max(),
                        line=dict(color="Red", width=2, dash="dash"),
                        xref="x", yref="y"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.write("*An RSI above 70 is a sign that the financial asset is overbought, and below 30 is a sign that it is oversold.")
                else:
                    st.error("Required columns (Beta, RSI) for plotting the bubble chart are missing.")

        # TREEMAP MARKET CAP
        st.markdown("---")
        st.markdown(f"### MARKET CAP: {selected_sector} sector")
        if 'data' in locals() and 'Market Cap' in data.columns:
            fig = px.treemap(data, path=[px.Constant(f"{selected_sector}"), 'Ticker'], values='Market Cap',
                color='Market Cap', hover_data=['Ticker'],
                color_continuous_scale='RdBu',
                title='')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("*Please download the data of the sector to see the treemap*")

# IMPLIED VOLATILITY
if page == "Implied Volatility":
    st.markdown("## IMPLIED VOLATILITY:chart_with_upwards_trend:")
    st.write("""
    Welcome to our Implied Volatility Analysis section, where we delve into the fluctuations 
    of implied volatility over the past year for 103 prominent U.S. companies, using data 
    provided by TastyLive. This area focuses on understanding how market expectations 
    reflected through options' implied volatility can shed light on risk perceptions and 
    investment opportunities in the current landscape. Dive into our interactive 
    visualizations to uncover patterns and trends that will help you make more informed 
    investment decisions.
    """)

    # URL del archivo CSV
    url = "https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv"

    # Leer datos del archivo CSV
    df = pd.read_csv(url)

    # Convertir la columna de tiempo a formato datetime
    df['time'] = pd.to_datetime(df['time'])

    # Filtrar datos para el período desde mayo de 2023 hasta mayo de 2024
    start_date = datetime.datetime(2023, 5, 1)
    end_date = datetime.datetime(2024, 5, 31)
    df_filtered = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

    # Extraer valores únicos del símbolo
    unique_symbols = df_filtered['Symbol'].unique()

    st.markdown("## Stock Price")
    # Crear un cuadro de selección para elegir un símbolo
    selected_symbol = st.selectbox('Select a symbol:', unique_symbols)

    # Filtrar datos para el símbolo seleccionado
    selected_data = df_filtered[df_filtered['Symbol'] == selected_symbol]

    # Calcular las bandas de Bollinger
    indicator_bb = BollingerBands(selected_data['close'], window=20, window_dev=2)
    selected_data['bb_upper'] = indicator_bb.bollinger_hband()
    selected_data['bb_lower'] = indicator_bb.bollinger_lband()

    # Calcular las medias móviles SMA20, SMA50, SMA200
    indicator_sma20 = SMAIndicator(selected_data['close'], window=20)
    selected_data['sma20'] = indicator_sma20.sma_indicator()
    indicator_sma50 = SMAIndicator(selected_data['close'], window=50)
    selected_data['sma50'] = indicator_sma50.sma_indicator()
    indicator_sma200 = SMAIndicator(selected_data['close'], window=200)
    selected_data['sma200'] = indicator_sma200.sma_indicator()

    # Crear el gráfico de velas
    fig_candles = go.Figure(data=[go.Candlestick(x=selected_data['time'],
                                                 open=selected_data['open'],
                                                 high=selected_data['high'],
                                                 low=selected_data['low'],
                                                 close=selected_data['close'],
                                                 name='Candlestick'),
                                  go.Scatter(x=selected_data['time'], y=selected_data['bb_upper'], 
                                             mode='lines', name='Bollinger Upper Band', line=dict(color='blue')),
                                  go.Scatter(x=selected_data['time'], y=selected_data['bb_lower'], 
                                             mode='lines', name='Bollinger Lower Band', line=dict(color='blue')),
                                  go.Scatter(x=selected_data['time'], y=selected_data['sma20'], 
                                             mode='lines', name='SMA20', line=dict(color='green')),
                                  go.Scatter(x=selected_data['time'], y=selected_data['sma50'], 
                                             mode='lines', name='SMA50', line=dict(color='orange')),
                                  go.Scatter(x=selected_data['time'], y=selected_data['sma200'], 
                                             mode='lines', name='SMA200', line=dict(color='red'))])

    fig_candles.update_layout(title=f'Candlestick Chart of: {selected_symbol}',
                              xaxis_title='Date',
                              yaxis_title='Price',
                              template='plotly_dark')

    # Mostrar el gráfico de velas
    st.plotly_chart(fig_candles, use_container_width=True)
    # Calcular el MACD
    macd = MACD(selected_data['close'], window_slow=26, window_fast=12, window_sign=9)
    selected_data['macd'] = macd.macd()
    selected_data['macd_signal'] = macd.macd_signal()
    selected_data['macd_diff'] = macd.macd_diff()

    st.markdown("## MACD")
    # Crear el gráfico MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=selected_data['time'], y=selected_data['macd'], mode='lines', name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=selected_data['time'], y=selected_data['macd_signal'], mode='lines', name='Signal', line=dict(color='orange')))
    fig_macd.add_trace(go.Bar(x=selected_data['time'], y=selected_data['macd_diff'], name='MACD Diff', marker_color='grey'))

    fig_macd.update_layout(title=f'MACD of: {selected_symbol}',
                           xaxis_title='Date',
                           yaxis_title='MACD',
                           template='plotly_dark')

    # Mostrar el gráfico MACD
    st.plotly_chart(fig_macd, use_container_width=True)
    
#PRECIOS NORMALIZADOS
    # Filtrar datos para el período desde mayo de 2023 hasta mayo de 2024
    start_date = datetime.datetime(2023, 5, 8)
    end_date = datetime.datetime(2024, 5, 6)
    df_filtered = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

    # Obtener tickers únicos para el gráfico de línea múltiple
    unique_symbols_line = df_filtered['Symbol'].unique()


    st.markdown("### Normalized Price Fluctuation")    
    # Crear un cuadro de selección para elegir múltiples símbolos
    selected_symbols_line = st.multiselect('Select symbols:', unique_symbols_line)

    # Filtrar datos para los símbolos seleccionados
    selected_data_line = df_filtered[df_filtered['Symbol'].isin(selected_symbols_line)]

    # Normalizar los precios de cierre para cada símbolo con base 0
    normalized_prices = selected_data_line.groupby('Symbol')['close'].transform(lambda x: (x - x.iloc[0]) / x.iloc[0])

    # Crear el gráfico de líneas para los símbolos seleccionados
    fig_line_chart = px.line(selected_data_line, x='time', y=normalized_prices, color='Symbol', title='', template='plotly_dark')

     # Agregar etiquetas a los ejes
    fig_line_chart.update_layout(yaxis_title='% Change from Initial Price')

    # Mostrar el gráfico de líneas
    st.plotly_chart(fig_line_chart, use_container_width=True)

    # Eliminar filas con valores NaN en alguna de las columnas y valores infinitos
    df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Calcular los rendimientos diarios de los activos
    df['Returns'] = df.groupby('Symbol')['close'].pct_change()

    # Calcular el Alpha anual y el riesgo anual
    annual_alpha = df.groupby('Symbol')['Returns'].mean() * 252 * 100  # Convertir a porcentaje (multiplicar por 100)
    annual_std = df.groupby('Symbol')['Returns'].std() * np.sqrt(252) * 100  # Convertir a porcentaje (multiplicar por 100)

    # Crear un DataFrame con los resultados
    annual_metrics_df = pd.DataFrame({'Symbol': annual_alpha.index, 'Annual Alpha': annual_alpha.values, 'Annual Risk': annual_std.values})

    # Crear un gráfico interactivo con Plotly Express
    fig = px.scatter(annual_metrics_df, x='Annual Risk', y='Annual Alpha', text='Symbol',
                    title='Annual Alpha vs Annual Risk',
                    labels={'Annual Risk': 'Annual Risk (%)', 'Annual Alpha': 'Annual Alpha (%)', 'Symbol': 'Símbolo'},
                    template='plotly_dark')

    # Agregar línea punteada en el eje Y en el valor 0
    fig.add_hline(y=0, line_dash='dot', line_color='green')

    # Mostrar el gráfico interactivo
    st.plotly_chart(fig)

#IMPLIED VOL, IVRANK, IVPERCENTILE, IVX

    # Calcular el IV Rank
    current_iv = selected_data['impVolatility'].iloc[-1]  # Volatilidad implícita actual
    iv_max = selected_data['impVolatility'].max()  # Volatilidad implícita máxima en el período
    iv_min = selected_data['impVolatility'].min()  # Volatilidad implícita mínima en el período
    iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100  # Fórmula para calcular IV Rank

    # Verificar si hay suficientes datos para calcular el IV Percentil
    if len(selected_data['impVolatility']) > 0:
        # Calcular el IV Percentil
        sorted_data = np.sort(selected_data['impVolatility'])
        idx = np.searchsorted(sorted_data, current_iv)
        iv_percentile = (idx + (current_iv - sorted_data[idx-1]) / (sorted_data[idx] - sorted_data[idx-1])) / len(sorted_data) * 100
    else:
        iv_percentile = np.nan

    # Calcular el IVX
    ivx = selected_data['impVolatility'].rolling(window=20).mean().iloc[-1]  # Media móvil de la volatilidad implícita
    
    st.markdown(f"### IMPLIED VOLATILITY: {selected_symbol}")
    # Mostrar los IV Metrics al lado del gráfico
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write(f"	##### :arrow_forward:**IV Rank:** {iv_rank:.2f}")
    with col2:
        st.write(f" ##### :arrow_forward:**IV Percentile:** {iv_percentile:.2f}")
    with col3:
        st.write("")
    # Crear el gráfico de volatilidad implícita
    fig_volatility = go.Figure(data=go.Scatter(x=selected_data['time'], y=selected_data['impVolatility'], mode='lines', name='Implied Volatility'))
    fig_volatility.update_layout(title='',
                                 xaxis_title='Date',
                                 yaxis_title='Implied Volatility',
                                 template='plotly_dark')

    # Mostrar el gráfico de volatilidad implícita
    st.plotly_chart(fig_volatility, use_container_width=True)
    # Agregar explicación abajo del gráfico
    st.write("**Interpretation:**")
    st.markdown(":arrow_forward:**IV RANK:** It's a measure indicating the current position of implied volatility relative to its historical range, expressed as a value between 0 and 100 without a percentage symbol. The higher the IV Rank, the higher the implied volatility compared to its recent history.")
    st.write(":arrow_forward:**IV PERCENTILE:** It represents the percentage of days during a specified period when implied volatility was lower than the current implied volatility. It's expressed as a percentage, where an IV Percentile of 50% would mean that implied volatility has been lower than the current value approximately half of the time during the selected period.")


#VIX PLOT
# Descargar los datos del VIX
    # Obtener la fecha actual
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Descargar los datos del VIX con la fecha actual como end date
    vix_data = yf.download("^VIX", start="2023-05-08", end=end_date)

    # Obtener el último precio del VIX
    latest_vix_price = vix_data['Close'].iloc[-1]

    # Crear el gráfico del VIX con las líneas de Low y High
    st.markdown(f"### Volatility Index (^VIX): {latest_vix_price:.2f}")
    fig_vix = go.Figure()

    # Añadir la línea de 'Close' del VIX en rojo
    fig_vix.add_trace(go.Scatter(x=vix_data.index, y=vix_data['Close'], mode='lines', name='VIX Close', line=dict(color='red')))

    # Añadir la línea de 'Low' del VIX en verde
    fig_vix.add_trace(go.Scatter(x=vix_data.index, y=vix_data['Low'], mode='lines', name='VIX Low', line=dict(color='green')))

    # Añadir la línea de 'High' del VIX en rojo oscuro
    fig_vix.add_trace(go.Scatter(x=vix_data.index, y=vix_data['High'], mode='lines', name='VIX High', line=dict(color='darkred')))

    # Actualizar el layout del gráfico
    fig_vix.update_layout(title='VIX Price with High and Low',
                        xaxis_title='Date',
                        yaxis_title='VIX Price',
                        template='plotly_dark')

    # Mostrar el gráfico del VIX
    st.plotly_chart(fig_vix, use_container_width=True)



#REGRESION 
    st.markdown("### Implied Vol vs Price regression")
    # Crear otro cuadro de selección para elegir otro símbolo
    selected_symbol_2 = st.selectbox('Select a ticker:', unique_symbols)

    # Filtrar los datos relevantes para el segundo símbolo seleccionado
    selected_data_2 = df_filtered[df_filtered['Symbol'] == selected_symbol_2]

    # Descargar los datos del VIX para el período correspondiente
    vix_data = yf.download("^VIX", start=start_date, end=end_date)

    # Fusionar los datos de la volatilidad implícita y el VIX
    merged_data = pd.merge(selected_data_2, vix_data, how='inner', left_on='time', right_index=True)

    # Eliminar filas con valores faltantes
    merged_data.dropna(subset=['impVolatility', 'Close'], inplace=True)

    # Separar las características (features) y la variable objetivo
    X = merged_data[['Close']]  # VIX como característica
    y = merged_data['impVolatility']  # Volatilidad implícita como variable objetivo

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un modelo de regresión lineal
    modelo = LinearRegression()

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular R cuadrada
    r2 = modelo.score(X_test, y_test)

    # Calcular el coeficiente Beta y el intercepto Alpha
    beta = modelo.coef_[0]
    alpha = modelo.intercept_

    # Mostrar las métricas usando st.columns()
    col1, col2, col3 = st.columns(3)
    col1.write("R2:")
    col1.write(round(r2, 4))
    col2.write("Beta:")
    col2.write(round(beta, 4))
    col3.write("Alpha:")
    col3.write(round(alpha, 4))

    # Crear la figura en 3D
    fig_3d = px.scatter_3d(merged_data, x='Close', y='Close', z='impVolatility', title='3D Scatter plot with regression plane')
    fig_3d.add_traces(go.Scatter3d(x=X_test.squeeze(), y=X_test.squeeze(), z=predicciones, mode='lines', name='Regression Plane'))
    fig_3d.update_layout(scene=dict(xaxis_title='Vix Close Price', yaxis_title='VIX Price', zaxis_title='Implied Volatility'))
    st.plotly_chart(fig_3d)

#COMPANY NEWS
if page == "News":
    st.markdown("# Financial News:newspaper:")
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", key='news_ticker')
    
    if ticker:
        # Asegurándose de que el query está codificado correctamente para URL
        search_query = quote(ticker)
        url = f'https://newsapi.org/v2/everything?q={search_query}&apiKey={token2}'

        response = requests.get(url)
        news = response.json()

        # Comprobar si la respuesta contiene artículos
        if response.status_code == 200 and 'articles' in news:
            articles = news['articles']
            if articles:
                st.subheader(f"Latest News for {ticker}")
                for article in articles:
                    with st.container():
                        col1, col2 = st.columns([1, 4])  # Ajusta la proporción según sea necesario
                        with col1:
                            if article['urlToImage']:  # Verificar si hay una imagen disponible
                                st.image(article['urlToImage'], use_column_width=True)
                        with col2:
                            st.markdown(f"#### [{article['title']}]({article['url']})")
                            st.write(article['publishedAt'])
                            st.write(article['description'])
            else:
                st.write("No news articles found for this ticker.")
        else:
            # Manejo de errores o problemas en la respuesta
            error_message = news.get('message', 'Failed to fetch news without a specific error message.')
            st.error(f"Error fetching news: {error_message}")

        url_sources = "https://newsapi.org/v2/sources?apiKey=" + token2
        response_sources = requests.get(url_sources)
        data_sources = response_sources.json()
        print(data_sources)

