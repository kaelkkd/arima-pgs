import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import numpy as np
import os
import time
import multiprocessing
import platform
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from joblib import Parallel, delayed
from utils import verify_stationarity, set_plot_style

class Arima:
    def __init__(self, p_upper, q_upper, ticker, start_date, end_date):
        self.p_upper = p_upper
        self.q_upper = q_upper
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
    def prepare_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if df.empty:
            raise ValueError(f"Não foi encontrado dados de {self.ticker} entre {self.start_date} e {self.end_date}")
        #index como datatimeindex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        closing_prices = df["Close"].copy()
        split = int(len(closing_prices) * 0.8)
        train, test = closing_prices.iloc[:split], closing_prices.iloc[split:]
        
        return train, test
    
    def evaluate_arima_order(self, order, data):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                model = ARIMA(data, order=order)
                model_fit = model.fit()
            return (order, model_fit.aic)
        except Exception:
            return (order, float('inf'))
    
    def evaluate_chunk(self, chunk, data):
        results = []
        for order in chunk:
            results.append(self.evaluate_arima_order(order, data))

        return results

    def select_best_order( self, train_data, lower_bound=0, upper_bound=0, n_jobs=-1, backend="loky", chunk_size=1):
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        range_p = range(lower_bound, upper_bound + 1)
        range_q = range(lower_bound, upper_bound + 1)
        d = range(0, verify_stationarity(train_data) + 1)
        parameter_grid = list(product(range_p, d, range_q))
        print(f"Iniciando grid search com {len(parameter_grid)} combinações (n_jobs={n_jobs}, chunk_size={chunk_size})...")
        data_arr = np.asarray(train_data)

        # chunking
        if chunk_size <= 1:
            # paralelismo normal: uma ordem para cada job
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self.evaluate_arima_order)(order, data_arr) for order in parameter_grid
            )

        else:
            # paralelismo com chunks: multiplas ordens testadas por job
            chunks = [parameter_grid[i:i + chunk_size] for i in range(0, len(parameter_grid), chunk_size)]
            chunk_results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self.evaluate_chunk)(chunk, data_arr) for chunk in chunks
            )
            results = [item for sublist in chunk_results for item in sublist]
        best_result = min(results, key=lambda x: x[1])
        best_order, best_aic = best_result
        
        return best_order, best_aic

    
    def run_arima_pipeline(self, ticker, start_date, end_date, p_lower=0, p_upper=6, q_lower=0, q_upper=6, n_jobs=-1, show_plot=True, save_path=None, backend="loky", figsize=(7, 5), chunk_size=1):
        train_series, test_series = self.prepare_data()  
        best_order, best_aic = self.select_best_order(train_series, lower_bound=p_lower, upper_bound=max(p_upper, q_upper), n_jobs=n_jobs, backend=backend, chunk_size=chunk_size)
        history = np.array(train_series).copy()
        test = np.array(test_series).copy()
        predictions = []

        for i in range(len(test)):
            model = ARIMA(history, order=best_order)
            model_fit = model.fit()
            hat = model_fit.forecast(steps=1)[0]
            predictions.append(hat)
            observed = test[i]
            history = np.concatenate((history, np.array([observed])))

        predictions = np.array(predictions)
        rmse = float(np.sqrt(np.mean((predictions - test) ** 2)))
        mae = float(np.mean(np.abs(predictions - test)))
        mape = float(np.mean(np.abs((test - predictions) / test)) * 100)
        preds_series = pd.Series(predictions, index=test_series.index)
        actual_series = test_series.copy()
        actual_series.index = pd.to_datetime(actual_series.index)
        preds_series.index = pd.to_datetime(preds_series.index)
        last_actual_value = float(test[-1]) if len(test) else None
        last_prediction_value = float(predictions[-1]) if len(predictions) else None
        final_model = ARIMA(history, order=best_order).fit()
        next_prediction_value = float(final_model.forecast(steps=1)[0])

        if show_plot or (save_path is not None):
            set_plot_style()
            plt.figure(figsize=figsize)
            plt.plot(actual_series.index, actual_series.values, label="Preço real", color="green")
            plt.plot(preds_series.index, preds_series.values, label="Estimativas", linestyle="--", color="purple")
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y')) 

            plt.ylabel(r"Preço de fechamento (R\$)")
            plt.xlabel("Data")
            plt.title(f"Estimativas do ARIMA vs. Preço real para o ativo {ticker}")
            plt.legend(loc="best", edgecolor="black")
            plt.tight_layout()
            plt.xticks(rotation=45, ha="right")

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                plt.savefig(save_path, format="pdf", bbox_inches="tight")
            if show_plot:
                plt.show()
            plt.close()

        return {
            "ticker": ticker,
            "order": best_order,
            "aic": best_aic,
            "rmse": rmse,
            "mae": mae,
            "predictions": predictions,
            "test": test,
            "test_dates": test_series.index.values,
            "mape": mape,
            "last_actual_value": last_actual_value,
            "last_prediction_value": last_prediction_value,
            "next_prediction_value": next_prediction_value,
        }
    
    def print_pipeline_evaluation(self, results):
        ticker = results["ticker"]
        best_order = results["order"]
        best_aic = results["aic"]
        rmse = results["rmse"]
        mae = results["mae"]
        mape = results["mape"]

        ######################
        print(f"\n--- Relatório de execução para {ticker} ---")
        print(f"Ordem otimizada do ARIMA: {best_order} (AIC: {best_aic:.2f})")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"MAPE: {mape:.4f}")
        last_actual = results.get("last_actual_value")
        last_pred = results.get("last_prediction_value")
        next_pred = results.get("next_prediction_value")
        if last_actual is not None:
            print(f"Último valor real do conjunto: {last_actual:.4f}")
        if last_pred is not None:
            print(f"Previsão ARIMA para o último ponto: {last_pred:.4f}")
        if next_pred is not None:
            print(f"Previsão ARIMA para o próximo ponto: {next_pred:.4f}")
        ####################
        
        print("\n*** Sumário ***")
        print("Um AIC baixo signfica que a ordem ajustada foi a mais apropriada para a configuração testada.")
        print("O RMSE e MAE são métricas de avaliação de acurácia, quanto menor, mais preciso o modelo foi.")
        print("O MAPE representa a porcentagem média de erros relativo aos valores reais.")
    
    def run_speed_test_span(self, chunk_size=1, show_plot=False, save_path=None):
        print("#" * 66)
        print("Platforma:", platform.platform())
        print("Num. de núcleos (logicos):", multiprocessing.cpu_count())
        print(f"Intervalo: {self.start_date} -> {self.end_date}")
        print(f"Tamanho de chunks: {chunk_size}")
        print("#" * 66)

        P_UPPER = self.p_upper
        Q_UPPER = self.q_upper

        # print("--- Iniciando Teste Serial (n_jobs=1) ---")
        # start_time_serial = time.perf_counter()
        # serial_results = self.run_arima_pipeline(
        #     self.ticker, self.start_date, self.end_date,
        #     p_lower=0, p_upper=P_UPPER,
        #     q_lower=0, q_upper=Q_UPPER,
        #     show_plot=False,
        #     save_path="images/resultado.pdf",
        #     n_jobs=1,
        #     backend="loky",
        #     chunk_size=1
        # )
        # end_time_serial = time.perf_counter()
        # time_serial = end_time_serial - start_time_serial
        # print(f"--- Teste Serial Concluído: {time_serial:.4f} segundos ---")
        # self.print_pipeline_evaluation(serial_results)

        # print("-" * 82)

        print("\n--- Iniciando a execução ---")
        start_time_parallel = time.perf_counter()
        parallel_results = self.run_arima_pipeline(
            self.ticker, self.start_date, self.end_date,
            p_lower=0, p_upper=P_UPPER,
            q_lower=0, q_upper=Q_UPPER,
            show_plot=show_plot,
            save_path=save_path,
            n_jobs=-1,
            backend="loky",
            chunk_size=chunk_size
        )
        end_time_parallel = time.perf_counter()
        time_parallel = end_time_parallel - start_time_parallel
        print(f"--- Execução concluída em: {time_parallel:.4f} segundos ---")
        self.print_pipeline_evaluation(parallel_results)
        print("-" * 82)

        # speedup = time_serial / time_parallel if time_parallel > 0 else float('inf')
        # print("\n--- Relatório de Speedup ---")
        # print(f"Tempo Serial (1 core):      {time_serial:.4f}s")
        # print(f"Tempo Paralelo (all cores): {time_parallel:.4f}s")
        # print(f"Speedup:                    {speedup:.2f}x")
        # print(f"(O processamento paralelo foi {speedup:.2f} vezes mais rápido)")
        # print("#" * 66)
        # print()


