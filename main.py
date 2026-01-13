import argparse
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from Arima import Arima

TEST_PRESETS = {
    "small": {
        "ticker": "PETR4.SA",
        "start_date": "2024-12-16",
        "end_date": "2025-06-16",
        "p_upper": 5,
        "q_upper": 5,
        "chunk_size": 10,
    },
    "medium": {
        "ticker": "PETR4.SA",
        "start_date": "2022-06-16",
        "end_date": "2025-06-16",
        "p_upper": 5,
        "q_upper": 5,
        "chunk_size": 10,
    },
    "long": {
        "ticker": "PETR4.SA",
        "start_date": "2017-06-16",
        "end_date": "2025-06-17",
        "p_upper": 5,
        "q_upper": 5,
        "chunk_size": 10,
    },
}

@dataclass
class TestParams:
    ticker: str
    start_date: str
    end_date: str
    p_upper: int
    q_upper: int
    chunk_size: int
    log_file: Optional[str]
    capture_output: bool
    show_plot: bool
    save_image: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa a pipeline com parâmetros personalizáveis.")
    parser.add_argument("--preset", choices=TEST_PRESETS.keys(), default="small", help="Carregar parâmetros padrão.")
    parser.add_argument("--ticker", help="Símbolo do ticker do Yahoo Finance.")
    parser.add_argument("--start-date", help="Data de início da série temporal (AAAA-MM-DD).")
    parser.add_argument("--end-date", help="Data final da série temporal (AAAA-MM-DD).")
    parser.add_argument("--p-upper", type=int, help="Limite superior para p em ARIMA(p,d,q).")
    parser.add_argument("--q-upper", type=int, help="Limite superior para q em ARIMA(p,d,q).")
    parser.add_argument("--chunk-size", type=int, help="Tamanho do chunk paralelo para o grid search.")
    parser.add_argument("--log-file", help="Diretório para salvar logs capturados.")
    parser.add_argument("--no-log", action="store_true", help="Não gravar logs de execução.")
    parser.add_argument("--no-capture", action="store_true", help="Exibir saída no console sem salvá-las.")
    parser.add_argument("--show-plot", action="store_true", help="Exibir o gráfico ao final da execução.")
    parser.add_argument("--save-image", help="Caminho do arquivo para salvar a figura gerada (ex: images/plot.png).")
    
    return parser.parse_args()

def build_params(args: argparse.Namespace):
    preset = TEST_PRESETS[args.preset]

    def value(key):
        return getattr(args, key, None) if getattr(args, key, None) is not None else preset[key]

    ticker = value("ticker")
    log_file = None if args.no_log else (args.log_file or f"{ticker}.txt")

    return TestParams(
        ticker=ticker,
        start_date=value("start_date"),
        end_date=value("end_date"),
        p_upper=value("p_upper"),
        q_upper=value("q_upper"),
        chunk_size=value("chunk_size"),
        log_file=log_file,
        capture_output=not args.no_capture,
        show_plot=args.show_plot,
        save_image=args.save_image,
    )

def run_test(params: TestParams):
    runner = Arima(
        ticker=params.ticker,
        start_date=params.start_date,
        end_date=params.end_date,
        p_upper=params.p_upper,
        q_upper=params.q_upper,
    )

    print(f"Iniciando testes para {params.ticker}")
    buffer = io.StringIO()
    with contextlib.ExitStack() as stack:
        if params.capture_output:
            stack.enter_context(contextlib.redirect_stdout(buffer))
            stack.enter_context(contextlib.redirect_stderr(buffer))
        runner.run_speed_test_span(
            chunk_size=params.chunk_size,
            show_plot=params.show_plot,
            save_path=params.save_image,
        )

    if params.capture_output and params.log_file:
        Path(params.log_file).write_text(buffer.getvalue(), encoding="utf-8")
        print(f"Saved all terminal output to '{params.log_file}'.")


if __name__ == "__main__":
    cli_args = parse_args()
    test_params = build_params(cli_args)
    run_test(test_params)