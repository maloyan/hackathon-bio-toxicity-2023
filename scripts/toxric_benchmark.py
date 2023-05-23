import requests
import json
import numpy as np
import pandas as pd


def get_data(url):
    response = requests.get(url)

    if response.status_code == 200:
        json_data = json.loads(response.text)
        # print(json.dumps(json_data, indent=2))
    json_data = json_data["data"][0]
    res = np.array(json_data['meanBarData']).T.max(axis=1)
    df = pd.DataFrame({
        "title": json_data['title'],
        "benchmark_name": json_data['categoryData'],
        "metric": json_data['yAxisName'],
        "baseline": res
    })
    return df

benchmarks = [
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=1_3&type=R2",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=3_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=288_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=292_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=5_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=4_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=7_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=6_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=103_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=9_3&type=R2",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=287_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=10_3",
    "https://toxric.bioinforai.tech/jk/BenchmarkController/toxicity/BenchmarksForAlgorithms?nodeId=8_3"
]

dfs = [get_data(i) for i in benchmarks]

pd.concat(dfs).reset_index().to_csv("benchmark/benchmark_toxric.csv", index=None)