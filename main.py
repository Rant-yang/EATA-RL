import schedule
from retrying import retry

# from data import DataWorker, DataStorage
# from preprocess import Preprocessor
from predict import Predictor
from evaluate import Evaluator

from visualize import WebServer
from globals import MAIN_PATH
import pysnooper

def test_webserver():
    ws = schedule.get_jobs("webserver")
    if ws is None:
        print("web server down, restarting")
        WebServer().run()

def predict():
    p = Predictor()
    a = p.latest_actions()
    a.save_action()

def evaluate():
    e = Evaluator()
    e.save_evaluated()


if __name__ == "__main__":
    schedule.every().day.at("18:00").do(predict)    # 读preprocessed数据，预处理，预测，并写入predicted
    schedule.every().day.at("18:00").do(evaluate)    # 读predicted数据，预处理，预测，并写入evaluated
    schedule.every().hour.do(test_webserver)   # 读predicted，进行evaluate，增加一些字段，供visualizer读取并可视化


