import logging
# 配置日志系统隐藏冗余输出
def configure_logging():
    logging.basicConfig(level=logging.ERROR)
    xgb_logger = logging.getLogger('xgboost')
    xgb_logger.setLevel(logging.ERROR)
    lgb_logger = logging.getLogger('lightgbm')
    lgb_logger.setLevel(logging.ERROR)