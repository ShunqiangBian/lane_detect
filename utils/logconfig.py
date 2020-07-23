import logging,os

rdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #获取上级目录的绝对路径
log_dir = rdir + '/record.log'
def getLogger():
    """
    ==============
    log-print handle
    ==============
    :return logger: log object
    """
    fm = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')  #设置日志格式
    fh = logging.FileHandler(log_dir,encoding='utf-8') #创建一个文件流并设置编码utf8
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fm)
    logger = logging.getLogger() #获得一个logger对象，默认是root
    logger.setLevel(logging.DEBUG)  #设置最低等级debug
    logger.addHandler(fh) #把文件流添加进来，流向写入到文件
    logger.addHandler(ch)
    fh.setFormatter(fm) #把文件流添加写入格式
    return logger
"""
============================
test
============================
log=get_logger()
log.debug('test debug')
log.info('test info')
log.warning('test warning')
log.error('test error')
log.critical('test critical')
log.info("a:%s\n b:%s\n c:%d\n\n",'a','b',10)
"""