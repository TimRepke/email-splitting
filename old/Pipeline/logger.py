import logging


def init(options):
    logging.addLevelName(5, 'TRACE')
    logging.addLevelName(3, 'MICROTRACE')
    log_format = '%(asctime)s - [%(module)s] %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.getLevelName(options.log_level),
                        format=log_format, datefmt='%H:%M:%S')

    if options.log_file:
        log('INFO', 'Writing log to file: %s at level %s', options.log_file, options.log_file_level)
        hdlr = logging.FileHandler(options.log_file, mode='w')
        hdlr.setLevel(logging.getLevelName(options.log_file_level))
        hdlr.setFormatter(logging.Formatter(log_format, datefmt='%H:%M:%S'))
        logging.root.addHandler(hdlr)
    else:
        log('INFO', 'No log file specified.')


def loglevel(as_int=False):
    level = logging.root.level
    if as_int:
        return level
    return logging.getLevelName(level)


def is_lower(than, le=True):
    if le:
        return loglevel(as_int=True) <= logging.getLevelName(than)
    return loglevel(as_int=True) < logging.getLevelName(than)


def log(lvl, msg, *args, **kwargs):
    logging.log(logging.getLevelName(lvl), msg, *args, **kwargs)
