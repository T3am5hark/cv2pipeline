from datetime import datetime


def filename_timestamp(dt=datetime.now()):

    return dt.strftime('%Y-%m-%d_%H:%M:%S')

