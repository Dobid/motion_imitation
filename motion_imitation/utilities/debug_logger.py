import logging

# logd = logging
FORMAT = '%(pathname)s:%(lineno)d - %(funcName)s()\n> %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logd = logging.getLogger('print_debug')