# custom exception handling
import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    err_msg = str(error)
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(filename, line_no, err_msg)

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    