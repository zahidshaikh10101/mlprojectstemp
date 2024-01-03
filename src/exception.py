import sys
from logger import logging


def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()  
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_message
     

class CustomException(Exception):
    def __init__(self, error_message, exception, sys):
        super().__init__(error_message, exception, sys)
        self.error_message = error_message
        self.exception = exception
        self.sys = sys

    def __str__(self) -> str:
        return f"{self.error_message}\nException: {self.exception}\nSys: {self.sys}"
        
if __name__=="__main__":

    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e,sys)