## The sys module in Python provides access to variables and functions that interact
#  closely with the Python interpreter and runtime environment
import sys 
def error_message_detail(error,error_details:sys):
    _,_,ecx_tb = error_details.exc_info()
    file_name = ecx_tb.tb_frame.f_code.co_filename
    error_message = "Error in python script namr[{0}] lin enumber[{1}] error measage[{3}]".format(
        file_name,ecx_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details=error_details)
    def __str__(self):
        return self.error_message
 

