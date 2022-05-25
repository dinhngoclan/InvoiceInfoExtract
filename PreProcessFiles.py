# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:05:24 2022

@author: laptop
"""

#=======================Convert .xls to .xlsx========================
import os, glob
import win32com.client as win32
def convert_xls2xlsx(path):
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    filenames = glob.glob(path + "\*.xls")
    for file in filenames:
        #os.path.abspath(r".\TCDKO-2204-08 120422 (Kumho-2309)-01 NYLON-BUSAN-XKD.XLS")
        wb = excel.Workbooks.Open(file)
        wb.SaveAs(file+'x', FileFormat = 51)    #FileFormat = 51 is for .xlsx extension
        wb.Close()                               #FileFormat = 56 is for .xls extension
        wb = None
    excel.Application.Quit()

convert_xls2xlsx(os.path.abspath(r".\InvoicesDirectory"))

#=======================Hide GridLine in .xlsx ========================
##OK OK
import openpyxl
from openpyxl.styles.alignment import Alignment
# Load workbook
import pyexcel
#pyexcel.save_book_as(file_name='.\TCDKO-2204-08 120422 (Kumho-2309)-01 NYLON-BUSAN-XKD.XLS', dest_file_name='.\TCDKO-2204-08 120422 (Kumho-2309)-01 NYLON-BUSAN-XKD.XLSX')
def hide_gridline_in_xlsx(path):
    filenames = glob.glob(path + "\*.xlsx")
    for file in filenames:
        wb = openpyxl.load_workbook(file)
        # Loop through all cells in all worksheets
        for sheet in wb.worksheets:
            sheet.sheet_view.showGridLines = False
            sheet.sheet_view.view='normal'
            sheet.paper_size_code = 1
            #sheet.alignment = Alignment(horizontal="center")
        # Save workbook
        wb.save(file)
 
hide_gridline_in_xlsx(os.path.abspath(r".\InvoicesDirectory"))

# =============================================================================
# #======================Convert .xlsx to .png ========================
# #OK OK
# import excel2img, glob
# def xlsx2img(path):
#     filenames = glob.glob(path + "\*.xlsx")
#     for file in filenames:
#         excel2img.export_img(file, file+'.png', "INVOICE", None)
# 
# xlsx2img(os.path.abspath(r".\InvoicesDirectory"))
# =============================================================================

#======================Convert .xlsx to .pdf ========================
from win32com import client
def xlsx2pdf(path):
    filenames = glob.glob(path + "\*.xlsx")
    xlApp = client.Dispatch("Excel.Application")
    xlApp.Visible = 0
    for file in filenames:
        books = xlApp.Workbooks.Open(file)
        ws = books.Worksheets['INVOICE']
        #ws.Range("A1", "P100").HorizontalAlignment = 2  #align LEFT 
        #ws.Range("A1", "P80").HorizontalAlignment = 1 #align RIGHT
        try:
            ws.SaveAs(file+'.pdf', FileFormat=57)
        except Exception as e:
            print("Failed to convert")
            print(str(e))
        finally:
            books.Close(True)
    xlApp.Quit()

xlsx2pdf(os.path.abspath(r".\InvoicesDirectory"))
#======================Convert .pdf to .png ========================
from pdf2image import convert_from_path
filenames = glob.glob('.\InvoicesDirectory' + "\*.pdf")
for file in filenames:
    images = convert_from_path(file,poppler_path=r'.\poppler-22.01.0\Library\bin')
    images[0].save(file+'.png', "PNG")
