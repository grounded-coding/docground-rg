
import os
import json
import xlsxwriter
from pathlib import Path
from enum import Enum

class Column(Enum):
    MODEL_NAME = 0
    SIZE = 1
    FINE_TUNED = 2
    LORA = 3
    DOCSORT = 4
    BLEU = 5
    METEOR = 6
    ROUGE_L = 7
    RANK = 8

def extract_metrics(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
        return data["generation"]["bleu"], data["generation"]["meteor"], data["generation"]["rouge_l"]

def write_data_to_excel(file_names, base_path):
    workbook = xlsxwriter.Workbook("metrics_table.xlsx")
    worksheet = workbook.add_worksheet()

    headers = ["Model Name", "Size", "Fine-tuned", "LoRA", "DocSort", "BLEU-1", "MT", "R-L", "Rank"]
    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header)

    percent_format = workbook.add_format({"num_format": "0.00%"})

    for row, file_name in enumerate(file_names, start=1):
        model_name = file_name.split("--")[1]
        file_path = Path(base_path) / (file_name + ".json")
        bleu, meteor, rouge_l = extract_metrics(file_path)

        worksheet.write(row, Column.MODEL_NAME.value, model_name)
        worksheet.write(row, Column.BLEU.value, bleu)
        worksheet.write(row, Column.METEOR.value, meteor)
        worksheet.write(row, Column.ROUGE_L.value, rouge_l)
        worksheet.write(row, Column.RANK.value, bleu + meteor + rouge_l)

    worksheet.set_column(Column.BLEU.value, Column.RANK.value, None, percent_format)
    workbook.close()

file_names = ["dstc_metrics--rg.llama-13b-peft-noemb-0712182215", 
              "dstc_metrics--rg.llama-7b-peft",
              "dstc_metrics--rg.oasst-12b-peft-noemb"]
base_path = "/u/nils.hilgers/setups/dialog_setup/output/dstc/dstc11_track5/generation/validation"
write_data_to_excel(file_names, base_path)