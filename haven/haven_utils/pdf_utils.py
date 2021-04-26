import pandas as pd
import pylab as plt


def merge_pdfs(fname_list, output_name="output.pdf"):
    from PyPDF2 import PdfFileReader, PdfFileMerger
    import PyPDF2
    import img2pdf

    from pdfrw import PdfReader, PdfWriter

    writer = PdfWriter()
    for inpfn in fname_list:
        decrypt_pdf(inpfn, output_name)
        writer.addpages(PdfReader(output_name).pages)
    writer.write(output_name)


def decrypt_pdf(input_name, output_name):
    import pikepdf

    pdf = pikepdf.open(input_name)
    pdf.save(output_name)


def images_to_pdf(fname_list, output_name="output.pdf"):
    from fpdf import FPDF

    pdf = FPDF()
    # imagelist is the list with all image filenames
    for image in fname_list:
        pdf.add_page()
        pdf.image(image, x=0, y=0, w=610, h=297)
    pdf.output(output_name, "F")
