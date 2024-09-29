#! /bin/python3
import subprocess
import sys
import tempfile


def handle_latex(lines, file_num = 0):
    latex_str = "".join(lines)

    with open(f"temp{file_num}.tex", "wb") as tex_file:
        tex_file.write(latex_str.encode())

    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file.name])


in_latex = False
current_lines = []
file_num = 0

for line in sys.stdin.readlines():
    if line.startswith("\\documentclass"):
        in_latex = True

    if in_latex:
        current_lines.append(line)

    if line.startswith("\\end{document}"):
        in_latex = False
        handle_latex(current_lines, file_num)
        current_lines = []
        file_num += 1
