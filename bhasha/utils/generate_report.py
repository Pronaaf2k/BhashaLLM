from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'BhashaLLM Project Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(md_file, output_file):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue
            
        # Simplistic markdown parsing
        try:
            # Replace common markdown formatting
            line = line.replace('**', '') # bold
            line = line.replace('*', '')  # italic
            line = line.replace('`', '')  # code

            # Handle headers
            if line.startswith('# '):
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 16)
                text = line[2:]
                pdf.cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
                pdf.set_font("Arial", '', 11)
            elif line.startswith('## '):
                pdf.ln(4)
                pdf.set_font("Arial", 'B', 14)
                text = line[3:]
                pdf.cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
                pdf.set_font("Arial", '', 11)
            elif line.startswith('### '):
                pdf.ln(3)
                pdf.set_font("Arial", 'B', 12)
                text = line[4:]
                pdf.cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
                pdf.set_font("Arial", '', 11)
            elif line.startswith('- '):
                pdf.set_font("Arial", '', 11)
                pdf.cell(5) # Indent
                text = f"- {line[2:]}"
                # Safe encode
                safe_text = text.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, safe_text)
            elif line.startswith('|'):
                # Simple table row handling
                pdf.set_font("Courier", '', 9)
                safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 4, safe_line)
                pdf.set_font("Arial", '', 11)
            else:
                # Normal paragraph
                safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, safe_line)
        except Exception as e:
            print(f"Error processing line: {line[:20]}... -> {e}")

    pdf.output(output_file)
    print(f"PDF generated: {output_file}")

if __name__ == "__main__":
    # Ensure paths work from workspace root
    md_path = "BhashaLLM/FINAL_RESEARCH_REPORT.md"
    pdf_path = "BhashaLLM/BhashaLLM_Report.pdf"
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
    else:
        generate_pdf(md_path, pdf_path)
