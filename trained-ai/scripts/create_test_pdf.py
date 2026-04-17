from pypdf import PdfWriter
from reportlab.pdfgen import canvas

filename = "proposal_template.pdf"
c = canvas.Canvas(filename)

c.drawString(100, 800, "DOST Form No. 1B - PROPOSAL")
c.drawString(100, 750, "Project Title: AI-Driven Agricultural Yield System for Zamboanga")
c.drawString(100, 730, "(2) Cooperating Agencies")
c.drawString(100, 715, "DA Region 9, LGU Zamboanga City, WMSU")

c.drawString(100, 650, "Duration: 24 months")

c.drawString(100, 600, "BUDGET SUMMARY")
c.drawString(100, 580, "PS: 400,000.00")
c.drawString(100, 560, "MOOE: 500,000.00")
c.drawString(100, 540, "CO: 100,000.00")
c.drawString(100, 520, "Total: 1,000,000.00")

c.save()
print(f"✅ Created dummy file: {filename}")